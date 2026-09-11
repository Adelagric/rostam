// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	hraft "github.com/hashicorp/raft"
)

// ISR SHRINK driver (primary-driven, meta-leader-committed, floor-
// enforced). GROW is a separate, deferred increment.
//
// A full-ISR write stalls forever on a dead backup (ErrPipelineStalled): commit
// requires EVERY ISR member to ack, and a dead member never does. The primary has
// the freshest signal for this (its engine's per-peer replication-failure
// counter), so — exactly like the 4b liveness beacon — the shrink is PRIMARY-
// driven: the primary detects the wedge, asks the meta leader to commit a smaller
// ISR (never below minISR), then, once it observes that committed shrink in its
// own MetaFSM, calls Engine.ShrinkISR to live-re-evaluate the in-flight writes so
// the stalled pipeline resumes — provably without losing any acked write (the
// no-acked-loss / no-false-commit proofs live on Engine.ShrinkISR).
//
// THE minISR FLOOR IS ENFORCED HERE, FIRST. This driver refuses to even request a
// shrink that would drop the ISR below the durability floor — the shard then stays
// stalled (H3: choose unavailability over durability loss). decidePBShrink is the
// pure, floor-enforcing decision; pbShrinkDriver.tick is its plumbing. The FSM apply
// of OpSetShardISR ALSO rejects a below-floor set as a structural backstop (see
// meta_fsm.go), so a floor breach needs BOTH this driver's check to be bypassed AND
// the FSM guard to be absent — the driver is the first line, not the only one.

// pbShrinkRequest is a decided ISR shrink: at epoch, set shard's ISR to newISR
// (the current ISR minus the confirmed-dead members). The caller commits it via
// OpSetShardISR (epoch-guarded so a fenced ex-primary's stale request is a no-op).
type pbShrinkRequest struct {
	shardID int
	epoch   uint64
	newISR  []string
}

// decidePBShrink decides whether to shrink one shard's ISR to drop its stalled
// members, ENFORCING THE minISR FLOOR as the first line (the FSM apply enforces it
// too, as a backstop — see meta_fsm.go's OpSetShardISR). It returns ok=false —
// no shrink — when there is nothing removable, or when removing the dead members
// would take the ISR below minISR (the shard then stays stalled: unavailability
// over durability, H3). The PRIMARY is never a removal candidate (a dead primary
// is the failover path's job, not shrink's). Pure and order-preserving.
func decidePBShrink(shardID int, epoch uint64, primary string, isr, stalled []string, minISR int) (pbShrinkRequest, bool) {
	if len(stalled) == 0 || len(isr) == 0 {
		return pbShrinkRequest{}, false
	}
	drop := make(map[string]struct{}, len(stalled))
	for _, s := range stalled {
		if s == primary {
			continue // never drop the primary via shrink — that is failover's domain
		}
		drop[s] = struct{}{}
	}
	newISR := make([]string, 0, len(isr))
	for _, m := range isr {
		if _, dead := drop[m]; dead {
			continue
		}
		newISR = append(newISR, m)
	}
	if len(newISR) == len(isr) {
		return pbShrinkRequest{}, false // nothing in `stalled` was actually in the ISR
	}
	if len(newISR) < minISR {
		// FLOOR: refuse. Removing the dead member(s) would breach the durability
		// floor, so the shard stays stalled rather than risk losing acked writes.
		return pbShrinkRequest{}, false
	}
	return pbShrinkRequest{shardID: shardID, epoch: epoch, newISR: newISR}, true
}

// submitSetShardISR commits (or forwards) an ISR update. On the meta leader it
// applies locally via ApplySetShardISR; on a follower it forwards to the leader
// over the __pb_set_isr__ admin op — mirroring submitShardLeaseRenew exactly, so
// a follower-hosted primary's shrink still reaches consensus. The leader's handler
// applies it locally (never re-entering this branch), so there is no forwarding
// loop. The FSM apply is epoch-guarded: a stale-epoch request (a fenced
// ex-primary that has since been superseded) is a no-op.
func (n *Node) submitSetShardISR(shardID int, epoch uint64, isr []string, timeout time.Duration) error {
	if n.meta == nil {
		return errNoMeta
	}
	if n.meta.Raft.State() != hraft.Leader {
		addr := n.metaLeaderServerAddr()
		if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
			return fmt.Errorf("cluster: submitSetShardISR: no meta-Raft leader yet")
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			return err
		}
		args, err := gobEncode(pbSetISRReq{ShardID: shardID, Epoch: epoch, ISR: isr})
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		_, err = cl.Call(ctx, opPBSetISRName, args)
		return err
	}
	return n.meta.ApplySetShardISR(shardID, epoch, isr, timeout)
}

// pbShrinkDriver is the per-node ISR-shrink driver goroutine. It mirrors
// leaseKeeper / pbBeacon / pbFailover lifecycle (start/stop/run/tick) and is
// started ONLY when PBAutoFailover is on — when off, no driver runs, no
// OpSetShardISR shrink is ever logged, and the static cluster's replicated state
// stays byte-identical.
type pbShrinkDriver struct {
	node          *Node
	interval      time.Duration
	submitTimeout time.Duration
	minFailures   int // consecutive-failure threshold before a backup is "dead enough"
	minISR        int // the durability floor (enforced by decidePBShrink)

	// baseline is the per-node ISR-reconcile baseline SHARED with the grow driver
	// (pb_grow.go), so a shrink and a grow at the same epoch never misread each
	// other's committed change: a grow widening updates the baseline so a later
	// re-narrow is still seen as a strict subset here, and vice versa.
	baseline *pbISRReconcile

	// abortStats is the shrink-abort observability: rate-limited
	// logging of a stuck OpSetShardISR submit (log the transition, not every
	// tick). See pbAbortTracker.
	abortStats *pbAbortTracker

	mu      sync.Mutex
	started bool
	done    chan struct{}
	stopped chan struct{}
}

// newPBShrinkDriver constructs the shrink driver for node n. minFailures is the
// per-peer consecutive-replication-failure count at which the driver treats a
// backup as dead and requests its removal — it MUST be set well above one RTT's
// worth of transient blips so a momentary hiccup never shrinks a healthy ISR.
func newPBShrinkDriver(n *Node, interval, submitTimeout time.Duration, minFailures, minISR int, baseline *pbISRReconcile) *pbShrinkDriver {
	return &pbShrinkDriver{
		node:          n,
		interval:      interval,
		submitTimeout: submitTimeout,
		minFailures:   minFailures,
		minISR:        minISR,
		baseline:      baseline,
		abortStats:    newPBAbortTracker(),
	}
}

// start spawns the driver goroutine (no-op if already started).
func (d *pbShrinkDriver) start() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.started {
		return
	}
	d.started = true
	d.done = make(chan struct{})
	d.stopped = make(chan struct{})
	go d.run()
}

// stop signals the goroutine to exit and blocks until it has. Safe without a
// prior start and safe to call repeatedly.
func (d *pbShrinkDriver) stop() {
	d.mu.Lock()
	if !d.started {
		d.mu.Unlock()
		return
	}
	done := d.done
	stopped := d.stopped
	d.started = false
	d.mu.Unlock()

	select {
	case <-done:
	default:
		close(done)
	}
	<-stopped
}

func (d *pbShrinkDriver) run() {
	defer close(d.stopped)
	ticker := time.NewTicker(d.interval)
	defer ticker.Stop()
	for {
		select {
		case <-d.done:
			return
		case <-ticker.C:
			d.tick()
		}
	}
}

// tick runs one shrink evaluation over every shard this node currently primaries.
// For each such shard it does TWO independent things:
//
//  1. RECONCILE (observe → apply): if the committed ISR in the local MetaFSM is a
//     strict narrowing of what the driver last handed the engine (at the same
//     epoch), call Engine.ShrinkISR to live-re-evaluate the in-flight writes. This
//     is the step that un-wedges a stalled pipeline once the shrink has committed.
//
//  2. DETECT → REQUEST: read the engine's wedge signal (StalledPeers); if a dead
//     backup can be dropped while keeping |ISR| >= minISR (decidePBShrink enforces
//     the floor), forward an OpSetShardISR removing it. Its commit lands in the
//     FSM by a later tick, where step (1) then applies it to the engine.
//
// Both steps are best-effort and idempotent: Engine.ShrinkISR re-checks epoch+lease
// and only narrows; OpSetShardISR is epoch-guarded and monotone at the FSM.
func (d *pbShrinkDriver) tick() {
	n := d.node
	nodeID := n.cfg.NodeID
	if n.meta == nil {
		return
	}
	st := n.meta.FSM.State()
	for shardID, eng := range n.pbEngines {
		if eng == nil {
			continue
		}
		if st.ShardPrimary[shardID] != nodeID {
			// Not this node's primary (never was, or a failover moved it). Forget any
			// stale baseline so a future re-primary at a new epoch re-establishes it.
			d.baseline.forget(shardID)
			d.abortStats.clear(shardID)
			continue
		}
		epoch := st.ShardEpoch[shardID]
		isr := st.ShardISR[shardID]
		// Only drive shrink for the epoch the engine actually holds as primary. If
		// the engine has not yet adopted this committed epoch (a just-committed
		// promotion still replicating in), skip — ShrinkISR would no-op anyway.
		if eng.Epoch() != epoch {
			continue
		}

		// (1) RECONCILE committed narrowings into the live engine. Uses the SHARED
		// baseline (also updated by the grow driver) so a widen-then-narrow at the
		// same epoch is still detected here as a strict subset.
		base, seen := d.baseline.get(shardID)
		switch {
		case !seen || base.epoch != epoch:
			// First sight of this shard at this epoch: establish the baseline WITHOUT
			// re-evaluating. A fresh epoch starts with a clean pipeline (no in-flight
			// records to narrow), and new writes already read this ISR via ctrl.ISR.
			d.baseline.set(shardID, epoch, isr)
		case isStrictSubset(isr, base.isr):
			// The committed ISR shrank at this epoch: live re-evaluate the in-flight
			// writes so the stalled head can commit against the smaller set.
			eng.ShrinkISR(epoch, isr)
			d.baseline.set(shardID, epoch, isr)
		}

		// (2) DETECT a dead backup and REQUEST its removal (floor-enforced).
		stalled := eng.StalledPeers(d.minFailures)
		if len(stalled) == 0 {
			d.abortStats.clear(shardID) // healthy: no stalled peer this tick
			continue
		}
		req, ok := decidePBShrink(shardID, epoch, nodeID, isr, stalled, d.minISR)
		if !ok {
			continue // nothing removable, or the floor refuses (shard stays stalled)
		}
		// Best-effort: a transient submit failure (no meta leader yet, slow forward)
		// is retried next tick; the write path stays stalled until the shrink commits.
		// Rate-limited log + counter on failure: log the transition, not
		// every tick, since this driver ticks sub-second (pbShrinkTickInterval).
		if err := n.submitSetShardISR(req.shardID, req.epoch, req.newISR, d.submitTimeout); err != nil {
			if d.abortStats.record(shardID, "", "submit_failed") {
				slog.Warn("pb: shrink submit failed", "component", "cluster", "shard", shardID, "epoch", epoch, "dropped", stalled, "err", err)
			}
			continue
		}
		d.abortStats.clear(shardID)
	}
}

// isStrictSubset reports whether every element of sub is in super AND sub is
// strictly smaller (|sub| < |super|) — i.e. sub is super with at least one member
// removed and nothing added. Used to detect a committed ISR NARROWING (never a
// widening — grow is out of scope). Both slices are small (an ISR), so the O(n·m)
// membership scan is fine.
func isStrictSubset(sub, super []string) bool {
	if len(sub) >= len(super) {
		return false
	}
	for _, s := range sub {
		found := false
		for _, p := range super {
			if s == p {
				found = true
				break
			}
		}
		if !found {
			return false // sub has a member super lacks — not a pure narrowing
		}
	}
	return true
}
