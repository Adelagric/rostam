// SPDX-License-Identifier: Apache-2.0

package pbisr

import "context"

// pipelineWindow bounds the primary write pipeline: at most this many writes may
// be in flight (uncommitted) per shard, measured as lastSeq - committed. It
// hard-bounds the OH2 uncommitted tail (a wedged shard fails at most W writes
// then refuses admission) — see PIPELINE-REDESIGN §4.
const pipelineWindow = 256

// Compile-time assertion: pipelineWindow must never exceed the catch-up ring
// capacity, so every in-flight (uncommitted) write is always replayable from the
// backlog (PIPELINE-REDESIGN §4). If pipelineWindow > defaultRingCapacity the
// constant subtraction underflows uint and the build fails here.
const _ = uint(defaultRingCapacity - pipelineWindow)

// maxInlinePeers sizes the inflight record's inline pending set. Typical ISRs
// are tiny (RF=2 -> 1 backup, RF=3 -> 2), so the set lives in the record with
// NO per-write map allocation (2026-07-22 alloc profile: the per-write pending
// map was the single largest allocation on the write path). An ISR with more
// backups than this falls back to a heap slice — correct, just not alloc-free.
const maxInlinePeers = 4

// inflight is one write awaiting full-ISR commit. It is the per-record commit
// state: the propose-time peers that still owe an exact (epoch,seq) ack, plus a
// single-shot resolution latch. Guarded by e.mu.
type inflight struct {
	epoch uint64
	seq   uint64
	// The pending set: peers[:npending] (order-insensitive; removal swaps with
	// the last live entry). overflow replaces the inline array entirely when
	// the propose-time backup count exceeds maxInlinePeers.
	npending int
	peers    [maxInlinePeers]string
	overflow []string
	resolved bool       // exactly-once resolution latch
	err      error      // nil == committed; non-nil == failed/unknown
	doneCh   chan error // buffered(1); the waiting Propose reads here
}

// pendingCount reports how many peers still owe an ack. Caller holds e.mu.
func (r *inflight) pendingCount() int {
	if r.overflow != nil {
		return len(r.overflow)
	}
	return r.npending
}

// removePending drops peer from the pending set (no-op if absent — H6 exact
// matching happens at the caller). Caller holds e.mu.
func (r *inflight) removePending(peer string) {
	if r.overflow != nil {
		for i, p := range r.overflow {
			if p == peer {
				r.overflow[i] = r.overflow[len(r.overflow)-1]
				r.overflow = r.overflow[:len(r.overflow)-1]
				return
			}
		}
		return
	}
	for i := 0; i < r.npending; i++ {
		if r.peers[i] == peer {
			r.npending--
			r.peers[i] = r.peers[r.npending]
			r.peers[r.npending] = ""
			return
		}
	}
}

// pendingPeers returns the live pending set (test/introspection helper; the
// returned slice must not be retained across mu releases). Caller holds e.mu.
func (r *inflight) pendingPeers() []string {
	if r.overflow != nil {
		return r.overflow
	}
	return r.peers[:r.npending]
}

// registerInflightLocked pushes a new record onto the seq-ordered FIFO and
// returns it. Seqs are dense and monotonic, so the record's FIFO offset is
// seq - baseSeq; the physical ring slot is (inflightHead + offset) % cap.
// Registering a seq also advances lastSeq (it is an assigned seq) so the
// admission window (lastSeq - committed) accounts for it. Caller holds e.mu.
func (e *Engine) registerInflightLocked(epoch, seq uint64, peers []string) *inflight {
	rec := &inflight{
		epoch:  epoch,
		seq:    seq,
		doneCh: make(chan error, 1),
	}
	if len(peers) <= maxInlinePeers {
		rec.npending = len(peers)
		copy(rec.peers[:], peers)
	} else {
		rec.overflow = append([]string(nil), peers...)
	}

	if e.inflightRing == nil {
		e.inflightRing = make([]*inflight, pipelineWindow)
		e.inflightHead = 0
	}
	if e.inflightN == 0 {
		// Empty FIFO: this record becomes the head; rebase the FIFO on its seq.
		e.baseSeq = seq
	} else {
		// Seqs are dense and monotonic, so the next registered seq MUST be exactly
		// one past the current tail (baseSeq + inflightN). A violation means a seq
		// was skipped or reused under writeMu — the dense-index ack routing would
		// silently misroute — so fail loud rather than corrupt commit state.
		if seq != e.baseSeq+uint64(e.inflightN) {
			panic("pbisr: non-dense in-flight seq registration")
		}
		if e.inflightN == len(e.inflightRing) {
			e.growInflightRingLocked()
		}
	}

	idx := (e.inflightHead + e.inflightN) % len(e.inflightRing)
	e.inflightRing[idx] = rec
	e.inflightN++
	if seq > e.lastSeq {
		e.lastSeq = seq
		// The seq watermark and its epoch move together — a seq without
		// the epoch it was assigned under is a position, not a history identity, and
		// appliedFrontierLocked reads the pair.
		e.lastSeqEpoch = epoch
	}
	return rec
}

// growInflightRingLocked doubles the in-flight ring, re-laying records out in
// FIFO order starting at index 0. The window bounds inflightN to pipelineWindow
// in steady state, so growth is a defensive backstop, not a hot path. Caller
// holds e.mu.
func (e *Engine) growInflightRingLocked() {
	oldCap := len(e.inflightRing)
	grown := make([]*inflight, oldCap*2)
	for i := 0; i < e.inflightN; i++ {
		grown[i] = e.inflightRing[(e.inflightHead+i)%oldCap]
	}
	e.inflightRing = grown
	e.inflightHead = 0
}

// inflightAtLocked returns the record for seq, or nil if no such record is in
// flight (already popped, or never registered). O(1) via the dense-seq index.
// Caller holds e.mu.
func (e *Engine) inflightAtLocked(seq uint64) *inflight {
	if e.inflightN == 0 || seq < e.baseSeq {
		return nil
	}
	offset := seq - e.baseSeq
	if offset >= uint64(e.inflightN) {
		return nil
	}
	return e.inflightRing[(e.inflightHead+int(offset))%len(e.inflightRing)]
}

// popInflightHeadLocked removes the FIFO head and advances baseSeq. Because
// seqs are dense, the new head's seq is exactly baseSeq+1. Caller holds e.mu.
func (e *Engine) popInflightHeadLocked() {
	e.inflightRing[e.inflightHead] = nil
	e.inflightHead = (e.inflightHead + 1) % len(e.inflightRing)
	e.inflightN--
	e.baseSeq++
}

// ackInflightLocked applies one backup ack to its record with LITERAL H6
// matching (byte-identical to engine.go:355): the peer is removed from the
// record's pending set iff the ack is OK and its (epoch,seq) match the record
// exactly. A liveness signal or an ack for a different seq/epoch is ignored.
// Then it drives the commit sweep. Caller holds e.mu.
func (e *Engine) ackInflightLocked(peer string, ack AckMsg) {
	rec := e.inflightAtLocked(ack.Seq)
	if rec != nil && ack.OK && ack.Epoch == rec.epoch && ack.Seq == rec.seq {
		rec.removePending(peer)
	}
	e.sweepLocked()
}

// sweepLocked advances the commit frontier from the FIFO head. It pops the head
// while it is resolved and commits a full-acked head — but ONLY behind the
// commit-time lease fence. It is the single commit decision point, driven by
// acks, timeouts, and epoch/lease changes. Caller holds e.mu.
//
// The commit-time lease fence (PIPELINE-REDESIGN Q5, strengthening P3/OH1): a
// client-visible commit is issued only under a still-valid lease for the
// record's own epoch. Otherwise the record fails as unknown-outcome, so no
// commit at epoch E can be concurrent with or later than the existence of E+1.
func (e *Engine) sweepLocked() {
	for e.inflightN > 0 {
		head := e.inflightRing[e.inflightHead]
		switch {
		case head.resolved:
			// Already resolved (timeout / flush / prior fence). Only FAILED
			// records are ever resolved-but-unpopped — a committed head is
			// popped in the same iteration it is resolved (below). So pop it
			// WITHOUT advancing committed; a later full-ISR commit exposes this
			// applied-but-untimed write transitively (P7/P10).
			e.popInflightHeadLocked()
		case head.pendingCount() == 0 || e.commitLevel == CommitPrimary:
			// Committable: either the FULL ISR has acked (P6, CommitFullISR), or
			// the engine runs CommitPrimary and the head is locally applied (it
			// always is once registered) so it commits without waiting for backup
			// acks. EITHER way the commit is issued ONLY behind the lease fence
			// (Q5/OH1): a fenced primary must not ack even on local apply. Under
			// CommitPrimary the record's still-pending backup acks arrive later
			// and no-op (the record is already popped).
			if e.leaseEpoch == head.epoch && e.epoch == head.epoch && e.now() < e.leaseExpiry {
				e.markCommittedLocked(head.seq) // in-order, head-only (P7)
				e.resolveLocked(head, nil)
			} else {
				e.resolveLocked(head, ErrLeaseExpired)
			}
			e.popInflightHeadLocked()
		default:
			// Head still owes acks (CommitFullISR) — commit is head-only, so stop.
			return
		}
	}
}

// resolveLocked is the exactly-once resolution latch for a record — the single
// point that any of {full-ack sweep, ctx timeout, transport-failure callback,
// epoch/lease-change flush} resolves through. The first caller wins; later ones
// are no-ops. It sends the outcome on the buffered doneCh (never blocks) and
// wakes admission waiters. Caller holds e.mu. Note resolveLocked(rec, nil) — a
// SUCCESS resolution — is only ever issued by sweepLocked, immediately after it
// has advanced committed under the lease fence; every other caller resolves with
// a non-nil error.
func (e *Engine) resolveLocked(rec *inflight, err error) {
	if rec.resolved {
		return
	}
	rec.resolved = true
	rec.err = err
	rec.doneCh <- err // buffered(1): exactly one send, never blocks
	// Wake admission waiters to re-check the window. The window is committed-
	// driven (lastSeq - committed), so a FAILED record does not itself free a
	// slot — this only lets a waiter re-evaluate; a wedged shard stays blocked
	// (the deliberate OH2 hard-bound). A successful resolution has already
	// advanced committed via markCommittedLocked, which frees the real slot.
	if e.windowCond != nil {
		e.windowCond.Broadcast()
	}
}

// flushEpochLocked resolves-FAILED every in-flight record whose epoch is older
// than minEpoch, then sweeps. It closes the mid-flight epoch/lease-change race:
// records of a superseded epoch can never commit under the fence, so failing
// them promptly gives clients the honest unknown-outcome answer instead of
// leaving them stuck behind a stalled head (PIPELINE-REDESIGN Q5). Caller holds
// e.mu.
func (e *Engine) flushEpochLocked(minEpoch uint64) {
	for i := 0; i < e.inflightN; i++ {
		rec := e.inflightRing[(e.inflightHead+i)%len(e.inflightRing)]
		if rec.epoch < minEpoch {
			e.resolveLocked(rec, ErrLeaseExpired)
		}
	}
	e.sweepLocked()
}

// windowFull reports whether the admission window is full: at most
// pipelineWindow writes (lastSeq - committed) may be in flight per shard. It
// acquires e.mu.
func (e *Engine) windowFull() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.windowFullLocked()
}

// windowFullLocked is the window predicate. Caller holds e.mu.
func (e *Engine) windowFullLocked() bool {
	return e.lastSeq-e.committed >= pipelineWindow
}

// windowWait blocks until the admission window has room or ctx is done. It is
// the ctx-bounded admission gate taken before writeMu in the pipelined
// Propose. Returns ctx.Err() if the wait is cancelled, else nil. It acquires
// e.mu.
func (e *Engine) windowWait(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.windowFullLocked() {
		return nil
	}
	// sync.Cond has no ctx; a watcher broadcasts on cancellation so Wait wakes
	// and the loop observes ctx.Err(). stop tears the watcher down on return.
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			e.mu.Lock()
			e.windowCond.Broadcast()
			e.mu.Unlock()
		case <-stop:
		}
	}()
	for e.windowFullLocked() {
		if err := ctx.Err(); err != nil {
			return err
		}
		e.windowCond.Wait()
	}
	return nil
}

// ============================================================================
// ISR SHRINK (live in-flight re-evaluation). GROW is a separate,
// deferred increment; nothing below ever WIDENS a required set.
// ============================================================================
//
// THE CRUX (live-ISR re-eval rule, proven): an in-flight write W(epoch=E, seq=s)
// may drop a required member M and commit against the smaller set S' IFF the
// engine has observed a MetaRaft-COMMITTED shrink to S' (M ∉ S') at the CURRENT
// epoch E, held under a still-valid lease for E, BEFORE the sweep's commit
// decision. ShrinkISR is that observation point. The drop is:
//
//   - MONOTONE-NARROWING: a required set only ever loses members, never gains
//     one. Grow (widening in-flight records) is unsound and explicitly NOT done
//     here — the asymmetry is load-bearing for the proofs below.
//   - EPOCH-SCOPED: the override (effISR/effISREpoch) is cleared on every epoch
//     advance, so a shrink decided for E can never narrow an E+1 write.
//
// PROOF — NO ACKED-LOSS. Suppose W commits after ShrinkISR narrowed it to S'.
// P6 (unchanged) still requires EVERY member of the record's propose-time
// EFFECTIVE required set to have acked (epoch,s); after narrowing that set is
// S'\{primary}, all of which acked, and the primary applied W locally. So every
// member of S' holds W at the moment of commit. ShrinkISR is only ever called
// AFTER OpSetShardISR(E,S') is MetaRaft-committed (the driver observes it in the
// local MetaFSM first), so the authoritative ISR at E is S'. A 4b failover picks
// the new primary FROM that committed ISR (⊆ S'), and every member of S' holds
// every W that committed under it. Hence no client-acked write can be lost. ∎
//
// PROOF — NO FALSE COMMIT. The drop is gated on an OBSERVED committed shrink; it
// never invents durability. After narrowing, the remaining required members
// S'\{primary} must STILL each ack (removePending only drops the removed M, not a
// live member), and the driver refuses any shrink that would take |S'| below
// minISR, so a commit still rests on ≥ minISR nodes. The commit-time lease fence
// in sweepLocked is UNTOUCHED (OH1/Q5 preserved): a fenced primary still cannot
// commit even after a narrowing. ∎
//
// THE STALE-READ RACE (why two halves are required). proposeSequenced snapshots
// ctrl.ISR OUTSIDE any lock. A Propose that captured a PRE-shrink ISR could
// register the removed M into a NEW record after ShrinkISR's narrowing pass ran
// — re-wedging the pipeline. Closing it needs BOTH:
//   (1) ShrinkISR narrows every ALREADY-registered in-flight record (below), and
//   (2) proposeSequenced intersects its stale snapshot with the live override
//       (engine.go, the effISREpoch==epoch branch).
// Both the override install here and the intersect there run under e.mu, so for
// any record either the intersect sees the override (drops M at registration) or
// this narrowing pass sees the record (drops M after registration) — no window.

// ShrinkISR installs a MetaRaft-committed ISR shrink to newISR for epoch, live
// re-evaluating every in-flight write of that epoch so a stalled pipeline can
// commit against the smaller set without losing any acked write. It is the
// engine-side landing point for a committed OpSetShardISR(epoch, newISR); the
// control-plane driver calls it only AFTER observing that op committed in the
// local MetaFSM, and only when |newISR| >= minISR (the driver enforces the floor
// as the first line; the meta FSM's OpSetShardISR apply also rejects a below-floor
// set as a structural backstop — so a below-floor set can never commit VIA
// OpSetShardISR. The failover reset to {primary} via OpSetShardEpoch is a
// deliberate, non-floor-checked exception — grow then re-widens above the floor).
//
// It is a NO-OP unless the shrink is for the engine's CURRENT epoch AND this node
// still holds that epoch's lease (epoch == e.epoch && e.leaseEpoch == epoch): a
// stale-epoch shrink, or one arriving after this node was fenced, must not touch
// live state. Ordering matches proposeSequenced (writeMu then e.mu) so the drop
// of a removed peer's sender is serialized against frame submission.
func (e *Engine) ShrinkISR(epoch uint64, newISR []string) {
	e.writeMu.Lock()
	e.mu.Lock()
	// Epoch/lease fence: only re-evaluate under a live lease for exactly this
	// epoch. A stale-epoch shrink is void (the epoch already advanced and
	// flushEpochLocked failed those records); a shrink for an epoch we hold no
	// lease for must not narrow anything (we are not the acting primary).
	if epoch != e.epoch || e.leaseEpoch != epoch {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return
	}

	// Install the durable per-epoch override. From here on, proposeSequenced's
	// intersect narrows any Propose that snapshotted a pre-shrink ISR (stale-read
	// race half 2), and this override survives across mu releases until the next
	// epoch advance clears it.
	e.effISR = append([]string(nil), newISR...)
	e.effISREpoch = epoch

	keep := make(map[string]struct{}, len(newISR))
	for _, m := range newISR {
		keep[m] = struct{}{}
	}

	// Narrow every ALREADY-registered in-flight record of THIS epoch: drop each
	// pending peer that is no longer in the ISR (stale-read race half 1). Only
	// records of `epoch` are touched — a different-epoch record (there is none
	// live here, but defensively) keeps its own required set. removePending is
	// idempotent, so re-narrowing or a later duplicate ack from M is harmless.
	for i := 0; i < e.inflightN; i++ {
		rec := e.inflightRing[(e.inflightHead+i)%len(e.inflightRing)]
		if rec.epoch != epoch {
			continue
		}
		// pendingPeers aliases the record's set, which removePending mutates in
		// place, so snapshot the removals first.
		var toDrop []string
		for _, p := range rec.pendingPeers() {
			if _, ok := keep[p]; !ok {
				toDrop = append(toDrop, p)
			}
		}
		for _, p := range toDrop {
			rec.removePending(p)
		}
	}

	// Drop the sender for every removed peer: discard its parked frames without
	// submitting (the WithDataRelease aliasing fix) and stop feeding it. Collect
	// first so we don't mutate e.peerQ while ranging it.
	var removed []string
	for peer := range e.peerQ {
		if _, ok := keep[peer]; !ok {
			removed = append(removed, peer)
		}
	}
	for _, peer := range removed {
		e.dropPeerLocked(peer)
	}

	// A narrowed head may now owe zero acks — commit it (and any newly-committable
	// successors) under the UNCHANGED lease fence. If it still owes a live member,
	// the sweep is a no-op and the shard stays correctly stalled on that member.
	e.sweepLocked()

	e.mu.Unlock()
	e.writeMu.Unlock()
}

// dropPeerLocked removes a peer that has left the ISR: it detaches the peer's
// ordered sender from peerQ and latches it into DISCARD mode, so the sender
// drains any frames still parked in its channel WITHOUT submitting them to the
// transport, then exits when the channel is closed. This is the fix for the
// WithDataRelease aliasing hazard (a post-shrink ring wrap can recycle a payload
// buffer still aliased by a parked frame) — see WithDataRelease's safety contract
// and peerSender.discard.
//
// Deliberately does NOT wait on senderWG: the sender drains and exits
// asynchronously. Waiting here would block under both writeMu and e.mu while the
// sender's own completion callbacks contend for e.mu — a self-deadlock. A late
// ack the sender already submitted before the drop is harmless: completeSend
// finds M already removed from the record's pending set (removePending is
// idempotent) or the record already resolved (resolveLocked is exactly-once), so
// there is no double-commit and no panic. Caller holds writeMu AND e.mu.
func (e *Engine) dropPeerLocked(peer string) {
	ps := e.peerQ[peer]
	if ps == nil {
		return // never had (or already dropped) a sender for this peer
	}
	delete(e.peerQ, peer)
	ps.discard.Store(true)
	close(ps.ch) // the sender drains-and-discards the remainder, then returns
}

// clearEffISRLocked voids the shrink override. Called on EVERY epoch advance so
// an override decided for one epoch can never narrow another epoch's writes.
// Caller holds e.mu.
func (e *Engine) clearEffISRLocked() {
	e.effISR = nil
	e.effISREpoch = 0
}

// intersectPeers returns the members of peers that are also in keep, reusing
// peers' backing array (safe: peers is a fresh slice built by distinctPeers).
// Order is preserved. It only ever REMOVES elements — narrowing, never widening.
func intersectPeers(peers, keep []string) []string {
	if len(keep) == 0 {
		return peers[:0]
	}
	keepSet := make(map[string]struct{}, len(keep))
	for _, k := range keep {
		keepSet[k] = struct{}{}
	}
	out := peers[:0]
	for _, p := range peers {
		if _, ok := keepSet[p]; ok {
			out = append(out, p)
		}
	}
	return out
}

// notePeerFailureLocked bumps a peer's consecutive-failure count (the shrink
// wedge signal). Caller holds e.mu.
func (e *Engine) notePeerFailureLocked(peer string) {
	if e.peerFailures == nil {
		e.peerFailures = make(map[string]int, 2)
	}
	e.peerFailures[peer]++
}

// notePeerSuccessLocked resets a peer's consecutive-failure count on a good ack.
// Caller holds e.mu.
func (e *Engine) notePeerSuccessLocked(peer string) {
	if e.peerFailures != nil && e.peerFailures[peer] != 0 {
		e.peerFailures[peer] = 0
	}
}

// StalledPeers returns the peers whose consecutive replication-failure count has
// reached minFailures — the wedge signal the primary-side shrink driver reads to
// decide a backup is dead enough to request its removal from the ISR. A pure
// backup / healthy shard returns nil. It snapshots under e.mu. minFailures should
// be set WELL above one RTT's worth of transient blips (a config knob on the
// driver) so a momentary hiccup never shrinks a healthy ISR.
func (e *Engine) StalledPeers(minFailures int) []string {
	if minFailures < 1 {
		minFailures = 1
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	var out []string
	for peer, n := range e.peerFailures {
		if n >= minFailures {
			out = append(out, peer)
		}
	}
	return out
}
