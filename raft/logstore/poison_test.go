// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"errors"
	"os"
	"testing"

	hraft "github.com/hashicorp/raft"
)

// failNextFsync arms the WAL's fsync seam to fail exactly once with errBoom,
// then restores the real (*os.File).Sync — modeling post-4.13 Linux, where the
// fsync AFTER a failure typically returns success (the dirty pages it should
// have covered were already dropped). Everything the latch protects against
// happens in that second, falsely-successful sync.
var errBoom = errors.New("injected fsync failure")

func failNextFsync(w *WAL) {
	fired := false
	w.fsyncf = func(f *os.File) error {
		if fired {
			return (*os.File).Sync(f)
		}
		fired = true
		return errBoom
	}
}

// TestStoreLogsFsyncFailurePoisons pins the fail-closed contract on the log
// path: the failing batch surfaces the fsync error, and every later write entry
// point (StoreLogs, Set/SetUint64, DeleteRange) returns ErrWALPoisoned even
// though the injected seam would now let an fsync "succeed". Reads of
// already-durable entries stay open.
func TestStoreLogsFsyncFailurePoisons(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()

	if err := w.StoreLog(mkLog(1, "a")); err != nil {
		t.Fatalf("seed: %v", err)
	}
	failNextFsync(w)
	if err := w.StoreLog(mkLog(2, "b")); !errors.Is(err, errBoom) {
		t.Fatalf("expected injected fsync error, got %v", err)
	}
	// The seam now succeeds — exactly the false-success retry the latch exists
	// to refuse.
	if err := w.StoreLog(mkLog(3, "c")); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("StoreLogs after fsync failure: want ErrWALPoisoned, got %v", err)
	}
	if err := w.SetUint64([]byte("term"), 7); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("Set after fsync failure: want ErrWALPoisoned, got %v", err)
	}
	if err := w.DeleteRange(1, 1); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("DeleteRange after fsync failure: want ErrWALPoisoned, got %v", err)
	}
	// Reads stay open: entry 1 was durably stored before the failure.
	var got hraft.Log
	if err := w.GetLog(1, &got); err != nil {
		t.Fatalf("GetLog on poisoned WAL: %v", err)
	}
}

// TestStableFsyncFailurePoisons pins the same latch on the stable-store path
// (currentTerm/votedFor): a falsely-durable vote is the split-brain hazard, so
// the first fsync failure there must poison the whole store too.
func TestStableFsyncFailurePoisons(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()

	failNextFsync(w)
	if err := w.Set([]byte("votedFor"), []byte("n1")); !errors.Is(err, errBoom) {
		t.Fatalf("expected injected fsync error, got %v", err)
	}
	if err := w.StoreLog(mkLog(1, "a")); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("StoreLogs after stable fsync failure: want ErrWALPoisoned, got %v", err)
	}
}

// TestPoisonClearsOnReopen pins the ONLY sanctioned recovery: a fresh OpenWAL.
// The reopened store recovers every entry the failed sync did not cover being
// torn away (here entry 1, durable before the injected failure) and accepts
// writes again.
func TestPoisonClearsOnReopen(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.StoreLog(mkLog(1, "a")); err != nil {
		t.Fatal(err)
	}
	failNextFsync(w)
	if err := w.StoreLog(mkLog(2, "b")); !errors.Is(err, errBoom) {
		t.Fatalf("expected injected fsync error, got %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close poisoned: %v", err)
	}

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w2.Close() }()
	last, err := w2.LastIndex()
	if err != nil || last < 1 {
		t.Fatalf("reopen lost the durable prefix: last=%d err=%v", last, err)
	}
	if err := w2.StoreLog(mkLog(last+1, "c")); err != nil {
		t.Fatalf("reopened WAL must accept writes: %v", err)
	}
}

// TestNoSyncModeNeverPoisonsOnStoreLogs pins the sync=false posture: StoreLogs
// runs no fsync there, so the latch has nothing to trip on the log path and the
// documented durability-from-replication contract is unchanged.
func TestNoSyncModeNeverPoisonsOnStoreLogs(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	w.fsyncf = func(*os.File) error { return errBoom } // would fail if ever called on this path
	if err := w.StoreLog(mkLog(1, "a")); err != nil {
		t.Fatalf("nosync StoreLogs must not fsync: %v", err)
	}
}
