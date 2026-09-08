// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
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
	failFsyncIf(w, func(*os.File) bool { return true })
}

// failFsyncIf arms the seam to fail exactly once with errBoom, on the first sync
// whose target matches want; every other sync (and everything after the failure)
// runs the real (*os.File).Sync. Selecting the target by predicate rather than by
// call count keeps the latch-site tests independent of how many incidental syncs
// a path performs. Per the fsyncf contract, arm it only while the WAL is idle.
func failFsyncIf(w *WAL, want func(*os.File) bool) {
	fired := false
	w.fsyncf = func(f *os.File) error {
		if fired || !want(f) {
			return (*os.File).Sync(f)
		}
		fired = true
		return errBoom
	}
}

// isDirHandle matches the directory handle syncDir opens, so a test can fail the
// directory fsync while letting the segment fsyncs ahead of it succeed.
func isDirHandle(f *os.File) bool {
	fi, err := f.Stat()
	return err == nil && fi.IsDir()
}

// assertFailClosed pins the post-latch contract: every write entry point and the
// gated stable-store read return ErrWALPoisoned.
func assertFailClosed(t *testing.T, w *WAL, next uint64) {
	t.Helper()
	if !w.poisoned {
		t.Fatal("latch not tripped")
	}
	if err := w.StoreLog(mkLog(next, "after")); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("StoreLogs: want ErrWALPoisoned, got %v", err)
	}
	if err := w.SetUint64([]byte("term"), 9); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("Set: want ErrWALPoisoned, got %v", err)
	}
	if _, err := w.Get([]byte("votedFor")); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("Get: want ErrWALPoisoned, got %v", err)
	}
	if err := w.DeleteRange(1, 1); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("DeleteRange: want ErrWALPoisoned, got %v", err)
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
	// Reads of the stable store are gated too: Set mutated the in-memory map
	// before the failed persist, so serving it would hand raft a term/vote no
	// reopen will reproduce (unlike GetLog, which fails loud via CRC on its own).
	if _, err := w.Get([]byte("votedFor")); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("Get after stable fsync failure: want ErrWALPoisoned, got %v", err)
	}
	if _, err := w.GetUint64([]byte("term")); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("GetUint64 after stable fsync failure: want ErrWALPoisoned, got %v", err)
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

// TestNoSyncModeNeverPoisonsOnStoreLogs pins the sync=false LOG posture:
// StoreLogs runs no fsync there, so the latch has nothing to trip on the log
// path and the documented durability-from-replication contract is unchanged.
// Only the log path is latch-free in this mode — see the stable-path test below.
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

// TestNoSyncModeStillPoisonsOnStablePath pins that sync=false does NOT exempt
// the stable store (currentTerm/votedFor) or the compaction floor: those are
// durable-unconditionally by design (see writeStableLocked — a granted vote must
// survive a crash in EVERY mode, or the node can double-vote), so the fsyncgate
// hazard and its latch apply there in every mode too.
func TestNoSyncModeStillPoisonsOnStablePath(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	failNextFsync(w)
	if err := w.Set([]byte("votedFor"), []byte("n1")); !errors.Is(err, errBoom) {
		t.Fatalf("expected injected fsync error, got %v", err)
	}
	if err := w.StoreLog(mkLog(1, "a")); !errors.Is(err, ErrWALPoisoned) {
		t.Fatalf("nosync StoreLogs after stable fsync failure: want ErrWALPoisoned, got %v", err)
	}
}

// TestStableNonFsyncFailurePoisons pins the widened contract: a durable write
// that fails ANYWHERE — not only at the fsync — is terminal. Set mutates w.kv
// before persisting, so a failed create (ENOSPC/ENOTDIR on the staging file) or
// a failed rename leaves the in-memory map holding a term/vote that is not on
// disk and that no reopen will reproduce. Without the latch the read gate would
// happily serve it. Both cases are driven through the real filesystem rather
// than the fsync seam, which by construction cannot reach these paths.
func TestStableNonFsyncFailurePoisons(t *testing.T) {
	cases := []struct {
		name string
		// stable returns the stable-store path to point the WAL at, having
		// created whatever makes the durable write fail.
		stable func(t *testing.T, dir string) string
		// wantStage is the stage of writeFileDurable expected to fail.
		wantStage string
	}{
		{
			name: "staging create fails",
			stable: func(t *testing.T, dir string) string {
				blocker := filepath.Join(dir, "blocker")
				if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
					t.Fatal(err)
				}
				return filepath.Join(blocker, "stable") // parent is a FILE => ENOTDIR
			},
			wantStage: "create",
		},
		{
			name: "rename fails",
			stable: func(t *testing.T, dir string) string {
				target := filepath.Join(dir, "stable-as-dir")
				if err := os.Mkdir(target, 0o750); err != nil {
					t.Fatal(err)
				}
				return target // renaming a file onto a directory => EISDIR
			},
			wantStage: "rename",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			w, err := OpenWAL(dir, true)
			if err != nil {
				t.Fatal(err)
			}
			w.stablePath = tc.stable(t, dir)

			err = w.Set([]byte("votedFor"), []byte("n1"))
			if err == nil {
				t.Fatal("Set must surface the durable-write failure")
			}
			if errors.Is(err, ErrWALPoisoned) {
				t.Fatalf("want the underlying cause, got the latch error: %v", err)
			}
			if !strings.Contains(err.Error(), tc.wantStage) {
				t.Fatalf("want a %s failure, got %v", tc.wantStage, err)
			}
			assertFailClosed(t, w, 1)
			if err := w.Close(); err != nil {
				t.Fatalf("close poisoned: %v", err)
			}

			// The latch clears only on a fresh open, which is then fully usable.
			w2, err := OpenWAL(dir, true)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = w2.Close() }()
			if err := w2.Set([]byte("votedFor"), []byte("n1")); err != nil {
				t.Fatalf("reopened WAL must accept Set: %v", err)
			}
			if err := w2.StoreLog(mkLog(1, "a")); err != nil {
				t.Fatalf("reopened WAL must accept writes: %v", err)
			}
			if v, err := w2.Get([]byte("votedFor")); err != nil || string(v) != "n1" {
				t.Fatalf("reopened Get: got %q err=%v", v, err)
			}
		})
	}
}

// TestRotationDirSyncFailurePoisons covers the syncDir latch site: a batch that
// rotates must make the NEW segment's directory entry durable, so a failing
// directory fsync is as terminal as a failing segment fsync (recovery could
// otherwise not find a segment whose records were acked).
func TestRotationDirSyncFailurePoisons(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	w.maxSeg = 1 // every append rotates

	if err := w.StoreLog(mkLog(1, "a")); err != nil { // creates the first segment
		t.Fatalf("seed: %v", err)
	}
	failFsyncIf(w, isDirHandle) // let the segment fsyncs through, fail the dir sync
	err = w.StoreLog(mkLog(2, "b"))
	if !errors.Is(err, errBoom) || !strings.Contains(err.Error(), "sync dir") {
		t.Fatalf("want the injected dir-sync failure, got %v", err)
	}
	assertFailClosed(t, w, 3)
}

// TestTruncateTailFsyncFailurePoisons covers the truncateTailLocked latch site:
// the conflict-truncation fsync is what stops a crash from resurrecting the
// removed tail, so its failure must fail closed like any other.
func TestTruncateTailFsyncFailurePoisons(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	for i := uint64(1); i <= 3; i++ {
		if err := w.StoreLog(mkLog(i, "a")); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}
	failNextFsync(w) // the truncation fsync is the next sync on this path
	err = w.DeleteRange(3, 3)
	if !errors.Is(err, errBoom) || !strings.Contains(err.Error(), "truncated segment") {
		t.Fatalf("want the injected truncate fsync failure, got %v", err)
	}
	assertFailClosed(t, w, 4)
}

// TestWriteFloorFsyncFailurePoisons covers the writeFloorLocked latch site: the
// compaction floor is what makes recovery skip the dropped prefix, and
// truncateFrontLocked advances w.first in memory before persisting it — so a
// failed floor write leaves memory ahead of disk and must poison.
func TestWriteFloorFsyncFailurePoisons(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	for i := uint64(1); i <= 3; i++ {
		if err := w.StoreLog(mkLog(i, "a")); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}
	failNextFsync(w) // the floor file's fsync is the next sync on this path
	err = w.DeleteRange(1, 1)
	if !errors.Is(err, errBoom) || !strings.Contains(err.Error(), "first.tmp") {
		t.Fatalf("want the injected floor fsync failure, got %v", err)
	}
	assertFailClosed(t, w, 4)
}

// TestDirOpenFailurePoisons covers syncDir's OPEN failure, which four callers
// reach without passing through writeFileDurable's defer. It is driven through
// truncateTailLocked because that caller is the sharp one in production: it has
// closed and removed the segments holding a conflicting tail, so an unlatched
// store would ack the next batch while a crash could still resurrect those
// segments and collide with the re-appended entries.
//
// The open failure is injected by dropping the directory's permissions, which
// root ignores — hence the skip. NOTE the same permission bits also block the
// unlink (os.Remove needs w+x on the directory, and truncateTailLocked discards
// its error), so under THIS injection the segment file actually stays on disk:
// the test pins the poison-on-open-failure latch for this call path, not the
// removed-then-unresolvable state itself, which no permission trick can create.
func TestDirOpenFailurePoisons(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory permission checks")
	}
	dir := t.TempDir()
	// Registered AFTER t.TempDir so it runs BEFORE TempDir's removal (cleanups are
	// LIFO); otherwise the unreadable directory would fail the test's own cleanup.
	t.Cleanup(func() { _ = os.Chmod(dir, 0o750) })

	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()
	w.maxSeg = 1 // one segment per entry, so a tail truncation removes segments
	for i := uint64(1); i <= 3; i++ {
		if err := w.StoreLog(mkLog(i, "a")); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}
	if len(w.segs) != 3 {
		t.Fatalf("want one segment per entry, got %d", len(w.segs))
	}
	if err := os.Chmod(dir, 0o000); err != nil {
		t.Fatal(err)
	}
	// Drops entries 2 and 3: the tail truncation runs (the in-memory index and
	// the fd truncate need no directory permission), then the directory sync that
	// should make the segment removal durable cannot even open the directory.
	// (The unlink itself is also blocked by the 0o000 bits and its error is
	// discarded — see the doc comment above.)
	err = w.DeleteRange(2, 3)
	if err == nil || !strings.Contains(err.Error(), "open dir") {
		t.Fatalf("want the directory open failure, got %v", err)
	}
	assertFailClosed(t, w, 2)
}
