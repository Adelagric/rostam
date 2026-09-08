// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"os"
	"path/filepath"
)

// syncDir fsyncs a directory so a rename within it is durable. Mirrors
// cache.syncDir (cache/compact.go): a temp+fsync+rename publish makes the NEW
// bytes durable, but the rename itself lives in the directory — without a
// directory fsync a crash can roll the entry back to the OLD file even though
// the rename returned nil.
//
// Unlike the cache's warn-and-continue posture, callers here PROPAGATE the
// error: every vector-package rename publishes a checkpoint whose durability a
// follow-up step relies on (most critically Flush, which truncates the WAL on
// the strength of "the checkpoint subsumes the log" — see CollectionStore.Flush).
// Failing the publish keeps that follow-up from running against a checkpoint
// that may not survive a crash.
func syncDir(dir string) error {
	// Nothing to force where the platform has no directory fsync (Windows): the
	// rename's metadata is the filesystem's own to commit there. See
	// dirsync_windows.go.
	if !dirSyncSupported {
		return nil
	}
	d, err := os.Open(dir) //nolint:gosec // G304: dir derives from a store-managed path
	if err != nil {
		return err
	}
	serr := d.Sync()
	if cerr := d.Close(); cerr != nil && serr == nil {
		serr = cerr
	}
	return serr
}

// syncDirf is the directory-fsync seam. Production always runs syncDir; tests
// replace it to exercise the one window a real fsync failure opens — the rename
// has LANDED but the publish reports failure — which is how a caller can be told
// its write failed while the new bytes are already visible on disk. Mirrors the
// fsyncf seam in raft/logstore. Not for concurrent use: a test swaps it in and
// restores it around a single sequential call.
var syncDirf = syncDir

// renameDurable completes an atomic publish: rename tmp over path, then fsync
// the parent directory so the rename survives a crash. The caller must already
// have fsynced tmp's CONTENTS (directly, or via atomicWriteFile) — this makes
// the directory ENTRY durable, the half temp+fsync+rename alone does not cover.
func renameDurable(tmp, path string) error {
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp) // best-effort: don't leave the fsync'd staging file behind
		return err
	}
	if err := syncDirf(filepath.Dir(path)); err != nil {
		return fmt.Errorf("vector: fsync dir %s: %w", filepath.Dir(path), err)
	}
	return nil
}

// atomicWriteFile durably publishes data at path via the full
// temp→fsync→rename→dir-fsync sequence. It replaces the bare
// os.WriteFile(tmp)+os.Rename pattern the small JSON markers used, which
// fsynced NEITHER the bytes NOR the rename: a crash could publish a truncated
// (or vanished) config/registry even though the write returned nil.
func atomicWriteFile(path string, data []byte) error {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600) //nolint:gosec // store-managed path
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return renameDurable(tmp, path)
}
