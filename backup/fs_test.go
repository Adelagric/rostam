// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFSObjectStorePutRoundTrip verifies that a value written via Put is
// durably persisted and reads back byte-for-byte, and that the temp file used
// for the atomic write is renamed away (never left behind) on success. The
// durability discipline the fix adds — tmp.Sync() before rename plus a
// parent-directory fsync — must not break this happy path.
func TestFSObjectStorePutRoundTrip(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}

	const key = "tenant/coll/2026-07-08T00-00-00Z.snap"
	want := []byte("durable snapshot payload")
	if err := store.Put(context.Background(), key, strings.NewReader(string(want)), int64(len(want))); err != nil {
		t.Fatalf("Put: %v", err)
	}

	rc, err := store.Get(context.Background(), key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer rc.Close()
	got, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("round-trip mismatch: got %q, want %q", got, want)
	}

	// After a successful Put the atomic temp file must be gone: it was either
	// renamed onto dst or removed on an error path. A lingering ".tmp-*" file
	// would signal a broken rename step.
	if tmps := listTempFiles(t, filepath.Join(root, "tenant", "coll")); len(tmps) != 0 {
		t.Fatalf("temp files left behind after Put: %v", tmps)
	}
}

// TestFSObjectStorePutCopyErrorNoDest verifies the error path still removes the
// temp file and leaves no destination behind when the reader fails mid-copy —
// the fix's added Sync steps must not regress this cleanup.
func TestFSObjectStorePutCopyErrorNoDest(t *testing.T) {
	root := t.TempDir()
	store, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatalf("NewFSObjectStore: %v", err)
	}

	const key = "tenant/coll/bad.snap"
	if err := store.Put(context.Background(), key, errReader{}, 0); err == nil {
		t.Fatal("Put: expected error from failing reader, got nil")
	}

	if _, err := os.Stat(filepath.Join(root, "tenant", "coll", "bad.snap")); !os.IsNotExist(err) {
		t.Fatalf("destination should not exist after failed Put, stat err = %v", err)
	}
	if tmps := listTempFiles(t, filepath.Join(root, "tenant", "coll")); len(tmps) != 0 {
		t.Fatalf("temp files left behind after failed Put: %v", tmps)
	}
}

// errReader always fails, exercising Put's io.Copy error branch.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }

// listTempFiles returns the names of any ".tmp-*" atomic-write temp files under
// dir. A missing dir yields none.
func listTempFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		t.Fatalf("ReadDir %q: %v", dir, err)
	}
	var tmps []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".tmp-") {
			tmps = append(tmps, e.Name())
		}
	}
	return tmps
}

// TestFSObjectStoreListPrefixScopedWalk pins that List, which now starts its
// walk at the directory the prefix implies instead of at root, returns exactly
// what a full-root walk filtered by key prefix would — including a trailing
// partial segment ("acme/col/2024-"), a single exact key, a prefix whose
// directory does not exist (nil, no error), and a ".." prefix (nothing, never
// a path outside root).
func TestFSObjectStoreListPrefixScopedWalk(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	fsStore, err := NewFSObjectStore(root)
	if err != nil {
		t.Fatal(err)
	}
	keys := []string{
		"acme/col/2024-01-01T00:00:00Z.snap",
		"acme/col/2024-01-01T00:00:00Z.cfg.json",
		"acme/col/2025-01-01T00:00:00Z.snap",
		"acme/other/2024-01-01T00:00:00Z.snap",
		"beta/col/2024-01-01T00:00:00Z.snap",
		"top.snap",
	}
	for _, k := range keys {
		if err := fsStore.Put(ctx, k, strings.NewReader(k), int64(len(k))); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}
	// A file outside root that a ".." prefix must never reach.
	outside := filepath.Join(filepath.Dir(root), "outside.snap")
	if err := os.WriteFile(outside, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Remove(outside) })

	listKeys := func(prefix string) []string {
		t.Helper()
		infos, err := fsStore.List(ctx, prefix)
		if err != nil {
			t.Fatalf("list %q: %v", prefix, err)
		}
		out := make([]string, 0, len(infos))
		for _, in := range infos {
			out = append(out, in.Key)
		}
		return out
	}
	// Reference: full-root walk filtered by key prefix.
	fullWalk := func(prefix string) []string {
		var out []string
		for _, k := range keys {
			if strings.HasPrefix(k, prefix) {
				out = append(out, k)
			}
		}
		sortStrings(out)
		return out
	}
	for _, prefix := range []string{
		"", "acme/", "acme/col/", "acme/col/2024-", "acme/col/2024-01-01T00:00:00Z.snap",
		"acme/co", "top", "nope/", "acme/nope/2024-", "../", "../outside",
	} {
		got, want := listKeys(prefix), fullWalk(prefix)
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("prefix %q: got %v, want %v", prefix, got, want)
		}
	}
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
