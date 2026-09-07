// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestAtomicWriteFile pins the durable-publish contract of the helper the JSON
// markers (config, named/MV config, key registry) now share: the payload lands
// byte-exact at the target, the .tmp staging file never survives (success or
// not), and a second publish over an existing file replaces it atomically.
func TestAtomicWriteFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "marker.json")

	want := []byte(`{"a":1}`)
	if err := atomicWriteFile(path, want); err != nil {
		t.Fatalf("atomicWriteFile: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("content mismatch: got %q want %q", got, want)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("staging file left behind: %v", err)
	}

	// Overwrite publish: the old content must be fully replaced.
	want2 := []byte(`{"a":2,"b":"longer than before"}`)
	if err := atomicWriteFile(path, want2); err != nil {
		t.Fatalf("atomicWriteFile overwrite: %v", err)
	}
	got2, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back 2: %v", err)
	}
	if !bytes.Equal(got2, want2) {
		t.Fatalf("overwrite mismatch: got %q want %q", got2, want2)
	}
}

// TestAtomicWriteFileMissingDir pins fail-loud on an unpublishable target: a
// missing parent directory must surface an error (and never a false success a
// caller like Flush would then truncate a WAL on).
func TestAtomicWriteFileMissingDir(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nope", "marker.json")
	if err := atomicWriteFile(path, []byte("x")); err == nil {
		t.Fatal("expected error for missing parent dir, got nil")
	}
}

// TestRenameDurable pins the happy path (rename lands, source gone) and the
// fail-loud path (renaming onto a target in a nonexistent directory errors).
func TestRenameDurable(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src")
	dst := filepath.Join(dir, "dst")
	if err := os.WriteFile(src, []byte("payload"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := renameDurable(src, dst); err != nil {
		t.Fatalf("renameDurable: %v", err)
	}
	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Fatalf("source still present after rename: %v", err)
	}
	got, err := os.ReadFile(dst)
	if err != nil || string(got) != "payload" {
		t.Fatalf("dest read: %q, %v", got, err)
	}

	if err := os.WriteFile(src, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := renameDurable(src, filepath.Join(dir, "nope", "dst")); err == nil {
		t.Fatal("expected error renaming into missing dir, got nil")
	}
	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Fatalf("staging file left behind after failed rename: %v", err)
	}
}

// TestAtomicWriteFileRenameFailureCleansTmp pins the failed-rename cleanup: a
// target that is an existing DIRECTORY makes the final rename fail after the
// staging file was written and fsync'd, and the .tmp must not survive it.
func TestAtomicWriteFileRenameFailureCleansTmp(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "occupied")
	if err := os.Mkdir(target, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := atomicWriteFile(target, []byte("x")); err == nil {
		t.Fatal("expected error publishing onto an existing directory, got nil")
	}
	if _, err := os.Stat(target + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("staging file left behind after failed rename: %v", err)
	}
}

// TestSyncDir smoke-checks the platform seam: on platforms with directory
// fsync it must succeed on a real directory and error on a missing one; where
// dirSyncSupported is false it is a documented no-op either way.
func TestSyncDir(t *testing.T) {
	if err := syncDir(t.TempDir()); err != nil {
		t.Fatalf("syncDir on real dir: %v", err)
	}
	if dirSyncSupported {
		if err := syncDir(filepath.Join(t.TempDir(), "nope")); err == nil {
			t.Fatal("expected error for missing dir, got nil")
		}
	}
}
