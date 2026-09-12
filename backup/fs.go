// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/rostamlabs/rostam/objstore"
)

// FSObjectStore is a filesystem-backed objstore.ObjectStore rooted at a local
// directory. Each object key maps to a file at <root>/<key>, with key's "/"
// separators becoming directory levels (parent dirs are created on Put). It is
// fully demoable today with no cloud credentials and lets the backup driver run
// against local disk before any S3 bucket is configured.
//
// It implements the SAME objstore.ObjectStore interface as the SigV4 S3 client,
// so swapping FSObjectStore for an S3Store in the server's backup driver is a
// one-line change and the Backup/Restore/retention logic is untouched.
//
// Writes are atomic per object: Put streams into a temp file in the destination
// directory and renames it over the final path, so a crash mid-write never
// leaves a torn snapshot (mirroring CollectionStore's tmp+rename discipline).
type FSObjectStore struct {
	root string
}

// compile-time assertion that FSObjectStore satisfies objstore.ObjectStore.
var _ objstore.ObjectStore = (*FSObjectStore)(nil)

// NewFSObjectStore returns an FSObjectStore rooted at root, creating root if it
// does not exist.
func NewFSObjectStore(root string) (*FSObjectStore, error) {
	if err := os.MkdirAll(root, 0o750); err != nil {
		return nil, fmt.Errorf("fsstore: mkdir root %q: %w", root, err)
	}
	return &FSObjectStore{root: root}, nil
}

// keyToPath resolves a forward-slash object key to an on-disk path under root,
// using the OS path separator. It REJECTS any key that would escape root: the
// key is path.Clean'd as an absolute path (collapsing ".."/"."), then the joined
// destination is verified to stay within root. A key containing "../" segments,
// a NUL byte, or one that resolves to root itself is an error — so Put cannot
// overwrite, Get cannot read, and Delete cannot remove a file outside root.
func (f *FSObjectStore) keyToPath(key string) (string, error) {
	if strings.ContainsRune(key, '\x00') {
		return "", fmt.Errorf("fsstore: invalid key %q", key)
	}
	// path.Clean("/"+key) forces an absolute path and resolves away ".."/"." so a
	// traversal key like "../../etc/passwd" cleans to "/etc/passwd" (a single
	// rooted segment list) rather than escaping.
	clean := path.Clean("/" + key)
	if clean == "/" {
		return "", fmt.Errorf("fsstore: invalid key %q (empty after clean)", key)
	}
	dst := filepath.Join(f.root, filepath.FromSlash(clean))
	rootAbs, err := filepath.Abs(f.root)
	if err != nil {
		return "", fmt.Errorf("fsstore: resolve root: %w", err)
	}
	dstAbs, err := filepath.Abs(dst)
	if err != nil {
		return "", fmt.Errorf("fsstore: resolve key %q: %w", key, err)
	}
	if dstAbs != rootAbs && !strings.HasPrefix(dstAbs, rootAbs+string(os.PathSeparator)) {
		return "", fmt.Errorf("fsstore: key %q escapes root", key)
	}
	return dst, nil
}

// Put writes r's full contents to <root>/<key> atomically (temp file + rename),
// creating parent directories as needed and overwriting any prior value. size is
// accepted for interface parity (Content-Length on the wire); the bytes actually
// written are whatever r yields.
func (f *FSObjectStore) Put(_ context.Context, key string, r io.Reader, size int64) error {
	dst, err := f.keyToPath(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
		return fmt.Errorf("fsstore: mkdir for %q: %w", key, err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".tmp-*"+snapExt)
	if err != nil {
		return fmt.Errorf("fsstore: temp for %q: %w", key, err)
	}
	tmpName := tmp.Name()
	if _, err := io.Copy(tmp, r); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: write %q: %w", key, err)
	}
	// fsync the temp file before the rename so the data blocks reach stable
	// storage first (mirroring CollectionStore.writeSnapshotFile). Without this,
	// ext4/xfs may make the rename's metadata durable before the file data,
	// leaving a torn/empty snapshot at dst after power loss even though Put
	// already returned nil.
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: sync temp for %q: %w", key, err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: close temp for %q: %w", key, err)
	}
	if err := os.Rename(tmpName, dst); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("fsstore: rename for %q: %w", key, err)
	}
	// fsync the destination's parent directory so the rename itself is durable;
	// otherwise the directory entry for dst may be lost on power loss.
	if err := syncDir(filepath.Dir(dst)); err != nil {
		return fmt.Errorf("fsstore: sync dir for %q: %w", key, err)
	}
	return nil
}

// syncDir fsyncs the directory at dir so a preceding rename/create within it is
// made durable. Both a failure to open the directory and a Sync error are
// surfaced to the caller, since either means the rename cannot be guaranteed
// durable across a crash.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err := d.Sync(); err != nil {
		_ = d.Close()
		return err
	}
	return d.Close()
}

// Get opens <root>/<key> for reading. The caller MUST Close the returned reader.
// A missing key returns objstore.ErrNotFound.
func (f *FSObjectStore) Get(_ context.Context, key string) (io.ReadCloser, error) {
	dst, err := f.keyToPath(key)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(dst)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, objstore.ErrNotFound
		}
		return nil, fmt.Errorf("fsstore: open %q: %w", key, err)
	}
	return file, nil
}

// List walks the subtree implied by prefix and returns every object whose key
// begins with prefix (the path relative to root, in forward-slash form), sorted
// by key. Directories are not reported. A still-empty root, or a prefix whose
// directory does not exist yet, yields no objects (nil), not an error.
//
// The walk starts at the deepest directory the prefix fully names (for
// "acme/col/2024-" that is <root>/acme/col), not at root: retention lists one
// collection prefix per collection and per run, and an existence probe lists a
// single key, so walking the whole root would make each backup run cost
// O(collections × all stored files). The key-prefix filter is still applied to
// every entry, so the result is identical to a full-root walk.
//
// NOTE: like the S3 store, List BUFFERS every matching key in memory (one
// ObjectInfo per matching file). This is bounded for the backup/retention use
// (a handful of snapshots per collection prefix) but a prefix spanning millions
// of files will materialize them all at once; scope the prefix narrowly.
func (f *FSObjectStore) List(_ context.Context, prefix string) ([]objstore.ObjectInfo, error) {
	start := f.listStart(prefix)
	if _, err := os.Stat(start); err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("fsstore: list %q: %w", prefix, err)
	}
	var out []objstore.ObjectInfo
	err := filepath.WalkDir(start, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, rerr := filepath.Rel(f.root, p)
		if rerr != nil {
			return rerr
		}
		key := filepath.ToSlash(rel)
		if !strings.HasPrefix(key, prefix) {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return ierr
		}
		out = append(out, objstore.ObjectInfo{
			Key:          key,
			Size:         info.Size(),
			LastModified: info.ModTime(),
		})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("fsstore: list %q: %w", prefix, err)
	}
	sortInfosByKey(out)
	return out, nil
}

// listStart returns the directory List should walk for prefix: <root> joined
// with every COMPLETE "/"-separated segment of prefix (the trailing partial
// segment, if any, is a name filter, not a directory). A prefix that would
// escape root (".." segments) falls back to root itself, where the key-prefix
// filter then matches nothing — never a path outside root.
func (f *FSObjectStore) listStart(prefix string) string {
	i := strings.LastIndex(prefix, "/")
	if i < 0 {
		return f.root
	}
	dir := filepath.Join(f.root, filepath.FromSlash(path.Clean("/"+prefix[:i])))
	rootAbs, err := filepath.Abs(f.root)
	if err != nil {
		return f.root
	}
	dirAbs, err := filepath.Abs(dir)
	if err != nil || (dirAbs != rootAbs && !strings.HasPrefix(dirAbs, rootAbs+string(os.PathSeparator))) {
		return f.root
	}
	return dir
}

// Delete removes <root>/<key>. Deleting a missing key returns
// objstore.ErrNotFound, matching the S3 client and the MemStore fake.
func (f *FSObjectStore) Delete(_ context.Context, key string) error {
	dst, err := f.keyToPath(key)
	if err != nil {
		return err
	}
	if err := os.Remove(dst); err != nil {
		if os.IsNotExist(err) {
			return objstore.ErrNotFound
		}
		return fmt.Errorf("fsstore: delete %q: %w", key, err)
	}
	return nil
}

// sortInfosByKey sorts object infos ascending by key, matching the
// objstore.ObjectStore.List contract (sorted by key).
func sortInfosByKey(in []objstore.ObjectInfo) {
	for i := 1; i < len(in); i++ {
		for j := i; j > 0 && in[j-1].Key > in[j].Key; j-- {
			in[j-1], in[j] = in[j], in[j-1]
		}
	}
}
