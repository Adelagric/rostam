// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// errInjectedDirSync is the failure a dirSyncTrap returns from the seam.
var errInjectedDirSync = errors.New("vector: injected directory-fsync failure")

// dirSyncTrap swaps the directory-fsync seam for one that records the directory
// and fails. That reproduces the only window where a durable publish reports
// failure with the RENAME ALREADY LANDED: the marker is on disk, but the create
// that wrote it returns an error. Recording the directory lets each test assert
// the failure really happened at the marker's publish (so the test still covers
// the intended path if an earlier step ever starts failing first).
//
// Tests using it must not call t.Parallel: the seam is a package-level var.
type dirSyncTrap struct {
	dirs    []string
	restore func()
}

func trapSyncDir(t *testing.T) *dirSyncTrap {
	t.Helper()
	orig := syncDirf
	tr := &dirSyncTrap{}
	tr.restore = func() { syncDirf = orig }
	syncDirf = func(dir string) error {
		tr.dirs = append(tr.dirs, dir)
		return errInjectedDirSync
	}
	t.Cleanup(tr.restore)
	return tr
}

// sawDirOf reports whether the seam was reached for path's parent directory,
// i.e. whether path's rename landed before the publish failed.
func (tr *dirSyncTrap) sawDirOf(path string) bool {
	want := filepath.Dir(path)
	for _, d := range tr.dirs {
		if d == want {
			return true
		}
	}
	return false
}

// assertMarkerCleaned pins the shared post-condition of a failed create: the
// publish reached the directory fsync (so the marker had landed), the marker is
// gone again, and nothing survives to be reloaded.
func assertMarkerCleaned(t *testing.T, tr *dirSyncTrap, cfgPath string) {
	t.Helper()
	if !tr.sawDirOf(cfgPath) {
		t.Fatalf("create failed before publishing %s; the test no longer covers the landed-rename window", cfgPath)
	}
	if _, err := os.Stat(cfgPath); !os.IsNotExist(err) {
		t.Fatalf("config marker %s survived a failed create: %v", cfgPath, err)
	}
}

// TestCreateCollectionRemovesMarkerWhenPublishFails covers the dense family: a
// create whose config publish fails after the rename landed must leave no marker,
// or a restart would resurrect the collection as an empty one.
func TestCreateCollectionRemovesMarkerWhenPublishFails(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	tr := trapSyncDir(t)
	cerr := cs.CreateCollection("docs", Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	tr.restore()
	if cerr == nil {
		t.Fatal("CreateCollection: expected an error when the durable publish fails, got nil")
	}

	cfgPath, _ := cs.collectionPath("default/docs")
	assertMarkerCleaned(t, tr, cfgPath)
	if _, ok := cs.Get("docs"); ok {
		t.Error("collection registered in memory despite a failed create")
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	if names := cs2.CollectionNames(); len(names) != 0 {
		t.Errorf("reopened store listed %v after a failed create; want none", names)
	}
}

// TestCreateNamedRemovesMarkerWhenPublishFails is the named-vector shape of
// TestCreateCollectionRemovesMarkerWhenPublishFails (WAL mode is the only named
// mode that writes a marker).
func TestCreateNamedRemovesMarkerWhenPublishFails(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	tr := trapSyncDir(t)
	cerr := cs.CreateNamedConfig("named", NamedConfig{Spaces: namedTestConfig(), WAL: true})
	tr.restore()
	if cerr == nil {
		t.Fatal("CreateNamedConfig: expected an error when the durable publish fails, got nil")
	}

	cfgPath, _, _ := cs.namedPaths("default/named")
	assertMarkerCleaned(t, tr, cfgPath)
	if _, ok := cs.GetNamed("named"); ok {
		t.Error("named collection registered in memory despite a failed create")
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	if _, ok := cs2.GetNamed("named"); ok {
		t.Error("named collection reloaded after a failed create")
	}
}

// TestCreateMultiVectorRemovesMarkerWhenPublishFails is the multi-vector shape of
// TestCreateCollectionRemovesMarkerWhenPublishFails.
func TestCreateMultiVectorRemovesMarkerWhenPublishFails(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	tr := trapSyncDir(t)
	cerr := cs.CreateMultiVector("mv", mvWALConfig())
	tr.restore()
	if cerr == nil {
		t.Fatal("CreateMultiVector: expected an error when the durable publish fails, got nil")
	}

	cfgPath, _, _ := cs.mvPaths("default/mv")
	assertMarkerCleaned(t, tr, cfgPath)
	if _, ok := cs.GetMultiVector("mv"); ok {
		t.Error("multi-vector index registered in memory despite a failed create")
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	if _, ok := cs2.GetMultiVector("mv"); ok {
		t.Error("multi-vector index reloaded after a failed create")
	}
}
