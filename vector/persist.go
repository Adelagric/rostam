// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/rostamlabs/rostam/vector/analysis"
)

// Native persistence (skip-rebuild instant restart).
//
// SavePersist + OpenPersist let an index be reopened by *mapping* its files
// rather than rebuilding the HNSW graph. The two big arrays — the float32
// vectors and the level-0 adjacency slab — already live in their mmap files
// (Config.MmapPath, Config.GraphMmapPath) and survive process exit, so on open
// they are re-mapped zero-copy (lazy page-in). Everything else needed to make
// them usable is small and written to a sidecar: per-slot ids, node levels,
// level-0 lengths, the rare upper-level edges, and the entry-point/max-level
// scalars. The int8 codes are re-encoded from the vectors in O(n) (a few seconds
// at 1M vs ~a minute to rebuild the graph).
//
// Scope: the index must be MmapPath+GraphMmapPath-backed and "dense" — no
// tombstoned/freed slots (SavePersist refuses rather than silently dropping
// them). Per-slot metadata, TTL, and sparse vectors are all persisted, and the
// payload + sparse indexes are rebuilt on open. This matches the bulk-loaded,
// read-mostly index that instant restart is for. Like the mmap files, the
// sidecar is a native-machine artifact, not a portable format.

var persistMagic = [8]byte{'R', 'S', 'T', 'M', 'I', '1', '\n', 0}

// persistVersion 2 added the metadata + expires (TTL) blocks; v3 the sparse
// block; v4 the free + tombstoned slot sets (so a non-dense index can be saved);
// v5 the keyExpires (per-key payload TTL) block, appended after the sparse block;
// v6 the versions (per-point CAS version) block, appended after keyExpires. v7
// added a PQ codebook block right after the header (before the per-slot arrays):
// a presence byte (1 only for a trained QuantPQ index), then M + the codebooks,
// so a restored PQ-HNSW index re-encodes codes from the TRAINED codebooks (ADC
// navigation survives restart) instead of degrading to exact-float. Earlier
// versions (no released artifacts) are not read. v4/v5/v6 sidecars remain readable
// (each newer block is gated on version >= its introduction); a v<6 sidecar
// defaults every live point to version 1; a v<7 sidecar has no PQ block (a PQ
// index re-encodes with an untrained codec → exact-float, the pre-Task-2
// behaviour). A non-PQ index NEVER emits v7 — it stays at v6 (persistVersionNoPQ)
// and writes no PQ block, so a non-PQ sidecar is byte-identical to pre-Task-2.
// v8 appends the OPQ rotation R (dim×dim floats) to the PQ block; it is stamped
// ONLY when the trained codec has a non-nil rotation, so a PQ sidecar WITHOUT OPQ
// stays at v7 (byte-identical to pre-OPQ). v7 reads back rotation nil; v8 restores
// R VERBATIM so the codec rotates identically (re-encoded codes match originals).
const persistVersion = 8

// persistVersionPQDrop (v9) is stamped ONLY for a float-drop PQ-HNSW sidecar
// (PQDropVecs ⇒ arena.vecsDropped). The resident floats are gone, so the sidecar
// does NOT need the vecs mmap file: it writes a dropped marker + the PQ codes
// VERBATIM (the codes are the only source of truth post-drop) and openPersist
// restores them WITHOUT re-encoding and WITHOUT mapping the vecs file. v6/v7/v8
// sidecars stay unchanged (dropped=false, vecs file mapped as today). Stamped only
// when dropped, so every keep-floats on-disk format is byte-identical to before.
const persistVersionPQDrop = 9

// persistVersionSQ (v10) is stamped ONLY for a QuantSQ sidecar (the trained
// metric-agnostic scalar quantizer). It carries an SQ-ranges block (presence
// byte, bits, per-dimension min[]/max[]) after the header, mirroring the v7 PQ
// block. On open the ranges rebuild the trainedSQ BEFORE restoreDense re-encodes
// codes from the mapped vectors, so re-encoded codes are identical. Stamped ONLY
// for QuantSQ — every QuantNone/SQ8/BQ1/PQ sidecar stays at its existing version
// and is BYTE-IDENTICAL.
const persistVersionSQ = 10

// persistVersionPRQ (v11) is stamped ONLY for a QuantPRQ sidecar (product-residual
// quantization). It carries a PRQ codebook block (presence byte, layer count L,
// R-present byte, per-layer PQ codebooks, then R once when present) after the
// header, mirroring the v7 PQ block but for L layers. On open the L layers'
// codebooks (+R) rebuild the prq codec BEFORE restoreDense re-encodes codes from the
// mapped vectors, so re-encoded codes are identical. Stamped ONLY for QuantPRQ —
// every QuantNone/SQ8/BQ1/PQ/SQ sidecar stays at its existing version and is
// BYTE-IDENTICAL.
const persistVersionPRQ = 11

// persistVersionPQNoOPQ is the version stamped for a trained QuantPQ sidecar
// WITHOUT OPQ (rotation nil): PQ block, no R, byte-identical to pre-OPQ v7.
const persistVersionPQNoOPQ = 7

// persistVersionNoPQ is the version stamped for a non-PQ sidecar (the historical
// v6): no version bump, no PQ block, byte-identical to pre-Task-2.
const persistVersionNoPQ = 6

// persistMinReadVersion is the oldest sidecar version openPersist can read. v4
// sidecars (no per-key TTL block) load with keyExpires left nil — backward
// compatible across the per-key-payload-TTL upgrade.
const persistMinReadVersion = 4

// Error sentinels for native persistence.
var (
	ErrPersistUnsupported = errors.New("vector: SavePersist/OpenPersist require an mmap-backed index (MmapPath + GraphMmapPath)")
	ErrPersistFormat      = errors.New("vector: persist sidecar has invalid magic or version")
	ErrPersistMismatch    = errors.New("vector: persist sidecar does not match the provided Config (dim/metric/M/quant)")
)

// SavePersist flushes the mmap-backed vectors and graph to disk and writes the
// sidecar at metaPath. After it returns, the index can be reopened with
// OpenPersist(cfg, metaPath) using the same Config (same MmapPath/GraphMmapPath).
// Requires an mmap-backed index (else ErrPersistUnsupported). Deleted records
// are handled: tombstoned slots and reclaimed (free) holes are persisted, so a
// Persistent collection can be flushed after Delete/DeleteByFilter. Metadata,
// TTL, and sparse vectors are persisted. Safe to call while live (read lock).
func (h *hnsw) SavePersist(metaPath string) error {
	// A failed mmap slab growth freed the backing region; reject rather than
	// flush/serialize from it (see arena.poisoned / ErrIndexPoisoned).
	if h.arena.poisoned.Load() {
		return ErrIndexPoisoned
	}
	// Exclusive side of the link barrier, for the same reason Snapshot takes it,
	// and in the same order — BEFORE h.mu, never after. writeMeta walks every
	// node's level0Len and upper-level edges; an insert holds the barrier's read
	// side from before its placement until after its link phase, so this waits
	// out any insert in flight and can never serialize a placed-but-unlinked
	// node, which would reopen as an unreachable orphan.
	h.linkMu.Lock()
	defer h.linkMu.Unlock()
	h.mu.RLock()
	defer h.mu.RUnlock()

	// Re-check under h.mu: poison + the nil-out are published under the write
	// lock, so a grow that failed in the window between the pre-lock check and
	// here would otherwise let this serialize (in a no-recover FSM/backup
	// goroutine) over the freed regions — a truncated sidecar or a crash.
	if h.arena.poisoned.Load() {
		return ErrIndexPoisoned
	}

	if h.graphMmapF == nil || h.arena.mmapF == nil {
		return ErrPersistUnsupported
	}
	n := h.arena.Capacity()

	// Flush the mmap'd vectors and graph so the sidecar we write references
	// durable bytes.
	if err := syncVecMmap(h.arena.mmapF, h.arena.mmapRegion); err != nil {
		return err
	}
	if err := syncVecMmap(h.graphMmapF, h.graphRegion); err != nil {
		return err
	}

	tmp := metaPath + ".tmp"
	f, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600) //nolint:gosec // caller-supplied path
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<16)
	if err := h.writeMeta(bw, n); err != nil {
		_ = f.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return renameDurable(tmp, metaPath) // atomic publish (rename + dir fsync)
}

// writeMeta serializes the sidecar: header + per-slot ids/levels/level0Len +
// upper-level edges. n is the dense slot count (arena capacity).
func (h *hnsw) writeMeta(w io.Writer, n int) error {
	if _, err := w.Write(persistMagic[:]); err != nil {
		return err
	}
	// A QuantPQ sidecar is stamped v7 and carries the PQ codebook block; a non-PQ
	// sidecar stays at v6 (persistVersionNoPQ) and writes NO block, so it is
	// byte-identical to pre-Task-2.
	pqIndex := h.cfg.Quant == QuantPQ
	sqIndex := h.cfg.Quant == QuantSQ
	prqIndex := h.cfg.Quant == QuantPRQ
	hasR := pqIndex && pqRotation(h.arena.quant) != nil
	dropped := h.arena.vecsDropped // float-drop PQ-HNSW (implies pqIndex + trained)
	ver := uint32(persistVersionNoPQ)
	switch {
	case dropped:
		ver = persistVersionPQDrop // v9: dropped marker + verbatim codes, no vecs file
	case prqIndex:
		ver = persistVersionPRQ // v11: PRQ codebook block (L layers + R once)
	case sqIndex:
		ver = persistVersionSQ // v10: SQ-ranges block (trained scalar quantizer)
	case hasR:
		ver = persistVersion // v8: PQ block + OPQ rotation R
	case pqIndex:
		ver = persistVersionPQNoOPQ // v7: PQ block, no R (byte-identical to pre-OPQ)
	}
	hdr := []uint32{
		ver,
		uint32(h.cfg.Dim),    //nolint:gosec
		uint32(h.cfg.Metric), //nolint:gosec
		uint32(h.cfg.M),      //nolint:gosec
		uint32(h.cfg.Quant),  //nolint:gosec
		uint32(n),            //nolint:gosec
		h.entryPoint,
	}
	for _, v := range hdr {
		if err := writeU32(w, v); err != nil {
			return err
		}
	}
	if err := writeI32(w, int32(h.maxLevel)); err != nil { //nolint:gosec
		return err
	}
	// v7: PQ codebook block (after the header, before the per-slot arrays) — written
	// ONLY for a QuantPQ index. It carries the trained codebooks so restoreDense
	// re-encodes from the TRAINED codebooks (ADC navigation survives restart)
	// instead of degrading to exact-float. The presence byte inside is 0 for an
	// untrained codec (a PQ index saved before BuildConcurrent). Shared writer with
	// the snapshot stream (writePQCodebooks).
	if pqIndex {
		if err := writePQCodebooks(w, h.arena.quant, hasR && !dropped); err != nil {
			return err
		}
	}
	// v10: SQ-ranges block — written ONLY for a QuantSQ sidecar (a non-SQ sidecar
	// emits nothing here). On open the ranges rebuild the trainedSQ BEFORE the codes
	// re-encode, so re-encoded codes are identical. Shared writer with the snapshot.
	if sqIndex {
		if err := writeSQRanges(w, h.arena.quant); err != nil {
			return err
		}
	}
	// v11: PRQ codebook block — written ONLY for a QuantPRQ sidecar (every other mode
	// emits nothing here). On open the L layers' codebooks (+R once) rebuild the prq
	// codec BEFORE the codes re-encode, so re-encoded codes are identical.
	if prqIndex {
		if err := writePRQCodebooks(w, h.arena.quant); err != nil {
			return err
		}
	}
	// v9 dropped trailer (after the PQ codebook block, before the per-slot arrays):
	// [hasR byte][R floats?][codeLen u32][len(codes) u32][codes]. The codes are the
	// ONLY source of truth after the drop (no floats to re-encode from), so they are
	// written VERBATIM; openPersist restores them without re-encoding and without
	// mapping the (gone) vecs file. R is carried here so loadCodebooks(cb, R) happens
	// in one shot before the arena is marked dropped.
	if dropped {
		rb := byte(0)
		if hasR {
			rb = 1
		}
		if err := writeByte(w, rb); err != nil {
			return err
		}
		if hasR {
			for _, v := range pqRotation(h.arena.quant) {
				if err := writeF32(w, v); err != nil {
					return err
				}
			}
		}
		if err := writeU32(w, uint32(h.arena.codeLen)); err != nil {
			return err
		}
		if err := writeU32(w, uint32(len(h.arena.codes))); err != nil {
			return err
		}
		if _, err := w.Write(h.arena.codes); err != nil {
			return err
		}
	}
	// Per-slot arrays.
	for slot := 0; slot < n; slot++ {
		if err := writeU64(w, h.arena.ids[slot]); err != nil {
			return err
		}
	}
	for slot := 0; slot < n; slot++ {
		nd := h.nodes[slot]
		lvl := 0
		if nd != nil {
			lvl = nd.level
		}
		if err := writeU32(w, uint32(lvl)); err != nil { //nolint:gosec
			return err
		}
	}
	for slot := 0; slot < n; slot++ {
		if err := writeU32(w, uint32(h.level0Len[slot])); err != nil {
			return err
		}
	}
	// Dead-slot sets so the index need not be dense: free (reclaimed) slots are
	// holes skipped on open; tombstoned (deleted, not yet reclaimed) slots stay
	// in the graph but are filtered from results. Persisting them lets a
	// Persistent collection be flushed after Delete/DeleteByFilter.
	if err := writeU32(w, uint32(len(h.arena.free))); err != nil { //nolint:gosec
		return err
	}
	for _, slot := range h.arena.free {
		if err := writeU32(w, slot); err != nil {
			return err
		}
	}
	if err := writeU32(w, uint32(len(h.tombstoned))); err != nil { //nolint:gosec
		return err
	}
	for slot := range h.tombstoned {
		if err := writeU32(w, slot); err != nil {
			return err
		}
	}
	// Upper-level edges (only nodes with level >= 1).
	var upperNodes []uint32
	for slot := 0; slot < n; slot++ {
		if nd := h.nodes[slot]; nd != nil && nd.level >= 1 {
			upperNodes = append(upperNodes, uint32(slot)) //nolint:gosec
		}
	}
	if err := writeU32(w, uint32(len(upperNodes))); err != nil { //nolint:gosec
		return err
	}
	for _, slot := range upperNodes {
		nd := h.nodes[slot]
		if err := writeU32(w, slot); err != nil {
			return err
		}
		for lc := 1; lc <= nd.level; lc++ {
			edges := nd.upper[lc-1]
			if err := writeU32(w, uint32(len(edges))); err != nil { //nolint:gosec
				return err
			}
			for _, e := range edges {
				if err := writeU32(w, e); err != nil {
					return err
				}
			}
		}
	}

	// Expires (TTL) block: present-only [slot u32, deadline u64] pairs, so a
	// no-TTL index pays just a zero count.
	var withExpires []uint32
	for slot := 0; slot < n; slot++ {
		if h.arena.expires[slot] != 0 {
			withExpires = append(withExpires, uint32(slot)) //nolint:gosec
		}
	}
	if err := writeU32(w, uint32(len(withExpires))); err != nil { //nolint:gosec
		return err
	}
	for _, slot := range withExpires {
		if err := writeU32(w, slot); err != nil {
			return err
		}
		if err := writeU64(w, h.arena.expires[slot]); err != nil {
			return err
		}
	}

	// Metadata block: present-only [slot u32, len u32, {key, value}...], reusing
	// the snapshot value encoding. Rebuilds the payload index on open.
	var withMeta []uint32
	for slot := 0; slot < n; slot++ {
		if h.arena.metadata[slot] != nil {
			withMeta = append(withMeta, uint32(slot)) //nolint:gosec
		}
	}
	if err := writeU32(w, uint32(len(withMeta))); err != nil { //nolint:gosec
		return err
	}
	for _, slot := range withMeta {
		m := h.arena.metadata[slot]
		if err := writeU32(w, slot); err != nil {
			return err
		}
		if err := writeU32(w, uint32(len(m))); err != nil { //nolint:gosec
			return err
		}
		for key, val := range m {
			if err := writeString(w, key); err != nil {
				return err
			}
			if err := writeValue(w, val); err != nil {
				return err
			}
		}
	}

	// Sparse block: present-only [slot u32, nnz u32, {dim u32, value f32}...],
	// reusing the snapshot encoding. The sparse inverted index is rebuilt on open.
	var withSparse []uint32
	for slot := 0; slot < n; slot++ {
		if h.arena.sparse[slot] != nil {
			withSparse = append(withSparse, uint32(slot)) //nolint:gosec
		}
	}
	if err := writeU32(w, uint32(len(withSparse))); err != nil { //nolint:gosec
		return err
	}
	for _, slot := range withSparse {
		sv := h.arena.sparse[slot]
		if err := writeU32(w, slot); err != nil {
			return err
		}
		if err := writeU32(w, uint32(len(sv.Indices))); err != nil { //nolint:gosec
			return err
		}
		for i, dim := range sv.Indices {
			if err := writeU32(w, dim); err != nil {
				return err
			}
			if err := writeF32(w, sv.Values[i]); err != nil {
				return err
			}
		}
	}

	// KeyExpires (per-key payload TTL) block (v5): present-only
	// [slot u32, count u32, {key string, deadline u64}...]. Absolute unix-ms
	// deadlines preserved verbatim so an instant-restart is time-stable. A
	// no-per-key-TTL index pays just a zero count.
	var withKeyExp []uint32
	for slot := 0; slot < n; slot++ {
		if len(h.arena.keyExpires[slot]) > 0 {
			withKeyExp = append(withKeyExp, uint32(slot)) //nolint:gosec
		}
	}
	if err := writeU32(w, uint32(len(withKeyExp))); err != nil { //nolint:gosec
		return err
	}
	for _, slot := range withKeyExp {
		ke := h.arena.keyExpires[slot]
		if err := writeU32(w, slot); err != nil {
			return err
		}
		if err := writeU32(w, uint32(len(ke))); err != nil { //nolint:gosec
			return err
		}
		for key, dl := range ke {
			if err := writeString(w, key); err != nil {
				return err
			}
			if err := writeU64(w, dl); err != nil {
				return err
			}
		}
	}

	// Versions (per-point CAS version) block (v6): present-only
	// [slot u32, version u64] pairs (a no-version index pays just a zero count).
	// Versions are restored verbatim so an instant-restart keeps each point's
	// version (NOT re-bumped).
	var withVer []uint32
	for slot := 0; slot < n; slot++ {
		if h.arena.versions[slot] != 0 {
			withVer = append(withVer, uint32(slot)) //nolint:gosec
		}
	}
	if err := writeU32(w, uint32(len(withVer))); err != nil { //nolint:gosec
		return err
	}
	for _, slot := range withVer {
		if err := writeU32(w, slot); err != nil {
			return err
		}
		if err := writeU64(w, h.arena.versions[slot]); err != nil {
			return err
		}
	}
	return nil
}

// OpenPersist reopens an index saved with SavePersist, mapping its vector and
// graph files (zero-copy) instead of rebuilding the graph. cfg must match the
// saved index and carry the same MmapPath + GraphMmapPath (the existing files).
// Open cost is O(n) — re-encode codes, rebuild idMap and node structs — with no
// graph linking.
func openPersist(cfg Config, metaPath string) (*hnsw, error) {
	// A Persistent Vamana reopen carries the user-facing config (R/L/alpha and often a
	// zero M). Normalize the Vamana defaults the same way newVamana did at build time —
	// BEFORE Validate and the header M-match — so the restored geometry (m0 = VamanaR,
	// M = 16) reproduces the built index. No-op for HNSW (the default reopen path).
	if cfg.IndexType == IndexVamana {
		cfg = applyVamanaDefaults(cfg)
	}
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}
	if cfg.MmapPath == "" || cfg.GraphMmapPath == "" {
		return nil, ErrPersistUnsupported
	}
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, err
	}
	r := bytes.NewReader(data)

	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, err
	}
	if magic != persistMagic {
		return nil, ErrPersistFormat
	}
	hdr := make([]uint32, 7)
	for i := range hdr {
		if hdr[i], err = readU32(r); err != nil {
			return nil, err
		}
	}
	version, dim, metric, m, quant, n, entryPoint := hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6]
	if version < persistMinReadVersion || version > persistVersionPRQ {
		return nil, ErrPersistFormat
	}
	if int(dim) != cfg.Dim || Metric(metric) != cfg.Metric || int(m) != cfg.M || QuantMode(quant) != cfg.Quant {
		return nil, ErrPersistMismatch
	}
	maxLevel, err := readI32(r)
	if err != nil {
		return nil, err
	}

	// v7+: PQ codebook block (after the header). Read into a local now; it is
	// loaded into the shell's pqQuantizer below, BEFORE restoreDense re-encodes
	// codes from the mapped vectors, so a restored PQ-HNSW index re-encodes with
	// the TRAINED codebooks (ADC navigation survives restart) instead of degrading
	// to exact-float. v<7 sidecars have no block (read nothing). nil when the
	// presence byte is 0 (non-PQ index, or an untrained PQ index at save time).
	var pqCodebooks [][][]float32
	var pqRot []float32
	dropped := version == persistVersionPQDrop
	// v9 dropped trailer fields (read after the codebook block, below).
	var droppedCodeLen int
	var droppedCodes []byte
	// v10 SQ-ranges (loaded into the trainedSQ below, before restoreDense re-encodes).
	var sqMin, sqMax []float32
	var sqBits int
	// v11 PRQ codebooks (loaded into the prqQuantizer below, before restoreDense).
	var prqCodebooks [][][][]float32
	var prqRot []float32
	if version == persistVersionPRQ {
		// v11 (QuantPRQ): a PRQ codebook block (L layers + R once) sits where the PQ
		// block would be. Read it now so the L layers' codebooks (+R) load into the
		// prqQuantizer BEFORE restoreDense re-encodes codes from the mapped vectors
		// (identical codes). nil cb ⇒ untrained at save.
		if prqCodebooks, prqRot, err = readPRQCodebooksRaw(r, int(dim)); err != nil {
			return nil, err
		}
	} else if version == persistVersionSQ {
		// v10 (QuantSQ): an SQ-ranges block sits where the PQ block would be. Read
		// the ranges now so they load into the trainedSQ BEFORE restoreDense re-encodes
		// codes from the mapped vectors (identical codes). nil min ⇒ untrained at save.
		if sqMin, sqMax, sqBits, err = readSQRangesRaw(r, int(dim)); err != nil {
			return nil, err
		}
	} else if version >= 7 {
		if dropped {
			// v9: codebooks are R-less here; the dropped trailer carries R + the
			// VERBATIM codes. Read them now so the codec is loaded BEFORE restoreDense
			// (which is skipped for the dropped path — codes are not re-encoded).
			if pqCodebooks, _, err = readPQCodebooksRaw(r, int(dim), false); err != nil {
				return nil, err
			}
			if pqCodebooks == nil {
				return nil, ErrPersistFormat // float-drop sidecar must carry codebooks
			}
			rb, rerr := readByte(r)
			if rerr != nil {
				return nil, rerr
			}
			if rb == 1 {
				pqRot = make([]float32, int(dim)*int(dim))
				for i := range pqRot {
					f, ferr := readF32(r)
					if ferr != nil {
						return nil, ferr
					}
					pqRot[i] = f
				}
			}
			cl, clerr := readU32(r)
			if clerr != nil {
				return nil, clerr
			}
			droppedCodeLen = int(cl)
			cn, cnerr := readU32(r)
			if cnerr != nil {
				return nil, cnerr
			}
			droppedCodes = make([]byte, int(cn))
			if _, rderr := io.ReadFull(r, droppedCodes); rderr != nil {
				return nil, rderr
			}
		} else if pqCodebooks, pqRot, err = readPQCodebooksRaw(r, int(dim), version >= persistVersion); err != nil {
			// version >= 8 ⇒ the PQ block carries the OPQ rotation R after the
			// codebooks, restored VERBATIM so the loaded codec rotates identically.
			return nil, err
		}
	}

	ids := make([]uint64, n)
	for i := range ids {
		if ids[i], err = readU64(r); err != nil {
			return nil, err
		}
	}
	levels := make([]int, n)
	for i := range levels {
		v, rerr := readU32(r)
		if rerr != nil {
			return nil, rerr
		}
		levels[i] = int(v)
	}
	level0Len := make([]uint16, n)
	for i := range level0Len {
		v, rerr := readU32(r)
		if rerr != nil {
			return nil, rerr
		}
		level0Len[i] = uint16(v) //nolint:gosec // bounded by 2*M <= 256
	}

	// Dead-slot sets: free (reclaimed) slots are holes — skipped when building
	// the id map and node table, but kept on the free list for reuse; tombstoned
	// slots stay live-in-graph (filtered from results via admits).
	freeSlots, err := readU32Slice(r)
	if err != nil {
		return nil, err
	}
	tombSlots, err := readU32Slice(r)
	if err != nil {
		return nil, err
	}
	holes := make(map[uint32]bool, len(freeSlots))
	for _, s := range freeSlots {
		holes[s] = true
	}

	// Construct the index shell (no fresh mmap — we attach existing files).
	h, err := newPersistShell(cfg)
	if err != nil {
		return nil, err
	}

	// v9 (float-drop): the vecs file is gone — do NOT map it. Load the codebooks
	// (+R), then the codes VERBATIM, and mark the arena dropped WITHOUT re-encoding
	// (there are no floats to encode from). v6/v7/v8 map the vecs file and re-encode
	// codes from it as before, byte-identically.
	if dropped {
		pq, ok := h.arena.quant.(*pqQuantizer)
		if !ok {
			_ = h.Close()
			return nil, ErrPersistFormat // float-drop sidecar but index is not QuantPQ
		}
		pq.loadCodebooks(pqCodebooks, pqRot)
		// Validate the verbatim codes against the loaded codec before installing:
		// codeLen must match, the buffer must cover exactly len(ids) slots, and
		// every sub-code index must be in range. A corrupt sidecar would otherwise
		// OOB-index a codebook in arena.Code/reconstruct/adc and panic at query time.
		if droppedCodeLen != pq.CodeLen() || droppedCodeLen <= 0 ||
			len(droppedCodes) != len(ids)*droppedCodeLen {
			_ = h.Close()
			return nil, ErrPersistFormat
		}
		if verr := pq.validateCodes(droppedCodes, droppedCodeLen); verr != nil {
			_ = h.Close()
			return nil, ErrPersistFormat
		}
		h.arena.restoreDenseDropped(ids, holes, droppedCodeLen, droppedCodes)
	} else {
		// Attach the existing vector + graph mmap files (map at their current size,
		// no truncation), then re-encode codes and rebuild the id map.
		if err := h.arena.loadMmapVecs(cfg.MmapPath, int(n)); err != nil {
			_ = h.Close()
			return nil, err
		}
		// Load the trained PQ codebooks into the shell's quantizer BEFORE restoreDense
		// re-encodes codes from the mapped vectors, so the re-encode is deterministic
		// and produces real ADC codes (a restored PQ-HNSW index navigates on ADC, not
		// the exact-float degrade). nil for non-PQ or v<7 sidecars (re-encode as before).
		if pqCodebooks != nil {
			pq, ok := h.arena.quant.(*pqQuantizer)
			if !ok {
				_ = h.Close()
				return nil, ErrPersistFormat // PQ codebook block but index is not QuantPQ
			}
			pq.loadCodebooks(pqCodebooks, pqRot)
		}
		// Load the learned SQ ranges into the shell's trainedSQ BEFORE restoreDense
		// re-encodes codes from the mapped vectors, so the re-encode is deterministic
		// and produces identical codes (a restored HNSW-SQ index navigates on real
		// codes, not the untrained exact-float degrade). nil for non-SQ or an untrained
		// SQ sidecar (re-encode is a no-op then, as before).
		if sqMin != nil {
			sq, ok := h.arena.quant.(*trainedSQ)
			if !ok {
				_ = h.Close()
				return nil, ErrPersistFormat // SQ ranges block but index is not QuantSQ
			}
			// Apply the serialized bit-depth too (defense-in-depth, MN2): a 4/6-bit SQ
			// sidecar decodes its bit-packed codes correctly even if the rebuilt config's
			// SQBits were wrong, since these bits came from the SAME trained quantizer.
			sq.loadRangesBits(sqMin, sqMax, sqBits)
		}
		// Load the trained PRQ layer codebooks (+R once) into the shell's quantizer
		// BEFORE restoreDense re-encodes codes from the mapped vectors, so the re-encode
		// is deterministic and produces identical codes (a restored HNSW-PRQ index
		// navigates on the summed-LUT ADC, not the untrained exact-float degrade). nil
		// for non-PRQ or an untrained PRQ sidecar (re-encode is a no-op then, as before).
		if prqCodebooks != nil {
			prq, ok := h.arena.quant.(*prqQuantizer)
			if !ok {
				_ = h.Close()
				return nil, ErrPersistFormat // PRQ codebook block but index is not QuantPRQ
			}
			prq.loadCodebooks(prqCodebooks, prqRot)
		}
		h.arena.restoreDense(ids, holes)
	}
	h.arena.free = append(h.arena.free[:0], freeSlots...) // reclaimed holes, reusable
	for _, s := range tombSlots {
		h.tombstoned[s] = true
	}
	if err := h.loadGraphMmap(cfg.GraphMmapPath, int(n)); err != nil {
		_ = h.Close()
		return nil, err
	}
	h.level0Len = level0Len

	// Rebuild node structs (level + upper). Level-0 edges already live in the
	// mapped graph slab; level0Len bounds them. Reclaimed (free) holes get no
	// node — their dangling in-edges are tolerated by searchLayer.
	h.nodes = make([]*node, n)
	for slot := uint32(0); slot < n; slot++ {
		if holes[slot] {
			continue
		}
		h.nodes[slot] = &node{slot: slot, level: levels[slot], upper: makeUpper(levels[slot])}
	}
	upperNodes, err := readU32(r)
	if err != nil {
		return nil, err
	}
	for i := uint32(0); i < upperNodes; i++ {
		slot, rerr := readU32(r)
		if rerr != nil {
			_ = h.Close()
			return nil, rerr
		}
		// Guard the upper-edge slot index against a corrupt/truncated sidecar:
		// slot must be in range and reference a live node (reclaimed holes leave
		// a nil entry). Mirrors the bounds checks in the expires/metadata/sparse
		// blocks below; without it an out-of-range slot panics and a hole slot
		// nil-derefs on nd.level.
		if slot >= n || h.nodes[slot] == nil {
			_ = h.Close()
			return nil, ErrPersistFormat
		}
		nd := h.nodes[slot]
		for lc := 1; lc <= nd.level; lc++ {
			cnt, cerr := readU32(r)
			if cerr != nil {
				_ = h.Close()
				return nil, cerr
			}
			row := make([]uint32, cnt)
			for j := range row {
				if row[j], cerr = readU32(r); cerr != nil {
					_ = h.Close()
					return nil, cerr
				}
			}
			nd.upper[lc-1] = row
		}
	}

	// Expires (TTL) block.
	expCount, err := readU32(r)
	if err != nil {
		_ = h.Close()
		return nil, err
	}
	for i := uint32(0); i < expCount; i++ {
		slot, serr := readU32(r)
		if serr != nil {
			_ = h.Close()
			return nil, serr
		}
		deadline, derr := readU64(r)
		if derr != nil {
			_ = h.Close()
			return nil, derr
		}
		if int(slot) < len(h.arena.expires) {
			h.arena.expires[slot] = deadline
		}
	}

	// Metadata block, then rebuild the payload (equality/range) index from it.
	metaCount, err := readU32(r)
	if err != nil {
		_ = h.Close()
		return nil, err
	}
	for i := uint32(0); i < metaCount; i++ {
		slot, serr := readU32(r)
		if serr != nil {
			_ = h.Close()
			return nil, serr
		}
		entries, eerr := readU32(r)
		if eerr != nil {
			_ = h.Close()
			return nil, eerr
		}
		m := make(Metadata, entries)
		for j := uint32(0); j < entries; j++ {
			key, kerr := readString(r)
			if kerr != nil {
				_ = h.Close()
				return nil, kerr
			}
			val, verr := readValue(r)
			if verr != nil {
				_ = h.Close()
				return nil, verr
			}
			m[key] = val
		}
		if int(slot) < len(h.arena.metadata) {
			h.arena.metadata[slot] = m
		}
	}
	h.payloadIdx.rebuild(h.arena)

	// Sparse block, then rebuild the sparse inverted index from it.
	sparseCount, err := readU32(r)
	if err != nil {
		_ = h.Close()
		return nil, err
	}
	for i := uint32(0); i < sparseCount; i++ {
		slot, serr := readU32(r)
		if serr != nil {
			_ = h.Close()
			return nil, serr
		}
		nnz, nerr := readU32(r)
		if nerr != nil {
			_ = h.Close()
			return nil, nerr
		}
		sv := &SparseVector{Indices: make([]uint32, nnz), Values: make([]float32, nnz)}
		for j := uint32(0); j < nnz; j++ {
			dim, derr := readU32(r)
			if derr != nil {
				_ = h.Close()
				return nil, derr
			}
			val, verr := readF32(r)
			if verr != nil {
				_ = h.Close()
				return nil, verr
			}
			sv.Indices[j] = dim
			sv.Values[j] = val
		}
		if int(slot) < len(h.arena.sparse) {
			h.arena.sparse[slot] = sv
		}
	}
	h.sparseIdx.rebuild(h.arena, h.tombstoned)

	// BM25 full-text index: like sparseIdx/payloadIdx it is NEVER serialized — it
	// is rebuilt from each restored slot's $content (a persisted metadata field, so
	// it is present in the metadata block above). newPersistShell already allocated
	// h.bm25/h.az from cfg.FullText (nil → no lane, no-op here, byte-identical).
	if h.bm25 != nil {
		h.bm25.rebuild(h.arena, h.tombstoned, h.az)
	}

	// KeyExpires (per-key payload TTL) block (v5+). v4 sidecars have no block;
	// keyExpires stays nil (restoreDense allocated it). Deadlines are restored
	// verbatim (absolute unix-ms) so a pending key TTL is time-stable across the
	// instant restart.
	if version >= 5 {
		keyExpCount, kerr := readU32(r)
		if kerr != nil {
			_ = h.Close()
			return nil, kerr
		}
		for i := uint32(0); i < keyExpCount; i++ {
			slot, serr := readU32(r)
			if serr != nil {
				_ = h.Close()
				return nil, serr
			}
			entries, eerr := readU32(r)
			if eerr != nil {
				_ = h.Close()
				return nil, eerr
			}
			ke := make(map[string]uint64, entries)
			for j := uint32(0); j < entries; j++ {
				key, kkerr := readString(r)
				if kkerr != nil {
					_ = h.Close()
					return nil, kkerr
				}
				dl, derr := readU64(r)
				if derr != nil {
					_ = h.Close()
					return nil, derr
				}
				ke[key] = dl
			}
			if int(slot) < len(h.arena.keyExpires) && len(ke) > 0 {
				h.arena.keyExpires[slot] = ke
			}
		}
	}

	// Versions (per-point CAS version) block (v6+). restoreDense zero-filled
	// versions; v6 fills present slots verbatim, while a v<6 sidecar defaults every
	// live point to version 1 (a sane existing-point version — backward-compatible).
	if version >= 6 {
		verCount, verr := readU32(r)
		if verr != nil {
			_ = h.Close()
			return nil, verr
		}
		for i := uint32(0); i < verCount; i++ {
			slot, serr := readU32(r)
			if serr != nil {
				_ = h.Close()
				return nil, serr
			}
			v, derr := readU64(r)
			if derr != nil {
				_ = h.Close()
				return nil, derr
			}
			if int(slot) < len(h.arena.versions) {
				h.arena.versions[slot] = v
			}
		}
	} else {
		for _, slot := range h.arena.idMap {
			if int(slot) < len(h.arena.versions) {
				h.arena.versions[slot] = 1
			}
		}
	}

	h.entryPoint = entryPoint
	h.maxLevel = int(maxLevel)
	h.insertOps.Store(uint64(n))
	// The Expires/KeyExpires blocks above wrote h.arena.expires/keyExpires
	// directly, bypassing SetExpires/SetKeyExpires's incremental deadline-count
	// maintenance (the TTL sweep's fast-path gate) — recompute once, in bulk.
	h.arena.RecomputeDeadlineCounts()
	return h, nil
}

// readU32Slice reads a [count u32][count × u32] block (used for the free and
// tombstoned slot sets).
func readU32Slice(r io.Reader) ([]uint32, error) {
	count, err := readU32(r)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	out := make([]uint32, count)
	for i := range out {
		if out[i], err = readU32(r); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// newPersistShell builds an hnsw with all the in-memory scaffolding of newHNSW
// but WITHOUT opening fresh mmap files (the caller attaches existing ones).
func newPersistShell(cfg Config) (*hnsw, error) {
	a := newArena(cfg.Dim, 0)
	// Mirrors newHNSW/newIVF: a declared vector cap sizes the slab reservation.
	// Without it an instant-restarted capped collection would fall back to the
	// generic 64x factor and size its reservation from whatever it happened to
	// hold at the crossover, rather than from what it is allowed to grow to.
	a.maxVectorsHint = cfg.MaxVectors
	quant := newQuantizer(cfg.Quant, cfg.Dim, cfg.QuantPQM, cfg.SQBits, cfg.PRQLayers, cfg.PQNBits, cfg.Metric)
	if quant != nil {
		a.setQuant(quant)
	}
	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	h := &hnsw{
		cfg:        cfg,
		arena:      a,
		quant:      quant,
		rng:        rand.New(rand.NewSource(seed)),
		mL:         1.0 / math.Log(float64(cfg.M)),
		m0:         effectiveM0(cfg),
		maxLevel:   -1,
		tombstoned: make(map[uint32]bool),
		now:        func() int64 { return time.Now().UnixMilli() },
		bucket:     newTokenBucket(cfg.MaxInsertsPerSecond),
		sparseIdx:  newSparseIndex(),
		payloadIdx: newPayloadIndex(),
		// Mirror newHNSW: start at 1 so a restored (idMap-repopulated) index forces
		// the first scroll to rebuild its snapshot (scrollSnap.ver==0 != 1).
		idSetVersion: 1,
	}
	h.initPairFns()
	// Vamana single-layer pinning on reopen: mirror newVamana so a restored Vamana
	// index behaves identically to a freshly-built one. mL=0 ⇒ assignLevel always
	// returns 0 (no upper edges ever allocated for a post-reopen Insert); m0 already
	// derives from R via effectiveM0 above; pruneAlpha = VamanaAlpha drives the
	// steady-state RobustPrune for incremental inserts; vamana=true engages the
	// single-layer forwardM/maxM0 budgets. The medoid entry point is restored by the
	// caller from the sidecar header (h.entryPoint).
	if cfg.IndexType == IndexVamana {
		h.mL = 0
		h.vamana = true
		alpha := cfg.VamanaAlpha
		if alpha == 0 {
			alpha = defaultVamanaAlpha
		}
		h.pruneAlpha = alpha
	}
	// Mirror newHNSW's full-text init so a restored FullText collection allocates
	// its bm25 lane + analyzer. A nil FullText leaves h.bm25/h.az nil → byte/
	// behavior-identical to a non-full-text collection; the index is populated from
	// the persisted $content by the caller's rebuild path.
	if cfg.FullText != nil {
		az, err := analysis.ByName(cfg.FullText.Analyzer)
		if err != nil {
			return nil, err
		}
		if _, ok := az.(analysis.CountingAnalyzer); !ok {
			return nil, fmt.Errorf("full-text analyzer %q does not implement CountingAnalyzer", cfg.FullText.Analyzer)
		}
		h.az = az
		h.bm25 = newBM25Index(cfg.FullText.K1, cfg.FullText.B)
	}
	return h, nil
}
