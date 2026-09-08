// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"time"
)

// IVF instant-restart mmap sidecar.
//
// IVF is otherwise snapshot-only: a cold restart reads the ENTIRE snapshot (every
// float vector) back into RAM. The sidecar gives IVF the same instant-restart HNSW
// has (persist.go): the float vectors live in the cfg.MmapPath mmap file (zero-copy,
// lazy page-in, survives process exit) and everything else — centroids, inverted
// lists, slotCell, tombstones, PQ codes/codebooks/OPQ-R, arena metadata — is written
// to a sidecar at the quiescent Flush point. Reopen re-maps the vecs file and reads
// the sidecar; NO full re-read of the float block.
//
// The sidecar IS the IVF snapshot with the float vectors EXTERNALIZED: it reuses
// writeArena/readArena (with the new vecsExternalMode marker, so the inline float
// block is omitted), writeIVFCore/readIVFCore (tombstones + centroids + lists), and
// writePQTrailer/readPQTrailer (the residual PQ codec + slotCell + OPQ R), all shared
// verbatim with Snapshot/Restore. Only the header + the externalized-vecs wiring are
// new. The float-dropped (PQ-only) state round-trips too: vecsDroppedMode writes no
// vecs file and the codes are serialized verbatim by writeArena's tail (the existing
// PQ-only path), so openPersistIVF restores them without re-encoding.
//
// Distinct from the HNSW sidecar magic (RSTMI1) and the IVF snapshot magic
// (rstmivf1): a torn/mismatched artifact can never be misread as either.

// ivfPersistMagic identifies an IVF instant-restart mmap sidecar.
var ivfPersistMagic = [8]byte{'R', 'S', 'T', 'M', 'I', 'V', '1', '\n'}

// ivfPersistVersion is the MAXIMUM sidecar version this binary can write or read.
// v1 = greenfield sidecar; v2 appends the drift-retrain checkpoint (lastTrainCount +
// lastTrainCost) to the IVF core block. A v1 sidecar still reopens (the checkpoint
// reads back as 0 — drift-retrain re-arms on the next train).
// v3 appends the SOAR secondary-assignment block (cellOf2 + per-slot code2) after the
// drift checkpoint, so a SOAR IVF restarts with its multi-assignment intact. At v3 the
// drift checkpoint is ALWAYS written (so the layout is unambiguous — readIVFCore reads
// the checkpoint at version>=2 and the SOAR block at version>=3). v1/v2 sidecars reopen
// as a plain single-assignment IVF (no SOAR block).
// savePersist writes v3 when SOAR is active, v2 when (only) drift is active, else v1 so
// the sidecar is byte-identical to the pre-feature format (old-reader compatible during
// rolling upgrades). See ivfDriftActive / ivfSOARActive.
const ivfPersistVersion = uint32(3)

// writeIVFCore serializes the tombstone set + the IVF state (trained, nprobe,
// centroids, lists). Shared by Snapshot and the mmap sidecar so the two formats
// stay in lockstep. Caller holds the read lock. withDriftMeta appends the
// drift-retrain checkpoint (lastTrainCount:u32 + lastTrainCost:f32) AFTER the lists
// block; the caller passes true only for the new snapshot/sidecar versions that read
// it back (Snapshot v4 / sidecar v2). False = the pre-drift byte layout, so an old
// reader (or a withDriftMeta=false re-serialization) is byte-identical.
func (ix *ivf) writeIVFCore(bw *bufio.Writer, withDriftMeta, withSOAR bool) error {
	if err := writeU32(bw, uint32(len(ix.tombstoned))); err != nil {
		return err
	}
	for s := range ix.tombstoned {
		if err := writeU32(bw, s); err != nil {
			return err
		}
	}
	trained := byte(0)
	if ix.trained {
		trained = 1
	}
	if err := bw.WriteByte(trained); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(ix.nprobe)); err != nil {
		return err
	}
	if err := writeU32(bw, uint32(len(ix.centroids))); err != nil {
		return err
	}
	for _, c := range ix.centroids {
		for _, v := range c {
			if err := writeF32(bw, v); err != nil {
				return err
			}
		}
	}
	if err := writeU32(bw, uint32(len(ix.lists))); err != nil {
		return err
	}
	for _, lst := range ix.lists {
		if err := writeU32(bw, uint32(len(lst))); err != nil {
			return err
		}
		for _, slot := range lst {
			if err := writeU32(bw, slot); err != nil {
				return err
			}
		}
	}
	// Drift-retrain checkpoint (lastTrainCount + lastTrainCost), appended only for the
	// drift-aware snapshot/sidecar versions. Absent in older blobs ⇒ they restore both
	// as 0 (drift simply won't fire until the next train resets them — see readIVFCore).
	if withDriftMeta {
		if err := writeU32(bw, uint32(ix.lastTrainCount)); err != nil { //nolint:gosec // count >= 0
			return err
		}
		if err := writeF32(bw, ix.lastTrainCost); err != nil {
			return err
		}
	}
	// SOAR secondary-assignment block (v5 snapshot / v3 sidecar), appended only when
	// the caller signals the SOAR feature is active. Layout: a 1 byte (soarTrained),
	// then cellOf2 (len + u32 per slot), then code2 (len + per-slot length-prefixed
	// bytes; an absent/IVF-Flat code is length 0). Absent in older blobs ⇒ a restored
	// index has soarTrained=false / empty cellOf2 (a plain single-assignment IVF).
	// withSOAR=false is byte-identical to the pre-SOAR layout (no byte written).
	if withSOAR {
		if err := bw.WriteByte(1); err != nil { // soarTrained marker
			return err
		}
		if err := writeU32(bw, uint32(len(ix.cellOf2))); err != nil {
			return err
		}
		for _, c := range ix.cellOf2 {
			if err := writeU32(bw, c); err != nil {
				return err
			}
		}
		if err := writeU32(bw, uint32(len(ix.code2))); err != nil {
			return err
		}
		for _, code := range ix.code2 {
			if err := writeU32(bw, uint32(len(code))); err != nil {
				return err
			}
			if len(code) > 0 {
				if _, err := bw.Write(code); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// readIVFCore reads the block written by writeIVFCore into ix (tombstones +
// trained/nprobe/centroids/lists). Shared by Restore and the mmap sidecar reader.
// withDriftMeta reads the trailing drift-retrain checkpoint (lastTrainCount +
// lastTrainCost) — true only for the drift-aware versions (Snapshot v4 / sidecar v2)
// that wrote it. False (an OLD snapshot, pre-drift) leaves both at 0 (back-compat:
// the fields simply weren't written, and a 0 lastTrainCount means drift-retrain won't
// fire until the next train sets it).
func (ix *ivf) readIVFCore(br *bufio.Reader, dim int, withDriftMeta, withSOAR bool) error {
	tCount, err := readU32(br)
	if err != nil {
		return err
	}
	ix.tombstoned = make(map[uint32]bool, tCount)
	for i := uint32(0); i < tCount; i++ {
		s, err := readU32(br)
		if err != nil {
			return err
		}
		ix.tombstoned[s] = true
	}
	tb, err := br.ReadByte()
	if err != nil {
		return err
	}
	ix.trained = tb == 1
	np, err := readU32(br)
	if err != nil {
		return err
	}
	ix.nprobe = int(np)
	cCount, err := readU32(br)
	if err != nil {
		return err
	}
	ix.centroids = make([][]float32, cCount)
	for c := range ix.centroids {
		cv := make([]float32, dim)
		for i := range cv {
			f, err := readF32(br)
			if err != nil {
				return err
			}
			cv[i] = f
		}
		ix.centroids[c] = cv
	}
	ix.nlist = len(ix.centroids)
	lCount, err := readU32(br)
	if err != nil {
		return err
	}
	ix.lists = make([][]uint32, lCount)
	for l := range ix.lists {
		n, err := readU32(br)
		if err != nil {
			return err
		}
		lst := make([]uint32, n)
		for i := range lst {
			s, err := readU32(br)
			if err != nil {
				return err
			}
			lst[i] = s
		}
		ix.lists[l] = lst
	}
	// Drift-retrain checkpoint, present only for the drift-aware versions. Old blobs
	// (withDriftMeta=false) leave both fields at their zero value (the *ivf is freshly
	// constructed, so lastTrainCount=0/lastTrainCost=0 already).
	if withDriftMeta {
		ltc, err := readU32(br)
		if err != nil {
			return err
		}
		ix.lastTrainCount = int(ltc)
		ltcost, err := readF32(br)
		if err != nil {
			return err
		}
		ix.lastTrainCost = ltcost
	}
	// SOAR secondary-assignment block (v5 snapshot / v3 sidecar). Present only when
	// the caller signals the blob carries it (withSOAR). Restores soarTrained +
	// cellOf2 + code2 verbatim so rebuildListsLocked is NOT needed — the lists block
	// already carries both memberships. Absent (older blob) ⇒ the fields stay at
	// their freshly-constructed zero values (a plain single-assignment IVF).
	if withSOAR {
		marker, err := br.ReadByte()
		if err != nil {
			return err
		}
		ix.soarTrained = marker == 1
		c2N, err := readU32(br)
		if err != nil {
			return err
		}
		ix.cellOf2 = make([]uint32, c2N)
		for i := range ix.cellOf2 {
			c, err := readU32(br)
			if err != nil {
				return err
			}
			ix.cellOf2[i] = c
		}
		codeN, err := readU32(br)
		if err != nil {
			return err
		}
		ix.code2 = make([][]byte, codeN)
		for i := range ix.code2 {
			ln, err := readU32(br)
			if err != nil {
				return err
			}
			if ln == 0 {
				continue
			}
			b := make([]byte, ln)
			if _, err := io.ReadFull(br, b); err != nil {
				return err
			}
			ix.code2[i] = b
		}
	}
	return nil
}

// savePersist writes the IVF instant-restart mmap sidecar at metaPath, the
// implementation behind ix.SavePersist. It mirrors hnsw.SavePersist:
//
//  1. when the floats are present (not dropped) they live in the cfg.MmapPath mmap
//     file — msync it so the sidecar references durable bytes; PQ-only (dropped)
//     has no vecs file (the codes in the sidecar are the source of truth);
//  2. write the sidecar = header (magic + version + dim/metric/M/capacity) + the
//     arena (writeArena with vecs EXTERNALIZED for the present case, or the dropped
//     codes-verbatim block) + the IVF core (centroids/lists/tombstones) + the PQ
//     trailer (codec/slotCell/OPQ R), all to a temp file, fsync, then atomic rename.
//
// Safe to call while live (read lock), exactly like the HNSW sidecar.
func (ix *ivf) savePersist(metaPath string) error {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	// A failed mmap slab growth freed the arena floats; refuse to persist rather
	// than externalize a truncated/unbacked sidecar (same hazard as Snapshot; runs
	// in the same durable-artifact paths). See arena.poisoned.
	if ix.arena.poisoned.Load() {
		return ErrIndexPoisoned
	}

	a := ix.arena
	vecsPresent := !a.vecsDropped
	if vecsPresent {
		// The floats must live in the mmap file so the sidecar can externalize them.
		// A Persistent IVF is created mmap-backed (newIndex sets useMmap), so this is
		// the normal path; reject a non-mmap arena loudly rather than silently writing
		// an inline-less sidecar with no backing file.
		if a.mmapF == nil {
			return ErrPersistUnsupported
		}
		if err := syncVecMmap(a.mmapF, a.mmapRegion); err != nil {
			return err
		}
	}

	tmp := metaPath + ".tmp"
	f, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600) //nolint:gosec // caller-supplied path
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<16)

	// Choose sidecar version: v2 (with drift checkpoint) only when the feature is
	// active; v1 otherwise (pre-feature format, old-reader compatible for rolling
	// upgrades). Mirrors the conditional version selection in Snapshot.
	driftOn := ivfDriftActive(ix.cfg)
	soarOn := ix.ivfSOARActive()
	persistVer := uint32(1)
	switch {
	case soarOn:
		persistVer = uint32(3)
	case driftOn:
		persistVer = uint32(2)
	}
	// At v3 the core block carries the drift checkpoint regardless of driftOn (so the
	// layout is unambiguous for the reader).
	writeDrift := driftOn || soarOn

	if _, err := bw.Write(ivfPersistMagic[:]); err != nil {
		_ = f.Close()
		return err
	}
	hdr := []uint32{
		persistVer,
		uint32(ix.cfg.Dim),
		uint32(ix.cfg.Metric),
		uint32(ix.cfg.M),
		uint32(ix.cfg.EfConstruction),
		uint32(ix.cfg.EfSearch),
	}
	for _, v := range hdr {
		if err := writeU32(bw, v); err != nil {
			_ = f.Close()
			return err
		}
	}
	if err := writeI64(bw, ix.cfg.Seed); err != nil {
		_ = f.Close()
		return err
	}

	// Arena (vecs externalized when present; codes-verbatim when dropped) + IVF core
	// + PQ trailer — the same writers Snapshot uses, so the sidecar IS the snapshot
	// with the float block lifted into the mmap file.
	if err := ix.writeArena(bw, vecsPresent); err != nil {
		_ = f.Close()
		return err
	}
	if err := ix.writeIVFCore(bw, writeDrift, soarOn); err != nil { // v2: drift, v3: + SOAR
		_ = f.Close()
		return err
	}
	if err := ix.writePQTrailer(bw); err != nil {
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

// openPersistIVF reopens an IVF index saved with SavePersist, mapping its vecs file
// (zero-copy) instead of re-reading every float. cfg must match the saved index and
// carry the same cfg.MmapPath. Mirrors openPersist (persist.go): validate the header
// (magic/version/dim/metric/M — fail loud on mismatch), restore the arena (mapping
// the vecs file, or loading verbatim codes for the dropped case), the IVF core, and
// the PQ trailer, then rebuild the derived payload + sparse indexes. Returns the live
// *ivf. Does NOT re-train or rebuild lists from scratch — everything is restored
// verbatim.
func openPersistIVF(cfg Config, metaPath string) (*ivf, error) {
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
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
	if magic != ivfPersistMagic {
		return nil, ErrPersistFormat
	}
	hdr := make([]uint32, 6)
	for i := range hdr {
		if hdr[i], err = readU32(r); err != nil {
			return nil, err
		}
	}
	// hdr[4]/hdr[5] are EfConstruction/EfSearch — stamped for symmetry with the
	// snapshot scalars but not validated (the reopen cfg supplies the live values).
	version, dim, metric, m := hdr[0], hdr[1], hdr[2], hdr[3]
	if version < 1 || version > ivfPersistVersion {
		return nil, ErrPersistFormat
	}
	if int(dim) != cfg.Dim || Metric(metric) != cfg.Metric || int(m) != cfg.M {
		return nil, ErrPersistMismatch
	}
	seed, err := readI64(r)
	if err != nil {
		return nil, err
	}

	// Build a live IVF shell with the reopen config (newIVF wires the arena's codes
	// side-array / quantizer per the IVF-PQ knobs). The arena it allocates is
	// discarded by readArena, which builds a fresh one (mapping the vecs file or
	// loading verbatim codes). The shell's cfg.IVFPQ etc. drive readArena's codeLen
	// carry-over + readPQTrailer, exactly as Restore relies on.
	//
	// Build the shell WITHOUT pre-mmapping the vecs file: a useMmap on the existing
	// file would Truncate it back to the initial reserve size (mmapInitVectors) and
	// destroy the persisted floats BEFORE readArena's loadMmapVecs re-maps them
	// (and leak that throwaway mapping). We strip MmapPath for the shell only — the
	// real mmapPath is passed straight to readArena below for the zero-copy re-map.
	shellCfg := cfg
	shellCfg.MmapPath = ""
	ix, err := newIVF(shellCfg)
	if err != nil {
		return nil, err
	}
	ix.cfg = cfg // restore the full reopen config (MmapPath etc.) on the live index
	ix.cfg.Seed = seed

	br := bufio.NewReader(r)
	// readArena maps the vecs file zero-copy for the external case (vecsExternalMode
	// + a non-empty mmapPath); for the dropped case it loads codes verbatim and marks
	// the arena dropped (no vecs file). The codebooks needed to interpret the codes
	// are restored by readPQTrailer below; the codes are NEVER re-encoded (the v3
	// codec serializes them verbatim), so codebook-before-decode ordering is not a
	// hazard here (unlike HNSW, which re-encodes from floats).
	if err := ix.readArena(br, int(dim), ivfSnapshotVer, cfg.MmapPath); err != nil {
		_ = ix.Close()
		return nil, err
	}
	if err := ix.readIVFCore(br, int(dim), version >= 2, version >= 3); err != nil {
		_ = ix.Close()
		return nil, err
	}
	if err := ix.readPQTrailer(br, int(dim)); err != nil {
		_ = ix.Close()
		return nil, err
	}

	// Rebuild the derived indexes from the restored arena (mirror Restore).
	ix.payloadIdx = newPayloadIndex()
	ix.payloadIdx.rebuild(ix.arena)
	ix.sparseIdx = newSparseIndex()
	ix.sparseIdx.rebuild(ix.arena, ix.tombstoned)

	if ix.now == nil {
		ix.now = func() int64 { return time.Now().UnixMilli() }
	}
	ix.idSetVersion = 1
	ix.scrollSnap.ver = 0
	ix.dataVersion = 1
	ix.orderSnaps = make(map[orderCacheKey]*orderSnap)
	ix.orderSeq = 0
	ix.insertOps.Store(uint64(ix.arena.Size()))
	return ix, nil
}
