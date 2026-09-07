// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	hraft "github.com/hashicorp/raft"
)

// defaultMaxSegment is the size at which the active segment rotates. Entries never
// straddle segments, so a single entry larger than this still gets its own
// segment.
const defaultMaxSegment = 64 << 20

const segExt = ".wal"

// segment is one append-only log file. base is the index of its first entry and
// names the file (%020d.wal).
type segment struct {
	base uint64
	path string
	f    *os.File
	size int64 // bytes written = next append offset
}

// recPos locates one record within a segment by the segment's INDEX in w.segs
// (not a *segment pointer). This keeps recPos pointer-free, so the potentially
// large offsets slice is scan-free for the GC — the index never appears in a GC
// mark cycle. Front-truncation (which removes segments from the front of w.segs)
// re-bases segIdx across the surviving offsets; that is off the hot path.
type recPos struct {
	segIdx int // index into w.segs
	off    int64
	size   int // total record bytes: frameHdr + payload
}

// WAL is a durable raft LogStore + StableStore: a segmented append-only log with
// an in-memory offset index and a fixed binary record format (codec.go), fsync'd
// per StoreLogs batch. It replaces raft-boltdb for the durable path.
//
// Concurrency: one mutex guards everything. raft serializes StoreLogs on the
// leader and reads via replication goroutines; a single lock keeps the index and
// files consistent, matching bbolt's single-writer model.
// ErrWALPoisoned marks a WAL whose ack-implies-durable contract can no longer be
// trusted: an earlier fsync (segment, directory, or durable-file write) FAILED,
// and on post-4.13 Linux the kernel may have DISCARDED that sync's dirty pages —
// a subsequent fsync on the same file can then return SUCCESS having synced
// nothing (the fsyncgate / PostgreSQL PANIC-on-fsync-failure case). If a later
// StoreLogs or stable-store Set simply retried, the node would ack raft entries
// (or persist a vote) on bytes that are gone, silently counting toward commit
// quorum on data no crash recovery can produce. So the WAL fails CLOSED: the
// first fsync failure trips a terminal latch and every subsequent write entry
// point returns this error. Reads stay open — a read of dropped pages already
// fails loud via the per-record CRC. The latch clears ONLY on a fresh OpenWAL:
// a later fsync that happens to succeed is no proof the device recovered.
// Mirrors vector/wal.go's poisoned latch (the #87 fix), adapted to this store's
// single-mutex model.
var ErrWALPoisoned = errors.New("logstore: WAL poisoned by an earlier fsync failure; reopen required")

type WAL struct {
	mu      sync.Mutex
	dir     string
	segs    []*segment // sorted by base; last is the active (appendable) segment
	first   uint64     // index of the first stored entry, 0 if empty
	offsets []recPos   // offsets[i] locates entry (first + i)
	maxSeg  int64
	sync    bool // fsync per StoreLogs batch (durable) vs page-cache only (fast)

	// poisoned is the terminal fail-closed latch (see ErrWALPoisoned). Guarded by
	// mu like everything else; set at the point an fsync failure is observed,
	// before the error is returned, so no later write can slip past it.
	poisoned bool

	// fsyncf is the file-sync seam: production is (*os.File).Sync; tests inject a
	// failing implementation to exercise the poison latch deterministically. It
	// covers every durability-bearing sync (segments, the directory handle, and
	// writeFileDurable's temp file).
	fsyncf func(*os.File) error

	encBuf  []byte // reused StoreLogs encode buffer (zero-alloc hot path)
	readBuf []byte // reused GetLog read buffer

	stablePath string
	kv         map[string][]byte
}

// OpenWAL opens (or creates) a WAL in dir, recovering any existing segments and
// truncating a torn tail.
//
// sync controls crash durability: true fsyncs the active segment after every
// StoreLogs batch (survives power loss); false writes to the page cache only
// (survives a clean process restart — the data is on disk — but loses the tail on
// a machine crash). Close fsyncs regardless, so a clean shutdown is always
// durable. sync=false is the fast "durability from replication" posture; it still
// avoids bbolt's B+tree and msgpack entirely.
func OpenWAL(dir string, sync bool) (*WAL, error) {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("logstore: mkdir %s: %w", dir, err)
	}
	w := &WAL{
		dir:        dir,
		maxSeg:     defaultMaxSegment,
		sync:       sync,
		stablePath: filepath.Join(dir, "stable"),
		kv:         make(map[string][]byte, 8),
		fsyncf:     (*os.File).Sync,
	}
	if err := w.recover(); err != nil {
		return nil, err
	}
	if err := w.loadStable(); err != nil {
		return nil, err
	}
	return w, nil
}

func (w *WAL) IsMonotonic() bool { return true }

func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	var first error
	// fsync EVERY segment (not just the active one — with sync=false the earlier
	// segments were never fsynced during operation) plus the directory, so a
	// clean shutdown is durable for the whole log.
	for _, s := range w.segs {
		if err := w.fsyncf(s.f); err != nil && first == nil {
			w.poisoned = true // a reopened handle must not trust a later false-success sync
			first = err
		}
	}
	// Always attempt the directory sync, even after a segment fsync failure:
	// in sync=false mode the directory entries of segments that DID sync were
	// never made durable during operation, so skipping this on an unrelated
	// segment's failure could lose whole healthy segments to recovery. Keep the
	// FIRST error for the return value.
	if len(w.segs) > 0 {
		if err := w.syncDir(); err != nil && first == nil {
			first = err
		}
	}
	for _, s := range w.segs {
		if err := s.f.Close(); err != nil && first == nil {
			first = err
		}
	}
	w.segs = nil
	return first
}

// closeAllLocked closes every open segment fd without fsync. Used on the recover
// error path, where the WAL is discarded and there is nothing to make durable.
func (w *WAL) closeAllLocked() {
	for _, s := range w.segs {
		_ = s.f.Close()
	}
	w.segs = nil
}

func (w *WAL) FirstIndex() (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.first, nil
}

func (w *WAL) LastIndex() (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lastLocked(), nil
}

func (w *WAL) lastLocked() uint64 {
	if len(w.offsets) == 0 {
		return 0
	}
	return w.first + uint64(len(w.offsets)) - 1
}

func (w *WAL) GetLog(index uint64, out *hraft.Log) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.offsets) == 0 || index < w.first || index > w.lastLocked() {
		return hraft.ErrLogNotFound
	}
	pos := w.offsets[index-w.first]
	if cap(w.readBuf) < pos.size {
		w.readBuf = make([]byte, pos.size)
	}
	buf := w.readBuf[:pos.size]
	if _, err := w.segs[pos.segIdx].f.ReadAt(buf, pos.off); err != nil {
		return fmt.Errorf("logstore: read index %d: %w", index, err)
	}
	payload := buf[frameHdr:]
	if binary.LittleEndian.Uint32(buf[recLenSize:]) != crc32.ChecksumIEEE(payload) {
		return fmt.Errorf("logstore: index %d: %w", index, errCorrupt)
	}
	return decodeInto(payload, out)
}

func (w *WAL) StoreLog(log *hraft.Log) error { return w.StoreLogs([]*hraft.Log{log}) }

func (w *WAL) StoreLogs(logs []*hraft.Log) error {
	if len(logs) == 0 {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.poisoned {
		return ErrWALPoisoned
	}
	preLen := len(w.segs)
	var preActive *segment
	if preLen > 0 {
		preActive = w.segs[preLen-1]
	}
	for _, l := range logs {
		if err := w.appendLocked(l); err != nil {
			return err
		}
	}
	if !w.sync {
		return nil // page cache only; the write already reached the OS
	}
	// fsync EVERY open segment, not just the active one: a batch that crossed a
	// rotation wrote to more than one segment, and all of them must be durable
	// before the batch is acknowledged. Unchanged segments have no dirty pages,
	// so their fsync is a near-free no-op.
	for _, s := range w.segs {
		if err := w.fsyncf(s.f); err != nil {
			w.poisoned = true // fail closed: see ErrWALPoisoned
			return fmt.Errorf("logstore: fsync segment: %w", err)
		}
	}
	// fsync the directory when its entries changed. Trigger on the ACTIVE segment
	// changing (a rotation created a new file) OR the count changing (a removal) —
	// not on count alone, since a batch that both conflict-truncates and rotates
	// can net a zero count change yet still have created a new segment whose name
	// must be made durable.
	if w.segs[len(w.segs)-1] != preActive || len(w.segs) != preLen {
		return w.syncDir()
	}
	return nil
}

// appendLocked writes l to the active segment. It handles the normal tail append
// and the conflict overwrite (index within the current range => truncate the tail
// first, then append).
func (w *WAL) appendLocked(l *hraft.Log) error {
	last := w.lastLocked()
	switch {
	case len(w.offsets) == 0 || l.Index == last+1:
		// normal append
	case l.Index >= w.first && l.Index <= last:
		if err := w.truncateTailLocked(l.Index); err != nil { // drop >= l.Index, then append
			return err
		}
	default:
		// forward gap after a monotonic clear: reset and start a fresh run
		if err := w.resetLocked(); err != nil {
			return err
		}
	}

	// Rotate if the active segment is non-empty and full.
	if len(w.segs) == 0 {
		if err := w.addSegmentLocked(l.Index); err != nil {
			return err
		}
	} else if act := w.segs[len(w.segs)-1]; act.size >= w.maxSeg {
		if err := w.addSegmentLocked(l.Index); err != nil {
			return err
		}
	}
	act := w.segs[len(w.segs)-1]

	w.encBuf = appendRecord(w.encBuf[:0], l)
	off := act.size
	if _, err := act.f.WriteAt(w.encBuf, off); err != nil {
		return fmt.Errorf("logstore: append index %d: %w", l.Index, err)
	}
	act.size += int64(len(w.encBuf))
	if len(w.offsets) == 0 {
		w.first = l.Index
	}
	w.offsets = append(w.offsets, recPos{segIdx: len(w.segs) - 1, off: off, size: len(w.encBuf)})
	return nil
}

func (w *WAL) addSegmentLocked(base uint64) error {
	name := fmt.Sprintf("%020d%s", base, segExt)
	path := filepath.Join(w.dir, name)
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("logstore: create segment %s: %w", name, err)
	}
	w.segs = append(w.segs, &segment{base: base, path: path, f: f})
	return nil
}

func (w *WAL) DeleteRange(minIdx, maxIdx uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.poisoned {
		return ErrWALPoisoned
	}
	if len(w.offsets) == 0 {
		return nil
	}
	last := w.lastLocked()
	if minIdx < w.first {
		minIdx = w.first
	}
	if maxIdx > last {
		maxIdx = last
	}
	if minIdx > maxIdx {
		return nil
	}
	switch {
	case minIdx == w.first && maxIdx == last:
		return w.resetLocked()
	case minIdx == w.first:
		return w.truncateFrontLocked(maxIdx + 1)
	case maxIdx == last:
		return w.truncateTailLocked(minIdx)
	default:
		// Middle deletion is not part of raft's log-store usage.
		return fmt.Errorf("logstore: unsupported middle DeleteRange(%d,%d)", minIdx, maxIdx)
	}
}

// truncateFrontLocked drops entries below newFirst (compaction after a snapshot)
// and deletes segments no longer referenced.
func (w *WAL) truncateFrontLocked(newFirst uint64) error {
	drop := int(newFirst - w.first)
	if drop >= len(w.offsets) {
		return w.resetLocked()
	}
	// Shift survivors to the front in place: reuses the backing array so
	// compaction allocates nothing (the array is reused as the log grows again).
	n := copy(w.offsets, w.offsets[drop:])
	w.offsets = w.offsets[:n]
	w.first = newFirst
	// Persist the compaction floor: a retained segment still physically holds the
	// dropped prefix records, and recovery must skip them rather than re-read them
	// as live. Written before segment deletion so a crash here still recovers
	// correctly (dead records below the floor are ignored).
	if err := w.writeFloorLocked(); err != nil {
		return err
	}
	// Every segment before the one holding the first surviving entry is dead.
	firstSeg := w.offsets[0].segIdx
	if firstSeg == 0 {
		return nil // nothing to drop; the boundary segment is still the first
	}
	for i := 0; i < firstSeg; i++ {
		_ = w.segs[i].f.Close()
		_ = os.Remove(w.segs[i].path)
	}
	w.segs = append(w.segs[:0], w.segs[firstSeg:]...)
	// Re-base every surviving offset's segIdx by the number of removed segments.
	for i := range w.offsets {
		w.offsets[i].segIdx -= firstSeg
	}
	if w.sync {
		return w.syncDir() // make the removals durable
	}
	return nil
}

// truncateTailLocked removes entries at newFirstRemoved and above (conflict
// resolution): truncate the owning segment to that record's offset and drop
// every later segment, so subsequent appends reuse the space.
func (w *WAL) truncateTailLocked(newFirstRemoved uint64) error {
	keepN := int(newFirstRemoved - w.first)
	pos := w.offsets[keepN]
	w.offsets = w.offsets[:keepN]

	// Drop every segment after the one holding the first removed entry.
	seg := w.segs[pos.segIdx]
	removed := pos.segIdx+1 < len(w.segs)
	for i := pos.segIdx + 1; i < len(w.segs); i++ {
		_ = w.segs[i].f.Close()
		_ = os.Remove(w.segs[i].path)
	}
	w.segs = w.segs[:pos.segIdx+1]
	if err := seg.f.Truncate(pos.off); err != nil {
		return fmt.Errorf("logstore: truncate tail: %w", err)
	}
	seg.size = pos.off
	// Make the file truncation and the segment removals durable, so a crash
	// cannot leave a removed higher-base segment on disk for recovery to
	// resurrect as live (its records would collide with the re-appended tail).
	if w.sync {
		if err := w.fsyncf(seg.f); err != nil {
			w.poisoned = true // fail closed: see ErrWALPoisoned
			return fmt.Errorf("logstore: fsync truncated segment: %w", err)
		}
		if removed {
			return w.syncDir()
		}
	}
	return nil
}

// resetLocked clears the whole log (all segments removed). Used by a full-log
// DeleteRange, a compaction that drops everything, and the monotonic-restore
// fallback.
func (w *WAL) resetLocked() error {
	removed := len(w.segs) > 0
	for _, s := range w.segs {
		_ = s.f.Close()
		_ = os.Remove(s.path)
	}
	w.segs = nil
	w.offsets = nil
	w.first = 0
	if err := os.Remove(w.floorPath()); err == nil { // no live entries => no floor
		removed = true
	}
	// Make the removals durable: without this a crash could leave the old segment
	// files (a valid, contiguous, monotonic-term log) on disk, which recovery
	// would rebuild wholesale — the term/base defenses cannot detect a deleted-
	// but-internally-consistent log.
	if removed && w.sync {
		return w.syncDir()
	}
	return nil
}

func (w *WAL) floorPath() string { return filepath.Join(w.dir, "first") }

// syncDir fsyncs the WAL directory. fsync of a file does NOT make its directory
// entry durable, so a create/rename/remove must be followed by a directory sync
// for the name change to survive a power loss.
func (w *WAL) syncDir() error {
	d, err := os.Open(w.dir)
	if err != nil {
		return fmt.Errorf("logstore: open dir: %w", err)
	}
	if err := w.fsyncf(d); err != nil {
		w.poisoned = true // fail closed: see ErrWALPoisoned
		_ = d.Close()
		return fmt.Errorf("logstore: sync dir: %w", err)
	}
	return d.Close()
}

// writeFileDurable writes data to path crash-safely: a temp file is written,
// fsynced, renamed into place, and the directory is fsynced. Used for the stable
// store and the compaction floor, which must be durable regardless of the log's
// sync mode.
func (w *WAL) writeFileDurable(path string, data []byte) error {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("logstore: create %s: %w", tmp, err)
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return fmt.Errorf("logstore: write %s: %w", tmp, err)
	}
	if err := w.fsyncf(f); err != nil {
		w.poisoned = true // fail closed: see ErrWALPoisoned
		_ = f.Close()
		return fmt.Errorf("logstore: fsync %s: %w", tmp, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("logstore: close %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("logstore: rename %s: %w", tmp, err)
	}
	return w.syncDir()
}

// writeFloorLocked persists w.first (the compaction floor) durably so recovery
// skips the dead prefix a retained segment still physically holds.
func (w *WAL) writeFloorLocked() error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], w.first)
	return w.writeFileDurable(w.floorPath(), b[:])
}

func (w *WAL) readFloor() uint64 {
	b, err := os.ReadFile(w.floorPath())
	if err != nil || len(b) < 8 {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

// recover scans existing segments, rebuilds the index, and truncates a torn tail.
func (w *WAL) recover() error {
	ents, err := os.ReadDir(w.dir)
	if err != nil {
		return fmt.Errorf("logstore: readdir: %w", err)
	}
	var bases []uint64
	for _, e := range ents {
		n := e.Name()
		if e.IsDir() || !strings.HasSuffix(n, segExt) {
			continue
		}
		b, err := strconv.ParseUint(strings.TrimSuffix(n, segExt), 10, 64)
		if err != nil {
			continue
		}
		bases = append(bases, b)
	}
	slices.Sort(bases)

	floor := w.readFloor() // compaction floor: records below this are dead prefix
	expected := uint64(0)  // next index we expect; 0 until the first live record
	lastTerm := uint64(0)  // terms are non-decreasing in a valid log; a drop = stale
	for _, base := range bases {
		path := filepath.Join(w.dir, fmt.Sprintf("%020d%s", base, segExt))
		f, err := os.OpenFile(path, os.O_RDWR, 0o600)
		if err != nil {
			return fmt.Errorf("logstore: open segment %s: %w", path, err)
		}
		seg := &segment{base: base, path: path, f: f}
		// Append before scanning so the segment's index in w.segs is stable and
		// scanSegment can record it in each recPos (len(w.segs)-1).
		w.segs = append(w.segs, seg)
		torn, err := w.scanSegment(seg, &expected, &lastTerm, floor)
		if err != nil {
			w.closeAllLocked() // close every fd opened so far before bailing
			return err
		}
		if torn {
			break // everything after a torn record is discarded
		}
	}
	return nil
}

// scanSegment reads records from seg, appending live entries (index >= floor) to
// the index and skipping the dead compacted prefix (index < floor). It stops at a
// torn or corrupt record, truncating the file there, and returns torn=true.
func (w *WAL) scanSegment(seg *segment, expected, lastTerm *uint64, floor uint64) (bool, error) {
	info, err := seg.f.Stat()
	if err != nil {
		return false, err
	}
	fileSize := info.Size()
	var off int64
	hdr := make([]byte, frameHdr)
	for off < fileSize {
		if off+frameHdr > fileSize {
			return w.stopScan(seg, off, fileSize, "short record header")
		}
		if _, err := seg.f.ReadAt(hdr, off); err != nil {
			return false, err
		}
		recLen := binary.LittleEndian.Uint32(hdr[:recLenSize])
		if recLen < crcSize || recLen > maxRecBytes {
			return w.stopScan(seg, off, fileSize, "invalid record length")
		}
		total := int64(recLenSize) + int64(recLen)
		if off+total > fileSize {
			return w.stopScan(seg, off, fileSize, "record body truncated")
		}
		buf := make([]byte, total)
		if _, err := seg.f.ReadAt(buf, off); err != nil {
			return false, err
		}
		payload := buf[frameHdr:]
		if binary.LittleEndian.Uint32(buf[recLenSize:frameHdr]) != crc32.ChecksumIEEE(payload) {
			return w.stopScan(seg, off, fileSize, "record crc mismatch")
		}
		idx := binary.LittleEndian.Uint64(payload)      // payload[0:8]  = Index
		term := binary.LittleEndian.Uint64(payload[8:]) // payload[8:16] = Term
		if idx < floor {
			off += total // dead compacted-prefix record; skip without indexing
			continue
		}
		if *expected != 0 && idx != *expected {
			return w.stopScan(seg, off, fileSize, "non-contiguous index")
		}
		// Terms never decrease with index in a valid raft log. A drop means this
		// record is a stale survivor of a tail-truncated segment that a crash left
		// on disk (its records predate the conflict) — reject it and everything
		// after rather than resurrect entries the cluster rewrote.
		if term < *lastTerm {
			return w.stopScan(seg, off, fileSize, "non-monotonic term (stale segment)")
		}
		if len(w.offsets) == 0 {
			w.first = idx
		}
		w.offsets = append(w.offsets, recPos{segIdx: len(w.segs) - 1, off: off, size: int(total)})
		*expected = idx + 1
		*lastTerm = term
		off += total
	}
	seg.size = fileSize
	return false, nil
}

// stopScan truncates a segment at a bad record during recovery and logs it
// loudly. A short/truncated record near EOF is the normal torn tail of a crash;
// a crc mismatch or a stop with many trailing bytes may instead be media
// corruption discarding committed entries — hence the warning includes how much
// is being dropped so an operator can tell the difference.
func (w *WAL) stopScan(seg *segment, off, fileSize int64, reason string) (bool, error) {
	slog.Warn("recovery stopped; discarding trailing byte(s) — if this was not a clean crash it may indicate corruption of committed entries",
		"component", "logstore", "segment", filepath.Base(seg.path), "offset", off, "reason", reason, "discarded_bytes", fileSize-off)
	return true, seg.truncateAt(off)
}

func (s *segment) truncateAt(off int64) error {
	if err := s.f.Truncate(off); err != nil {
		return fmt.Errorf("logstore: truncate torn tail: %w", err)
	}
	s.size = off
	return nil
}

// --- StableStore: a small map rewritten to one fsync'd file per Set ---

func (w *WAL) Set(key, val []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.poisoned {
		return ErrWALPoisoned
	}
	w.kv[string(key)] = append([]byte(nil), val...)
	return w.writeStableLocked()
}

func (w *WAL) Get(key []byte) ([]byte, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	// Poison gate on READS too, unlike the log path (whose GetLog fails loud via
	// the per-record CRC when the kernel dropped pages). The stable map is served
	// from MEMORY, and Set mutates the map before persisting — so after a failed
	// stable fsync the map can hold a term/vote that no reopen will reproduce.
	// Serving it would let raft act on a never-durable vote; fail closed instead.
	if w.poisoned {
		return nil, ErrWALPoisoned
	}
	v, ok := w.kv[string(key)]
	if !ok {
		return nil, nil
	}
	return append([]byte(nil), v...), nil
}

func (w *WAL) SetUint64(key []byte, val uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], val)
	return w.Set(key, b[:])
}

func (w *WAL) GetUint64(key []byte) (uint64, error) {
	v, err := w.Get(key)
	if err != nil || len(v) < 8 {
		return 0, err
	}
	return binary.LittleEndian.Uint64(v), nil
}

// writeStableLocked serializes the stable map and atomically replaces the file.
// Format: repeated [u32 keyLen][key][u32 valLen][val].
func (w *WAL) writeStableLocked() error {
	var buf []byte
	var tmp [4]byte
	for k, v := range w.kv {
		binary.LittleEndian.PutUint32(tmp[:], uint32(len(k))) //nolint:gosec // key lengths are tiny
		buf = append(buf, tmp[:]...)
		buf = append(buf, k...)
		binary.LittleEndian.PutUint32(tmp[:], uint32(len(v))) //nolint:gosec // value lengths are tiny
		buf = append(buf, tmp[:]...)
		buf = append(buf, v...)
	}
	// Append a CRC and write durably. The stable store holds raft's currentTerm
	// and votedFor: a granted vote MUST survive a crash or the node could vote
	// again in the same term and elect two leaders (split-brain). So this fsyncs
	// the file AND the directory unconditionally — independent of the log's sync
	// mode — and the CRC lets loadStable reject a torn/corrupt file rather than
	// load a partial vote.
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], crc32.ChecksumIEEE(buf))
	buf = append(buf, crc[:]...)
	return w.writeFileDurable(w.stablePath, buf)
}

func (w *WAL) loadStable() error {
	buf, err := os.ReadFile(w.stablePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("logstore: read stable: %w", err)
	}
	// Verify the trailing CRC: a corrupt stable file is a lost-vote hazard, so
	// refuse to start rather than load a partial/garbage term or vote. (With the
	// atomic durable write, a torn file only arises from actual media corruption.)
	if len(buf) < 4 {
		return fmt.Errorf("logstore: stable file too short: %w", errCorrupt)
	}
	payload := buf[:len(buf)-4]
	if binary.LittleEndian.Uint32(buf[len(buf)-4:]) != crc32.ChecksumIEEE(payload) {
		return fmt.Errorf("logstore: stable file crc mismatch: %w", errCorrupt)
	}
	p := payload
	for len(p) >= 4 {
		kl := binary.LittleEndian.Uint32(p)
		p = p[4:]
		if uint32(len(p)) < kl+4 {
			break
		}
		k := string(p[:kl])
		p = p[kl:]
		vl := binary.LittleEndian.Uint32(p)
		p = p[4:]
		if uint32(len(p)) < vl {
			break
		}
		w.kv[k] = append([]byte(nil), p[:vl]...)
		p = p[vl:]
	}
	return nil
}
