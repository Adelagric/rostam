// SPDX-License-Identifier: Apache-2.0

// Package logstore is a purpose-built raft LogStore + StableStore, replacing the
// raft-boltdb (bbolt) backend for both the durable and in-memory cases.
//
// A raft log is an append-mostly, integer-indexed sequence with front-truncation
// (after a snapshot) and rare tail-truncation (conflict). bbolt is a
// general-purpose disk B+tree (mmap, MVCC transactions, page freelist) and pays
// a B+tree commit plus a msgpack encode per entry — together the dominant cost
// and allocation in the cluster write path. This package instead uses:
//
//   - Mem:  a contiguous in-memory slice (NoSync — durability from replication).
//   - WAL:  a segmented append-only file log with an in-memory offset index and a
//     fixed binary record format (durable; fsync per batch).
//
// Both declare MonotonicLogStore, so the index is always gap-free and the log is
// addressable as base+offset.
package logstore

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"time"

	hraft "github.com/hashicorp/raft"
)

// errCorrupt marks a record that failed framing or CRC validation. On WAL
// recovery it marks the torn tail: everything from that point is discarded.
var errCorrupt = errors.New("logstore: corrupt record")

// Record wire format (little-endian), written by appendRecord:
//
//	u32 recLen        // = 4 (crc) + len(payload); frames the record for scanning
//	u32 crc           // crc32(payload)
//	payload:
//	  u64 index
//	  u64 term
//	  u8  type
//	  i64 appendedAtUnixNano
//	  u32 dataLen ; data
//	  u32 extLen  ; ext
//
// recLen up front lets recovery skip whole records and detect a short/torn tail;
// crc catches a corrupt or half-written record.
const (
	recLenSize  = 4
	crcSize     = 4
	frameHdr    = recLenSize + crcSize // bytes before the payload
	payloadHdr  = 8 + 8 + 1 + 8        // index+term+type+appendedAt, before data/ext
	maxRecBytes = 256 << 20            // sanity bound so a corrupt recLen can't allocate wildly
)

// appendRecord encodes l onto buf (reused across calls for zero-alloc on the hot
// path) and returns the grown slice. It backfills recLen and crc after writing
// the payload.
func appendRecord(buf []byte, l *hraft.Log) []byte {
	start := len(buf)
	buf = append(buf, 0, 0, 0, 0, 0, 0, 0, 0) // recLen + crc placeholders
	var tmp [8]byte
	binary.LittleEndian.PutUint64(tmp[:], l.Index)
	buf = append(buf, tmp[:]...)
	binary.LittleEndian.PutUint64(tmp[:], l.Term)
	buf = append(buf, tmp[:]...)
	buf = append(buf, byte(l.Type))
	binary.LittleEndian.PutUint64(tmp[:], uint64(l.AppendedAt.UnixNano())) //nolint:gosec // time round-trips through int64
	buf = append(buf, tmp[:]...)
	binary.LittleEndian.PutUint32(tmp[:4], uint32(len(l.Data))) //nolint:gosec // entry sizes are bounded by MaxFrameSize upstream
	buf = append(buf, tmp[:4]...)
	buf = append(buf, l.Data...)
	binary.LittleEndian.PutUint32(tmp[:4], uint32(len(l.Extensions))) //nolint:gosec // bounded upstream
	buf = append(buf, tmp[:4]...)
	buf = append(buf, l.Extensions...)

	payload := buf[start+frameHdr:]
	recLen := uint32(crcSize + len(payload)) //nolint:gosec // payload length bounded by MaxFrameSize
	binary.LittleEndian.PutUint32(buf[start:], recLen)
	binary.LittleEndian.PutUint32(buf[start+recLenSize:], crc32.ChecksumIEEE(payload))
	return buf
}

// decodeInto validates crc and fills out from payload (crc already stripped).
// Data/Extensions are copied into out's existing capacity, so a reused *Log
// causes no allocation once warmed.
func decodeInto(payload []byte, out *hraft.Log) error {
	if len(payload) < payloadHdr {
		return errCorrupt
	}
	p := payload
	out.Index = binary.LittleEndian.Uint64(p)
	out.Term = binary.LittleEndian.Uint64(p[8:])
	out.Type = hraft.LogType(p[16])
	out.AppendedAt = time.Unix(0, int64(binary.LittleEndian.Uint64(p[17:]))) //nolint:gosec // round-trips
	p = p[payloadHdr:]

	if len(p) < 4 {
		return errCorrupt
	}
	dl := binary.LittleEndian.Uint32(p)
	p = p[4:]
	if uint32(len(p)) < dl {
		return errCorrupt
	}
	out.Data = append(out.Data[:0], p[:dl]...)
	p = p[dl:]

	if len(p) < 4 {
		return errCorrupt
	}
	el := binary.LittleEndian.Uint32(p)
	p = p[4:]
	if uint32(len(p)) < el {
		return errCorrupt
	}
	if el == 0 {
		out.Extensions = nil
	} else {
		out.Extensions = append(out.Extensions[:0], p[:el]...)
	}
	p = p[el:]
	// A record is sealed by its recLen frame + crc, so a well-formed payload ends
	// exactly here. Any remaining bytes mean a misframed or corrupt record; reject
	// it rather than silently dropping the tail — matching the trailing-bytes
	// checks the wire decoders elsewhere already make (e.g. decodeReplicateGroup).
	if len(p) != 0 {
		return errCorrupt
	}
	return nil
}

// cloneLog deep-copies l into store-owned memory. raft reuses the input buffers
// after StoreLogs returns, so the payload must be copied; this is the one
// unavoidable allocation per stored entry.
func cloneLog(l *hraft.Log) hraft.Log {
	c := *l
	if l.Data != nil {
		c.Data = append([]byte(nil), l.Data...)
	}
	if l.Extensions != nil {
		c.Extensions = append([]byte(nil), l.Extensions...)
	}
	return c
}
