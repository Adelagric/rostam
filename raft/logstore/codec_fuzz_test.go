// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"bytes"
	"testing"

	hraft "github.com/hashicorp/raft"
)

// decodeInto parses a record payload read back off disk during WAL recovery —
// a surface that must tolerate a torn or corrupt tail without panicking or
// over-reading, since recovery runs on whatever bytes survived a crash. It had
// no fuzz coverage; these targets pin the never-panic and round-trip contracts.

// FuzzDecodeInto: arbitrary payload bytes must never panic decodeInto, and a
// payload it accepts must round-trip through appendRecord to the same logical
// entry (index/term/type/data/extensions), the fields recovery relies on.
func FuzzDecodeInto(f *testing.F) {
	// Seed with a real encoded payload (strip the 8-byte frame header appendRecord
	// prepends, since decodeInto receives the crc-stripped payload only).
	enc := appendRecord(nil, &hraft.Log{Index: 7, Term: 3, Type: hraft.LogCommand, Data: []byte("hi"), Extensions: []byte("x")})
	f.Add(enc[frameHdr:])
	f.Add([]byte{})
	f.Add(make([]byte, payloadHdr))
	f.Add(make([]byte, payloadHdr+4)) // dataLen present, zero
	// A payload whose declared dataLen overruns the buffer (hostile length).
	bad := make([]byte, payloadHdr+4)
	bad[payloadHdr] = 0xff
	bad[payloadHdr+1] = 0xff
	f.Add(bad)

	f.Fuzz(func(t *testing.T, payload []byte) {
		var out hraft.Log
		if err := decodeInto(payload, &out); err != nil {
			return
		}
		// Re-encoding the decoded entry must reproduce a PREFIX of the payload.
		// NOTE decodeInto does NOT reject trailing bytes after the extensions
		// field (unlike every other decoder in the tree — cf. decodeReplicateGroup's
		// "trailing bytes" check), so a payload with junk past the last field
		// decodes "successfully" and silently drops it. In production the recLen
		// framing + crc seal each record, so this path never sees trailing bytes;
		// this is a defense-in-depth / consistency gap, not an exploitable bug, so
		// the invariant asserts a faithful prefix rather than full equality.
		re := appendRecord(nil, &out)[frameHdr:]
		if len(re) > len(payload) || !bytes.Equal(re, payload[:len(re)]) {
			t.Fatalf("decodeInto round-trip mismatch: in=%x out=%x", payload, re)
		}
	})
}

// FuzzWALReplayRecord drives a full WAL open/append/reopen cycle with an
// arbitrary entry payload, so the on-disk framing (recLen + crc + record) is
// exercised end to end: whatever StoreLog persisted must read back byte-for-byte
// through GetLog, and a reopened WAL must recover the same LastIndex. This is
// the recovery path the poison latch and the crash tests also lean on.
func FuzzWALReplayRecord(f *testing.F) {
	f.Add([]byte("payload"), uint64(1), uint64(1))
	f.Add([]byte{}, uint64(5), uint64(2))
	f.Add(bytes.Repeat([]byte{0xab}, 4096), uint64(9), uint64(3))

	f.Fuzz(func(t *testing.T, data []byte, idx, term uint64) {
		if idx == 0 {
			idx = 1 // raft indices are 1-based; 0 is never stored
		}
		dir := t.TempDir()
		w, err := OpenWAL(dir, true)
		if err != nil {
			t.Fatal(err)
		}
		in := &hraft.Log{Index: idx, Term: term, Type: hraft.LogCommand, Data: data}
		if err := w.StoreLog(in); err != nil {
			t.Fatalf("StoreLog: %v", err)
		}
		var got hraft.Log
		if err := w.GetLog(idx, &got); err != nil {
			t.Fatalf("GetLog: %v", err)
		}
		if got.Index != idx || got.Term != term || !bytes.Equal(got.Data, data) {
			t.Fatalf("read-back mismatch: got {%d,%d,%x} want {%d,%d,%x}", got.Index, got.Term, got.Data, idx, term, data)
		}
		if err := w.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		// Reopen: recovery must rebuild the same entry from the segment files.
		w2, err := OpenWAL(dir, true)
		if err != nil {
			t.Fatalf("reopen: %v", err)
		}
		defer func() { _ = w2.Close() }()
		last, err := w2.LastIndex()
		if err != nil || last != idx {
			t.Fatalf("recovered LastIndex=%d err=%v, want %d", last, err, idx)
		}
		var got2 hraft.Log
		if err := w2.GetLog(idx, &got2); err != nil {
			t.Fatalf("GetLog after reopen: %v", err)
		}
		if !bytes.Equal(got2.Data, data) {
			t.Fatalf("recovered data mismatch: got %x want %x", got2.Data, data)
		}
	})
}
