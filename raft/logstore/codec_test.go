// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"errors"
	"testing"

	hraft "github.com/hashicorp/raft"
)

// TestDecodeIntoRejectsTrailingBytes pins that decodeInto rejects a payload with
// bytes past the last field rather than silently dropping them. A record is
// sealed by its recLen frame + crc, so a well-formed payload ends exactly at the
// extensions field; trailing bytes mean a misframed/corrupt record, and the
// decoder must fail closed like the other wire decoders in the tree.
func TestDecodeIntoRejectsTrailingBytes(t *testing.T) {
	enc := appendRecord(nil, &hraft.Log{Index: 7, Term: 3, Type: hraft.LogCommand, Data: []byte("hi"), Extensions: []byte("x")})
	payload := enc[frameHdr:] // strip recLen+crc; decodeInto takes the payload only

	var out hraft.Log
	if err := decodeInto(payload, &out); err != nil {
		t.Fatalf("valid payload must decode: %v", err)
	}

	// One junk byte past the extensions field.
	if err := decodeInto(append(append([]byte(nil), payload...), 0x00), &out); !errors.Is(err, errCorrupt) {
		t.Fatalf("trailing byte: want errCorrupt, got %v", err)
	}
}
