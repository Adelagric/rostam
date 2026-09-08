// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"
)

func TestRequestRoundtrip(t *testing.T) {
	op := "update_session"
	args := []byte(`{"user":42}`)
	frame := EncodeRequest(op, args)

	gotOp, gotArgs, err := DecodeRequest(frame)
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if gotOp != op {
		t.Fatalf("op = %q, want %q", gotOp, op)
	}
	if !bytes.Equal(gotArgs, args) {
		t.Fatalf("args mismatch")
	}
}

func TestRequestEmptyArgs(t *testing.T) {
	frame := EncodeRequest("__ping__", nil)
	op, args, err := DecodeRequest(frame)
	if err != nil {
		t.Fatal(err)
	}
	if op != "__ping__" || len(args) != 0 {
		t.Fatalf("op=%q args=%v", op, args)
	}
}

func TestResponseRoundtripOK(t *testing.T) {
	payload := []byte("hello")
	frame := EncodeResponse(StatusOK, payload)
	status, gotPayload, err := DecodeResponse(frame)
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	if status != StatusOK {
		t.Fatalf("status = %d, want OK", status)
	}
	if !bytes.Equal(gotPayload, payload) {
		t.Fatalf("payload mismatch")
	}
}

func TestResponseRoundtripNotLeader(t *testing.T) {
	payload := EncodeLeaderAddrPayload("10.0.0.5:7001")
	frame := EncodeResponse(StatusNotLeader, payload)
	status, gotPayload, err := DecodeResponse(frame)
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	if status != StatusNotLeader {
		t.Fatalf("status = %d, want NotLeader", status)
	}
	addr, derr := DecodeLeaderAddrPayload(gotPayload)
	if derr != nil {
		t.Fatalf("DecodeLeaderAddrPayload: %v", derr)
	}
	if addr != "10.0.0.5:7001" {
		t.Fatalf("addr = %q", addr)
	}
}

func TestRequestDecodeTruncated(t *testing.T) {
	frame := EncodeRequest("op", []byte("hi"))
	for n := 0; n < len(frame); n++ {
		if _, _, err := DecodeRequest(frame[:n]); err == nil {
			t.Fatalf("DecodeRequest at len=%d: nil err, want truncated", n)
		}
	}
}

func TestRequestOversizeRejected(t *testing.T) {
	// argsLen claims 1 GiB but buffer is small — must reject.
	frame := []byte{2, 'o', 'p', 0x40, 0, 0, 0} // opNameLen=2 op argsLen=2^30
	_, _, err := DecodeRequest(frame)
	if err == nil {
		t.Fatal("oversize argsLen accepted; want error")
	}
}

func TestEncodeRequestOpNameTooLongPanics(t *testing.T) {
	long := make([]byte, 256)
	for i := range long {
		long[i] = 'a'
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 256-byte opName")
		}
	}()
	EncodeRequest(string(long), nil)
}

func TestErrorTruncatedConstant(t *testing.T) {
	if !errors.Is(ErrFrameTruncated, ErrFrameTruncated) {
		t.Fatal("ErrFrameTruncated identity broken")
	}
}

// TestDecodeLengthOverflowNeverPanics pins the 32-bit regression: a declared
// length of 2^31 or more must never reach the int conversion in DecodeRequest
// or DecodeResponse unrejected. On GOARCH=386, int is 32-bit, so converting
// an unrejected uint32 that large wraps negative, which defeats every bounds
// check downstream and panics on the final slice expression. This must pass
// identically on amd64 and 386.
func TestDecodeLengthOverflowNeverPanics(t *testing.T) {
	lengths := []uint32{0xffffffff, 0x80000000, 0x7fffffff}

	for _, l := range lengths {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("DecodeRequest panicked for declared argsLen=%#x: %v", l, r)
				}
			}()
			frame := make([]byte, 1+2+4) // opNameLen=2 "op" argsLen
			frame[0] = 2
			frame[1] = 'o'
			frame[2] = 'p'
			binary.BigEndian.PutUint32(frame[3:7], l)
			_, _, err := DecodeRequest(frame)
			if err == nil {
				t.Fatalf("DecodeRequest accepted declared argsLen=%#x; want error", l)
			}
			if !errors.Is(err, ErrFrameTooLarge) && !errors.Is(err, ErrFrameTruncated) {
				t.Fatalf("DecodeRequest argsLen=%#x: err=%v, want ErrFrameTooLarge or ErrFrameTruncated", l, err)
			}
		}()

		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("DecodeResponse panicked for declared payloadLen=%#x: %v", l, r)
				}
			}()
			frame := make([]byte, 1+4) // status payloadLen
			frame[0] = StatusOK
			binary.BigEndian.PutUint32(frame[1:5], l)
			_, _, err := DecodeResponse(frame)
			if err == nil {
				t.Fatalf("DecodeResponse accepted declared payloadLen=%#x; want error", l)
			}
			if !errors.Is(err, ErrFrameTooLarge) && !errors.Is(err, ErrFrameTruncated) {
				t.Fatalf("DecodeResponse payloadLen=%#x: err=%v, want ErrFrameTooLarge or ErrFrameTruncated", l, err)
			}
		}()
	}

	// A declared length that is individually within MaxFrameSize can still
	// push the TOTAL reconstructed frame (header + declared length) over
	// MaxFrameSize — EncodeRequest/EncodeResponse panic on that total, so a
	// decoder that accepted such a frame could panic later on re-encode.
	// Neither case below needs a MaxFrameSize-sized buffer: the reject
	// happens purely from the header, before the body is ever touched.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("DecodeRequest panicked for a total-over-bound frame: %v", r)
			}
		}()
		nameLen := maxOpNameLen                   // 255: maximizes the header overhead
		argsLen := uint32(MaxFrameSize - nameLen) // valid alone (<= MaxFrameSize) ...
		header := make([]byte, 1+nameLen+4)       // ... but header+argsLen exceeds MaxFrameSize
		header[0] = byte(nameLen)
		binary.BigEndian.PutUint32(header[1+nameLen:1+nameLen+4], argsLen)
		_, _, err := DecodeRequest(header)
		if !errors.Is(err, ErrFrameTooLarge) {
			t.Fatalf("DecodeRequest total-over-bound: err=%v, want ErrFrameTooLarge", err)
		}
	}()

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("DecodeResponse panicked for a total-over-bound frame: %v", r)
			}
		}()
		frame := make([]byte, 1+4)
		frame[0] = StatusOK
		// payloadLen == MaxFrameSize is valid alone, but the 5-byte header
		// pushes the reconstructed total over MaxFrameSize.
		binary.BigEndian.PutUint32(frame[1:5], uint32(MaxFrameSize))
		_, _, err := DecodeResponse(frame)
		if !errors.Is(err, ErrFrameTooLarge) {
			t.Fatalf("DecodeResponse total-over-bound: err=%v, want ErrFrameTooLarge", err)
		}
	}()
}
