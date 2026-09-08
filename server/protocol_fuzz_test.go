// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"testing"
)

// The client-facing wire decoders parse untrusted bytes straight off the TCP
// listener, so their whole contract is: never panic, never over-read, and
// round-trip anything the encoders produced. These fuzz targets pin that. They
// mirror the coverage raft/fabric already has for its peer decoders — the
// server protocol and the pbisr peer frame (see pb_frame_fuzz_test.go) were the
// untrusted surfaces that had none.

// FuzzDecodeRequest: DecodeRequest must not panic on arbitrary input, and any
// frame it accepts must re-encode to a prefix of itself (trailing bytes past
// the declared args are tolerated by design, so equality would be wrong).
func FuzzDecodeRequest(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{1, 'x', 0, 0, 0, 0})
	f.Add(EncodeRequest("get", []byte("hello")))
	f.Add(EncodeRequest("", nil))
	f.Add([]byte{0xff}) // nameLen=255, truncated
	f.Add([]byte{2, 'o', 'p', 0xff, 0xff, 0xff, 0xff}) // huge argsLen

	f.Fuzz(func(t *testing.T, frame []byte) {
		name, args, err := DecodeRequest(frame)
		if err != nil {
			return
		}
		// Accepted: opName+args must be a faithful prefix of the input.
		re := EncodeRequest(name, args)
		if len(re) > len(frame) || !bytes.Equal(re, frame[:len(re)]) {
			t.Fatalf("decode/encode mismatch: frame=%x reencoded=%x", frame, re)
		}
	})
}

// FuzzDecodeRequestV2: the v2 auth-prefixed decoder must not panic; whatever
// v1 suffix it returns must itself be safe to hand to DecodeRequest (that is
// exactly what the dispatcher does), so chain them.
func FuzzDecodeRequestV2(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{2})
	f.Add([]byte{2, 3, 'a', 'b', 'c'})
	f.Add(EncodeRequestV2("tok", "get", []byte("v")))
	f.Add(EncodeRequestV2("", "", nil))
	f.Add([]byte{2, 0xff}) // tokenLen=255, truncated

	f.Fuzz(func(t *testing.T, frame []byte) {
		tok, v1, err := DecodeRequestV2(frame)
		if err != nil {
			return
		}
		// The token must be a faithful slice of the frame at its fixed offset.
		if len(tok) > 0 && !bytes.Equal([]byte(tok), frame[2:2+len(tok)]) {
			t.Fatalf("token not a faithful slice: frame=%x tok=%q", frame, tok)
		}
		// The suffix is fed straight to DecodeRequest downstream — must not panic.
		_, _, _ = DecodeRequest(v1)
	})
}

// FuzzDecodeResponse: response decoder on untrusted bytes (a client reading a
// server reply, or a peer proxy) must not panic and must round-trip as a prefix.
func FuzzDecodeResponse(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0})
	f.Add(EncodeResponse(StatusOK, []byte("ok")))
	f.Add([]byte{3, 0xff, 0xff, 0xff, 0xff}) // huge payloadLen

	f.Fuzz(func(t *testing.T, frame []byte) {
		status, payload, err := DecodeResponse(frame)
		if err != nil {
			return
		}
		re := EncodeResponse(status, payload)
		if len(re) > len(frame) || !bytes.Equal(re, frame[:len(re)]) {
			t.Fatalf("decode/encode mismatch: frame=%x reencoded=%x", frame, re)
		}
	})
}

// FuzzDecodeLeaderAddrPayload and FuzzDecodeErrorPayload: the two u16-prefixed
// string payloads a client parses out of a redirect/error response.
func FuzzDecodeLeaderAddrPayload(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 5, 'a'})
	f.Add(EncodeLeaderAddrPayload("10.0.0.1:7000"))
	f.Fuzz(func(t *testing.T, payload []byte) {
		addr, err := DecodeLeaderAddrPayload(payload)
		if err != nil {
			return
		}
		re := EncodeLeaderAddrPayload(addr)
		if !bytes.Equal(re, payload[:len(re)]) {
			t.Fatalf("leader-addr mismatch: payload=%x reencoded=%x", payload, re)
		}
	})
}

func FuzzDecodeErrorPayload(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0, 9, 'x'})
	f.Add(EncodeErrorPayload("boom"))
	f.Fuzz(func(t *testing.T, payload []byte) {
		msg, err := DecodeErrorPayload(payload)
		if err != nil {
			return
		}
		re := EncodeErrorPayload(msg)
		if !bytes.Equal(re, payload[:len(re)]) {
			t.Fatalf("error-payload mismatch: payload=%x reencoded=%x", payload, re)
		}
	})
}
