// SPDX-License-Identifier: Apache-2.0

// Package server hosts the Rostam TCP listener that dispatches frames
// into a Dispatcher.
package server

import (
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"
)

// Status codes carried in the response frame.
const (
	StatusOK           uint8 = 0
	StatusNotFound     uint8 = 1
	StatusNotLeader    uint8 = 2
	StatusError        uint8 = 3
	StatusUnauthorized uint8 = 4 // returned when auth is required and token is missing/invalid
)

// MaxFrameSize bounds each individual frame. Both encode and decode reject
// payloads above this size.
const MaxFrameSize = 16 << 20 // 16 MiB

const maxOpNameLen = 255

// ErrFrameTruncated indicates a frame buffer is shorter than its header claims.
var ErrFrameTruncated = errors.New("server: frame truncated")

// ErrFrameTooLarge indicates a frame exceeds MaxFrameSize.
var ErrFrameTooLarge = errors.New("server: frame too large")

// ProtocolV1 and ProtocolV2 are the version codes that may appear at byte 0
// of a request body. v1 is implicit — the byte is actually the opNameLen,
// guaranteed non-zero and !=2 by the registry's opName-length invariant.
// v2 explicitly writes 0x02 at byte 0 and adds a [tokenLen:u8][token]
// auth prefix before the v1-shaped body.
const (
	ProtocolV1 byte = 1 // implicit; byte 0 is opNameLen
	ProtocolV2 byte = 2 // explicit; byte 0 is the version, followed by tokenLen+token+v1-body
)

// EncodeRequestV2 writes a protocol-v2 request body with an auth token.
// Wire layout: [version:1=0x02][tokenLen:1][token][opNameLen:1][opName][argsLen:4][args].
func EncodeRequestV2(token, opName string, args []byte) []byte {
	if len(token) > 255 {
		panic(fmt.Sprintf("server: token length %d exceeds 255", len(token)))
	}
	if len(opName) > maxOpNameLen {
		panic(fmt.Sprintf("server: opName length %d exceeds %d", len(opName), maxOpNameLen))
	}
	total := 1 + 1 + len(token) + 1 + len(opName) + 4 + len(args)
	if total > MaxFrameSize {
		panic("server: encoded v2 request exceeds MaxFrameSize")
	}
	out := make([]byte, total)
	out[0] = ProtocolV2
	out[1] = byte(len(token)) //nolint:gosec
	off := 2
	copy(out[off:off+len(token)], token)
	off += len(token)
	out[off] = byte(len(opName)) //nolint:gosec
	off++
	copy(out[off:off+len(opName)], opName)
	off += len(opName)
	binary.BigEndian.PutUint32(out[off:off+4], uint32(len(args))) //nolint:gosec
	off += 4
	copy(out[off:], args)
	return out
}

// DecodeRequestV2 reads a protocol-v2 request body produced by EncodeRequestV2.
// Returns the bearer token + the v1-shaped suffix; callers pass the suffix
// to DecodeRequest. Returns ErrFrameTruncated if the body is short.
func DecodeRequestV2(frame []byte) (token string, v1Body []byte, err error) {
	if len(frame) < 2 {
		return "", nil, ErrFrameTruncated
	}
	if frame[0] != ProtocolV2 {
		return "", nil, fmt.Errorf("server: not a v2 frame (byte0=%d)", frame[0])
	}
	tokenLen := int(frame[1])
	if len(frame) < 2+tokenLen {
		return "", nil, ErrFrameTruncated
	}
	if tokenLen > 0 {
		token = unsafe.String(&frame[2], tokenLen)
	}
	return token, frame[2+tokenLen:], nil
}

// EncodeRequest writes a request frame body (excluding the leading 4-byte length).
// Wire layout: [opNameLen u8][opName][argsLen u32][args].
// Panics if opName exceeds 255 bytes — names are static; callers validate upstream.
func EncodeRequest(opName string, args []byte) []byte {
	if len(opName) > maxOpNameLen {
		panic(fmt.Sprintf("server: opName length %d exceeds %d", len(opName), maxOpNameLen))
	}
	if 1+len(opName)+4+len(args) > MaxFrameSize {
		panic("server: encoded request exceeds MaxFrameSize")
	}
	out := make([]byte, 1+len(opName)+4+len(args))
	out[0] = byte(len(opName)) //nolint:gosec // bounded by maxOpNameLen check above
	copy(out[1:1+len(opName)], opName)
	binary.BigEndian.PutUint32(out[1+len(opName):1+len(opName)+4], uint32(len(args))) //nolint:gosec // bounded by MaxFrameSize above
	copy(out[1+len(opName)+4:], args)
	return out
}

// DecodeRequest reads a request body produced by EncodeRequest. The returned
// opName aliases the frame bytes via unsafe.String to avoid the per-request
// heap allocation of a fresh string header — the dispatcher uses the name
// only for a synchronous registry.Lookup before the frame is returned to
// the pool, so the aliased bytes are guaranteed to outlive the string.
// Callers MUST NOT retain opName beyond the frame's lifetime.
func DecodeRequest(frame []byte) (opName string, args []byte, err error) {
	if len(frame) < 1 {
		return "", nil, ErrFrameTruncated
	}
	nameLen := int(frame[0])
	if len(frame) < 1+nameLen+4 {
		return "", nil, ErrFrameTruncated
	}
	if nameLen > 0 {
		opName = unsafe.String(&frame[1], nameLen)
	}
	rawArgsLen := binary.BigEndian.Uint32(frame[1+nameLen : 1+nameLen+4])
	// Reject on the TOTAL reconstructed frame size (header + args), computed
	// in uint64 so nothing can overflow on any platform, matching the bound
	// EncodeRequest panics on. Two things this guards against:
	//   - the int conversion below: on 32-bit platforms (GOARCH=386/arm),
	//     int is 32-bit and a raw length >= 2^31 would wrap negative,
	//     defeating every bounds check that follows.
	//   - a per-field-valid argsLen (<= MaxFrameSize) whose header+args
	//     total still exceeds MaxFrameSize; EncodeRequest panics on that
	//     total, so a decoder that accepted such a frame could panic later
	//     on re-encode.
	headerLen := uint64(1 + nameLen + 4) // <= 1+255+4, never overflows
	if headerLen+uint64(rawArgsLen) > uint64(MaxFrameSize) {
		return "", nil, ErrFrameTooLarge
	}
	argsLen := int(rawArgsLen)
	if len(frame) < 1+nameLen+4+argsLen {
		return "", nil, ErrFrameTruncated
	}
	args = frame[1+nameLen+4 : 1+nameLen+4+argsLen]
	return opName, args, nil
}

// EncodeResponse writes a response body. Wire layout: [status u8][payloadLen u32][payload].
func EncodeResponse(status uint8, payload []byte) []byte {
	if 1+4+len(payload) > MaxFrameSize {
		panic("server: encoded response exceeds MaxFrameSize")
	}
	out := make([]byte, 1+4+len(payload))
	out[0] = status
	binary.BigEndian.PutUint32(out[1:5], uint32(len(payload))) //nolint:gosec // bounded by MaxFrameSize above
	copy(out[5:], payload)
	return out
}

// DecodeResponse reads a response body produced by EncodeResponse.
func DecodeResponse(frame []byte) (status uint8, payload []byte, err error) {
	if len(frame) < 1+4 {
		return 0, nil, ErrFrameTruncated
	}
	status = frame[0]
	rawPayloadLen := binary.BigEndian.Uint32(frame[1:5])
	// Reject on the TOTAL reconstructed frame size (header + payload),
	// computed in uint64 so nothing can overflow on any platform, matching
	// the bound EncodeResponse panics on. See DecodeRequest for why both the
	// int-conversion and total-size cases matter.
	const headerLen = uint64(1 + 4)
	if headerLen+uint64(rawPayloadLen) > uint64(MaxFrameSize) {
		return 0, nil, ErrFrameTooLarge
	}
	payloadLen := int(rawPayloadLen)
	if len(frame) < 5+payloadLen {
		return 0, nil, ErrFrameTruncated
	}
	payload = frame[5 : 5+payloadLen]
	return status, payload, nil
}

// EncodeLeaderAddrPayload packs a "leader address" string for the
// StatusNotLeader payload. Wire: [addrLen u16][addr ASCII].
func EncodeLeaderAddrPayload(addr string) []byte {
	if len(addr) > 0xFFFF {
		panic("server: leader addr exceeds uint16 length")
	}
	out := make([]byte, 2+len(addr))
	binary.BigEndian.PutUint16(out[0:2], uint16(len(addr))) //nolint:gosec // bounded above
	copy(out[2:], addr)
	return out
}

// DecodeLeaderAddrPayload reads a leader-addr payload.
func DecodeLeaderAddrPayload(payload []byte) (string, error) {
	if len(payload) < 2 {
		return "", ErrFrameTruncated
	}
	n := int(binary.BigEndian.Uint16(payload[0:2]))
	if len(payload) < 2+n {
		return "", ErrFrameTruncated
	}
	return string(payload[2 : 2+n]), nil
}

// EncodeErrorPayload packs an error message. Wire: [msgLen u16][msg ASCII].
func EncodeErrorPayload(msg string) []byte {
	if len(msg) > 0xFFFF {
		msg = msg[:0xFFFF]
	}
	out := make([]byte, 2+len(msg))
	binary.BigEndian.PutUint16(out[0:2], uint16(len(msg))) //nolint:gosec // bounded above
	copy(out[2:], msg)
	return out
}

// DecodeErrorPayload reads an error message payload.
func DecodeErrorPayload(payload []byte) (string, error) {
	if len(payload) < 2 {
		return "", ErrFrameTruncated
	}
	n := int(binary.BigEndian.Uint16(payload[0:2]))
	if len(payload) < 2+n {
		return "", ErrFrameTruncated
	}
	return string(payload[2 : 2+n]), nil
}
