// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// APIKey is one entry in the auth registry. Token is the opaque bearer string
// the client sends on every RPC; Tenant scopes the key to a tenant; Permissions
// is the legacy coarse permission set (kept for backward-compat); Scopes is the
// granular per-collection RBAC scope list ("<action>:<pattern>", e.g.
// "read:default/docs", "write:default/*", "*:*"); CertCN is the mTLS client-cert
// CommonName this key is bound to (used by LookupByCN for cert→principal
// resolution; empty means the key is not cert-bound).
type APIKey struct {
	Token       string   `json:"token"`
	Tenant      string   `json:"tenant"`
	Permissions []string `json:"permissions"` // any subset of {"read", "write", "admin"}
	Scopes      []string `json:"scopes,omitempty"`
	CertCN      string   `json:"cert_cn,omitempty"`
}

// Permission constants for APIKey.Permissions.
const (
	PermRead  = "read"
	PermWrite = "write"
	PermAdmin = "admin"
)

// Has reports whether the key carries the given permission. Admin implies
// all other permissions.
func (k APIKey) Has(perm string) bool {
	for _, p := range k.Permissions {
		if p == perm || p == PermAdmin {
			return true
		}
	}
	return false
}

// ErrAPIKeyExists is returned by AddKey when a token is already registered.
var ErrAPIKeyExists = errors.New("vector: api key already exists")

// ErrAPIKeyNotFound is returned by RevokeKey when the token is not registered.
var ErrAPIKeyNotFound = errors.New("vector: api key not found")

// ErrInvalidPermission is returned by AddKey when a permission is unknown.
var ErrInvalidPermission = errors.New("vector: invalid permission")

// validPermissions is the closed set of permissions an APIKey may carry.
var validPermissions = map[string]struct{}{
	PermRead: {}, PermWrite: {}, PermAdmin: {},
}

// KeyRegistry is the in-memory map of API tokens to APIKey records. It
// persists to a single JSON file on disk via atomic tmp+rename writes.
//
// KeyRegistry is safe for concurrent use — readers (Lookup) take an RWMutex
// read lock; mutators (AddKey, RevokeKey) take the write lock + rewrite the
// JSON file synchronously.
type KeyRegistry struct {
	path string

	mu   sync.RWMutex
	keys map[string]APIKey // token -> entry
	byCN map[string]APIKey // CertCN -> entry (empty CN never indexed)
}

// scopesForPermissions synthesizes the equivalent granular RBAC scopes for a
// legacy Permissions-only key, so old key files authorize identically under the
// scope engine. Mapping: read -> "read:*"; write -> "read:*","write:*"; admin
// (alone or alongside any other perm) -> "*:*". Returns nil for an empty
// permission set (deny-by-default: a key with no permissions grants nothing).
func scopesForPermissions(perms []string) []string {
	hasRead, hasWrite, hasAdmin := false, false, false
	for _, p := range perms {
		switch p {
		case PermRead:
			hasRead = true
		case PermWrite:
			hasWrite = true
		case PermAdmin:
			hasAdmin = true
		}
	}
	if hasAdmin {
		return []string{"*:*"}
	}
	var out []string
	// write implies read at the key level (a writer is conventionally also a
	// reader), matching the documented Permissions->scopes mapping.
	if hasRead || hasWrite {
		out = append(out, "read:*")
	}
	if hasWrite {
		out = append(out, "write:*")
	}
	return out
}

// indexLocked (re)builds the CertCN index from r.keys. Caller holds the write
// lock. An entry with an empty CertCN is never indexed (an empty CN must never
// match in LookupByCN). If two entries share a CertCN the last one wins; this is
// a misconfiguration the operator owns.
func (r *KeyRegistry) indexLocked() {
	r.byCN = make(map[string]APIKey, len(r.keys))
	for _, k := range r.keys {
		if k.CertCN != "" {
			r.byCN[k.CertCN] = k
		}
	}
}

// OpenKeyRegistry loads the registry from the JSON file at path, or returns
// an empty registry if the file does not exist. Creates the parent directory
// if needed.
func OpenKeyRegistry(path string) (*KeyRegistry, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return nil, fmt.Errorf("vector: mkdir auth dir: %w", err)
	}
	r := &KeyRegistry{path: path, keys: make(map[string]APIKey), byCN: make(map[string]APIKey)}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return r, nil
		}
		return nil, fmt.Errorf("vector: read %s: %w", path, err)
	}
	if len(data) == 0 {
		return r, nil
	}
	var entries []APIKey
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("vector: parse %s: %w", path, err)
	}
	for _, k := range entries {
		// Backward-compat: a legacy key file carries Permissions but no Scopes;
		// synthesize equivalent scopes so it authorizes identically under the
		// scope engine. A key that already declares Scopes is taken verbatim.
		if len(k.Scopes) == 0 && len(k.Permissions) > 0 {
			k.Scopes = scopesForPermissions(k.Permissions)
		}
		r.keys[k.Token] = k
	}
	r.indexLocked()
	return r, nil
}

// AddKey registers a new API key. Returns ErrAPIKeyExists if the token is
// already in use, ErrInvalidPermission if any permission is unknown.
func (r *KeyRegistry) AddKey(k APIKey) error {
	if k.Token == "" || k.Tenant == "" {
		return errors.New("vector: AddKey requires non-empty token and tenant")
	}
	for _, p := range k.Permissions {
		if _, ok := validPermissions[p]; !ok {
			return fmt.Errorf("%w: %q", ErrInvalidPermission, p)
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.keys[k.Token]; exists {
		return ErrAPIKeyExists
	}
	// Mirror the on-load synthesis: a key added with only legacy Permissions and
	// no explicit Scopes gets equivalent scopes so it authorizes under the scope
	// engine (without this, a runtime-added Permissions-only key would silently
	// match no scope and be denied everything — a fail-closed foot-gun).
	if len(k.Scopes) == 0 && len(k.Permissions) > 0 {
		k.Scopes = scopesForPermissions(k.Permissions)
	}
	r.keys[k.Token] = k
	r.indexLocked()
	return r.flushLocked()
}

// RevokeKey removes the key with the given token. Returns ErrAPIKeyNotFound
// if the token is not registered.
func (r *KeyRegistry) RevokeKey(token string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.keys[token]; !ok {
		return ErrAPIKeyNotFound
	}
	delete(r.keys, token)
	r.indexLocked()
	return r.flushLocked()
}

// Lookup returns the APIKey for the given token, or (zero, false) if not
// registered.
func (r *KeyRegistry) Lookup(token string) (APIKey, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	k, ok := r.keys[token]
	return k, ok
}

// LookupByCN returns the APIKey bound to the given verified mTLS client-cert
// CommonName, or (zero, false) if no entry is bound to that CN. An empty cn
// never matches (a cert with no/empty CN must not resolve to a principal).
func (r *KeyRegistry) LookupByCN(cn string) (APIKey, bool) {
	if cn == "" {
		return APIKey{}, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	k, ok := r.byCN[cn]
	return k, ok
}

// ListKeys returns a snapshot of all registered keys. The slice is freshly
// allocated; the caller may modify it freely without affecting the registry.
func (r *KeyRegistry) ListKeys() []APIKey {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]APIKey, 0, len(r.keys))
	for _, k := range r.keys {
		out = append(out, k)
	}
	return out
}

// TokenFingerprint returns a short, NON-reversible identifier for a token: the
// first 16 hex chars (8 bytes) of sha256(token). An empty token returns "".
//
// SECURITY: this is the ONLY way a token may be surfaced outside the registry.
// The raw token is a secret and must NEVER be serialized into a list result, an
// audit record, a log line, or any wire frame. sha256 is one-way; the truncation
// to 16 hex chars keeps the value compact while remaining collision-resistant
// enough for correlation. The audit layer (authz) delegates to this single
// definition so there is exactly one fingerprint scheme across the codebase.
func TokenFingerprint(token string) string {
	if token == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:8]) // 8 bytes -> 16 hex chars
}

// RedactedKey is the SECURE, token-free view of an APIKey for the online
// key-admin list op. It carries the token's non-reversible Fingerprint
// (TokenFingerprint) in place of the raw Token, plus the non-secret descriptive
// fields. It deliberately has NO Token field, so no codec/transport can ever
// serialize the secret — redaction is enforced at this type boundary, not just
// at the HTTP/gRPC edge.
type RedactedKey struct {
	Fingerprint string   `json:"fingerprint"`
	Tenant      string   `json:"tenant"`
	Scopes      []string `json:"scopes,omitempty"`
	CertCN      string   `json:"cert_cn,omitempty"`
}

// ListRedacted returns a token-free snapshot of every registered key: each
// entry's raw token is replaced by its TokenFingerprint. The returned slice is
// freshly allocated. This is the ONLY list method safe to surface over a
// transport — ListKeys returns raw tokens and must stay internal.
func (r *KeyRegistry) ListRedacted() []RedactedKey {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]RedactedKey, 0, len(r.keys))
	for _, k := range r.keys {
		out = append(out, RedactedKey{
			Fingerprint: TokenFingerprint(k.Token),
			Tenant:      k.Tenant,
			Scopes:      append([]string(nil), k.Scopes...),
			CertCN:      k.CertCN,
		})
	}
	return out
}

// flushLocked writes the current keys to disk atomically. Caller must hold
// the write lock.
func (r *KeyRegistry) flushLocked() error {
	entries := make([]APIKey, 0, len(r.keys))
	for _, k := range r.keys {
		entries = append(entries, k)
	}
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	return atomicWriteFile(r.path, data)
}
