// SPDX-License-Identifier: Apache-2.0
//go:build windows

package vector

// dirSyncSupported is false because Windows exposes no directory fsync: a
// directory handle opened through os.Open cannot be flushed (FlushFileBuffers
// answers ERROR_ACCESS_DENIED), and NTFS carries the rename through its own
// metadata log rather than leaving it for the caller to force. Mirrors
// cache/dirsync_windows.go. See syncDir in dirsync.go.
const dirSyncSupported = false
