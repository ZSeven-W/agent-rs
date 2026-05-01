//! Async file-state cache with LRU + total-size-based eviction (Phase 1 /
//! Task 1.5).
//!
//! Reads file contents through the cache, evicts least-recently-used entries
//! once the cumulative byte budget is exceeded. Used by tool call result
//! caching, prompt cache previews, and any future read-heavy file workflow.
//!
//! Concurrent writes to the underlying files are intentionally **not** locked
//! at the OS level here — that's the responsibility of writers (session JSONL
//! rotation in Phase 4 uses `fs4` exclusive locks behind the
//! `session-jsonl` feature; mailbox in Phase 6 ditto behind `swarm`).
//! The cache assumes the on-disk file is logically point-in-time-consistent
//! during a read; if a writer truncates mid-read the resulting byte buffer
//! is whatever the OS actually returned, and the cached entry is invalid
//! until the next [`Self::invalidate`] call.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lru::LruCache;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use crate::error::AgentError;

#[derive(Debug, Clone)]
struct CacheEntry {
    bytes: Vec<u8>,
}

/// LRU + size-bounded cache over `Path → file bytes`.
#[derive(Debug)]
pub struct FileStateCache {
    inner: Mutex<Inner>,
    max_size_bytes: usize,
}

#[derive(Debug)]
struct Inner {
    cache: LruCache<PathBuf, CacheEntry>,
    current_size_bytes: usize,
}

impl FileStateCache {
    /// Create a new cache.
    ///
    /// - `max_entries` — hard limit on the number of distinct paths cached.
    ///   When exceeded, the least-recently-used entry is evicted regardless
    ///   of the byte budget.
    /// - `max_size_bytes` — soft limit on the cumulative size of all cached
    ///   entries. Eviction continues popping LRU entries until the new
    ///   entry fits. A single entry larger than `max_size_bytes` is
    ///   **not inserted** (returned but uncached).
    pub fn new(max_entries: NonZeroUsize, max_size_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                cache: LruCache::new(max_entries),
                current_size_bytes: 0,
            }),
            max_size_bytes,
        }
    }

    /// Read bytes from `path`. On cache hit, returns the cached bytes
    /// without touching the filesystem. On miss, reads + inserts (subject
    /// to eviction).
    pub async fn read(&self, path: impl AsRef<Path>) -> Result<Vec<u8>, AgentError> {
        let path = path.as_ref();

        if let Some(bytes) = self.try_get(path) {
            return Ok(bytes);
        }

        let mut file = File::open(path).await?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).await?;

        self.insert(path, buf.clone());
        Ok(buf)
    }

    /// Drop the cached entry (if any) for `path`. Idempotent.
    pub fn invalidate(&self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        if let Ok(mut inner) = self.inner.lock() {
            if let Some(prev) = inner.cache.pop(path) {
                inner.current_size_bytes =
                    inner.current_size_bytes.saturating_sub(prev.bytes.len());
            }
        }
    }

    /// Currently cached entry count (after recent operations have settled).
    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.cache.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cumulative byte size of all cached entries.
    pub fn size_bytes(&self) -> usize {
        self.inner.lock().map(|i| i.current_size_bytes).unwrap_or(0)
    }

    fn try_get(&self, path: &Path) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock().ok()?;
        inner.cache.get(path).map(|entry| entry.bytes.clone())
    }

    fn insert(&self, path: &Path, bytes: Vec<u8>) {
        let entry_size = bytes.len();

        // Single entry exceeding the budget is uncacheable — read still
        // succeeds, just doesn't cache.
        if entry_size > self.max_size_bytes {
            return;
        }

        let Ok(mut inner) = self.inner.lock() else {
            return;
        };

        // If the path is already cached, reclaim its old size first so the
        // budget calculation is accurate.
        if let Some(prev) = inner.cache.pop(path) {
            inner.current_size_bytes = inner.current_size_bytes.saturating_sub(prev.bytes.len());
        }

        // Evict LRU entries until the new entry fits.
        while inner.current_size_bytes + entry_size > self.max_size_bytes
            && !inner.cache.is_empty()
        {
            let Some((_, evicted)) = inner.cache.pop_lru() else {
                break;
            };
            inner.current_size_bytes = inner.current_size_bytes.saturating_sub(evicted.bytes.len());
        }

        inner.cache.put(path.to_path_buf(), CacheEntry { bytes });
        inner.current_size_bytes += entry_size;
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn write_temp(name: &str, contents: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(contents).unwrap();
        f.flush().unwrap();
        (dir, path)
    }

    #[tokio::test]
    async fn read_caches_subsequent_reads() {
        let (_d, path) = write_temp("a.txt", b"hello");
        let c = FileStateCache::new(NonZeroUsize::new(8).unwrap(), 1024);
        let a = c.read(&path).await.unwrap();
        let b = c.read(&path).await.unwrap();
        assert_eq!(a, b"hello");
        assert_eq!(b, b"hello");
        assert_eq!(c.len(), 1);
        assert_eq!(c.size_bytes(), 5);
    }

    #[tokio::test]
    async fn lru_evicts_oldest_on_capacity() {
        let c = FileStateCache::new(NonZeroUsize::new(2).unwrap(), 1024);
        let (_d1, p1) = write_temp("a.txt", b"aaaa");
        let (_d2, p2) = write_temp("b.txt", b"bbbb");
        let (_d3, p3) = write_temp("c.txt", b"cccc");
        c.read(&p1).await.unwrap();
        c.read(&p2).await.unwrap();
        c.read(&p3).await.unwrap(); // evicts p1 (LRU)
        assert_eq!(c.len(), 2);
        // After p3 read, p1 should not be in cache.
        assert!(c.try_get(&p1).is_none());
        assert!(c.try_get(&p2).is_some());
        assert!(c.try_get(&p3).is_some());
    }

    #[tokio::test]
    async fn size_budget_evicts_when_exceeded() {
        // 2 entries fit nominally, but byte budget = 6 — only one big entry fits.
        let c = FileStateCache::new(NonZeroUsize::new(8).unwrap(), 6);
        let (_d1, p1) = write_temp("a.txt", b"aaaa"); // 4
        let (_d2, p2) = write_temp("b.txt", b"bbbb"); // 4 — would push total to 8
        c.read(&p1).await.unwrap();
        assert_eq!(c.size_bytes(), 4);
        c.read(&p2).await.unwrap();
        // p1 evicted because 4+4 > 6.
        assert_eq!(c.size_bytes(), 4);
        assert!(c.try_get(&p1).is_none());
        assert!(c.try_get(&p2).is_some());
    }

    #[tokio::test]
    async fn entry_larger_than_budget_is_not_cached() {
        let c = FileStateCache::new(NonZeroUsize::new(8).unwrap(), 4);
        let (_d, path) = write_temp("big.txt", b"too-big-for-budget"); // 18
        let bytes = c.read(&path).await.unwrap();
        assert_eq!(bytes, b"too-big-for-budget");
        // Read still returned the file, but cache stays empty.
        assert_eq!(c.len(), 0);
        assert_eq!(c.size_bytes(), 0);
    }

    #[tokio::test]
    async fn invalidate_removes_entry() {
        let (_d, path) = write_temp("a.txt", b"hello");
        let c = FileStateCache::new(NonZeroUsize::new(8).unwrap(), 1024);
        c.read(&path).await.unwrap();
        assert_eq!(c.len(), 1);
        c.invalidate(&path);
        assert!(c.is_empty());
        assert_eq!(c.size_bytes(), 0);
    }

    #[tokio::test]
    async fn read_missing_file_errors() {
        let c = FileStateCache::new(NonZeroUsize::new(8).unwrap(), 1024);
        match c.read("/no/such/path/here.dat").await {
            Err(AgentError::Io(_)) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn double_insert_replaces_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");

        // First version: 4 bytes
        std::fs::write(&path, b"aaaa").unwrap();
        let c = FileStateCache::new(NonZeroUsize::new(8).unwrap(), 1024);
        c.read(&path).await.unwrap();
        assert_eq!(c.size_bytes(), 4);

        // Replace on disk with 10 bytes; force re-read via invalidate.
        std::fs::write(&path, b"bbbbbbbbbb").unwrap();
        c.invalidate(&path);
        c.read(&path).await.unwrap();
        assert_eq!(c.size_bytes(), 10);
    }
}
