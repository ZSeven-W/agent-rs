//! Long-term session memory store + extraction (claude-code parity, Tier 1).
//!
//! Mirror of `services/compact/sessionMemoryCompact.ts` (the
//! "promote salient observations from a compaction analysis into
//! durable memory" flow). The actual model-driven extraction lives
//! in [`extract_memories_from_analysis`]; the **storage layer** is
//! pluggable via [`SessionMemoryStore`].

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs;
use uuid::Uuid;

use super::summarize::CompactionResult;

#[derive(Debug, Error)]
pub enum SessionMemoryError {
    #[error("session memory io: {0}")]
    Io(#[from] std::io::Error),
    #[error("session memory json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("session memory: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionMemoryKind {
    Decision,
    Observation,
    Constraint,
    OpenQuestion,
    Requirement,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionMemoryEntry {
    pub id: Uuid,
    pub kind: SessionMemoryKind,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_message_uuids: Vec<Uuid>,
    pub created_at_ms: u64,
}

impl SessionMemoryEntry {
    pub fn new(kind: SessionMemoryKind, text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind,
            text: text.into(),
            source_message_uuids: Vec::new(),
            created_at_ms: now_ms(),
        }
    }

    pub fn with_source_uuids(mut self, uuids: Vec<Uuid>) -> Self {
        self.source_message_uuids = uuids;
        self
    }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Pluggable storage backend.
#[async_trait::async_trait]
pub trait SessionMemoryStore: Send + Sync + std::fmt::Debug {
    async fn append(&self, entry: SessionMemoryEntry) -> Result<(), SessionMemoryError>;
    async fn list(&self) -> Result<Vec<SessionMemoryEntry>, SessionMemoryError>;
    /// Optional: drop everything. Default implementation errors with
    /// `Other("clear not supported")` so consumers can rely on it
    /// being absent.
    async fn clear(&self) -> Result<(), SessionMemoryError> {
        Err(SessionMemoryError::Other("clear not supported".into()))
    }
}

/// In-memory store. Useful for tests + ephemeral runs.
#[derive(Debug, Default, Clone)]
pub struct InMemoryStore {
    inner: Arc<Mutex<Vec<SessionMemoryEntry>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl SessionMemoryStore for InMemoryStore {
    async fn append(&self, entry: SessionMemoryEntry) -> Result<(), SessionMemoryError> {
        self.inner
            .lock()
            .map_err(|_| SessionMemoryError::Other("inmemory lock poisoned".into()))?
            .push(entry);
        Ok(())
    }

    async fn list(&self) -> Result<Vec<SessionMemoryEntry>, SessionMemoryError> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| SessionMemoryError::Other("inmemory lock poisoned".into()))?
            .clone())
    }

    async fn clear(&self) -> Result<(), SessionMemoryError> {
        self.inner
            .lock()
            .map_err(|_| SessionMemoryError::Other("inmemory lock poisoned".into()))?
            .clear();
        Ok(())
    }
}

/// JSONL-backed store. One entry per line. Append-only; `clear`
/// truncates the file.
#[derive(Debug, Clone)]
pub struct JsonlMemoryStore {
    path: PathBuf,
}

impl JsonlMemoryStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[async_trait::async_trait]
impl SessionMemoryStore for JsonlMemoryStore {
    async fn append(&self, entry: SessionMemoryEntry) -> Result<(), SessionMemoryError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        let mut existing = if self.path.exists() {
            fs::read(&self.path).await?
        } else {
            Vec::new()
        };
        let line = serde_json::to_vec(&entry)?;
        existing.extend_from_slice(&line);
        existing.push(b'\n');
        fs::write(&self.path, existing).await?;
        Ok(())
    }

    async fn list(&self) -> Result<Vec<SessionMemoryEntry>, SessionMemoryError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let bytes = fs::read(&self.path).await?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| SessionMemoryError::Other(format!("non-utf8 jsonl: {e}")))?;
        let mut out = Vec::new();
        for line in text.split('\n') {
            if line.trim().is_empty() {
                continue;
            }
            let entry: SessionMemoryEntry = serde_json::from_str(line)?;
            out.push(entry);
        }
        Ok(out)
    }

    async fn clear(&self) -> Result<(), SessionMemoryError> {
        if self.path.exists() {
            fs::write(&self.path, Vec::<u8>::new()).await?;
        }
        Ok(())
    }
}

/// Heuristic extraction of memories from a compaction analysis.
/// Looks for bullet-list lines starting with markers:
///
/// - `DECISION:` / `decision:` — captured as [`SessionMemoryKind::Decision`].
/// - `OBSERVATION:` / `observation:` — [`SessionMemoryKind::Observation`].
/// - `CONSTRAINT:` / `constraint:` — [`SessionMemoryKind::Constraint`].
/// - `REQUIREMENT:` / `requirement:` — [`SessionMemoryKind::Requirement`].
/// - `OPEN QUESTION:` / `OPEN_QUESTION:` / `question:` — [`SessionMemoryKind::OpenQuestion`].
///
/// Bullets without an explicit kind tag default to `Observation`.
/// Caller can run an LLM pass for more precise extraction; this
/// heuristic is the "good enough" fallback.
pub fn extract_memories_from_analysis(
    analysis: &str,
    source_uuids: &[Uuid],
) -> Vec<SessionMemoryEntry> {
    let mut out = Vec::new();
    for raw in analysis.lines() {
        let trimmed = raw.trim_start_matches(|c: char| c == '-' || c == '*' || c.is_whitespace());
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        let (kind, body) = if lower.starts_with("decision:") {
            (
                SessionMemoryKind::Decision,
                trimmed["decision:".len()..].trim(),
            )
        } else if lower.starts_with("observation:") {
            (
                SessionMemoryKind::Observation,
                trimmed["observation:".len()..].trim(),
            )
        } else if lower.starts_with("constraint:") {
            (
                SessionMemoryKind::Constraint,
                trimmed["constraint:".len()..].trim(),
            )
        } else if lower.starts_with("requirement:") {
            (
                SessionMemoryKind::Requirement,
                trimmed["requirement:".len()..].trim(),
            )
        } else if lower.starts_with("open question:") {
            (
                SessionMemoryKind::OpenQuestion,
                trimmed["open question:".len()..].trim(),
            )
        } else if lower.starts_with("open_question:") {
            (
                SessionMemoryKind::OpenQuestion,
                trimmed["open_question:".len()..].trim(),
            )
        } else if lower.starts_with("question:") {
            (
                SessionMemoryKind::OpenQuestion,
                trimmed["question:".len()..].trim(),
            )
        } else {
            // Unmarked bullet — default to Observation.
            (SessionMemoryKind::Observation, trimmed)
        };
        if body.is_empty() {
            continue;
        }
        out.push(SessionMemoryEntry::new(kind, body).with_source_uuids(source_uuids.to_vec()));
    }
    out
}

/// Promote a [`CompactionResult`]'s analysis bullets into the store.
/// Returns the number of entries appended.
pub async fn promote_to_store(
    store: &dyn SessionMemoryStore,
    result: &CompactionResult,
) -> Result<usize, SessionMemoryError> {
    let entries = extract_memories_from_analysis(&result.analysis, &result.replaced_uuids);
    let n = entries.len();
    for e in entries {
        store.append(e).await?;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inmemory_append_and_list() {
        let s = InMemoryStore::new();
        s.append(SessionMemoryEntry::new(SessionMemoryKind::Decision, "X"))
            .await
            .unwrap();
        s.append(SessionMemoryEntry::new(SessionMemoryKind::Observation, "Y"))
            .await
            .unwrap();
        let listed = s.list().await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].text, "X");
        assert_eq!(listed[0].kind, SessionMemoryKind::Decision);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn jsonl_append_persist_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mem.jsonl");
        let s = JsonlMemoryStore::new(&path);
        s.append(SessionMemoryEntry::new(
            SessionMemoryKind::Constraint,
            "must run on macOS",
        ))
        .await
        .unwrap();
        s.append(SessionMemoryEntry::new(
            SessionMemoryKind::OpenQuestion,
            "should we use rmcp?",
        ))
        .await
        .unwrap();

        let s2 = JsonlMemoryStore::new(&path);
        let listed = s2.list().await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].kind, SessionMemoryKind::Constraint);
        assert_eq!(listed[1].kind, SessionMemoryKind::OpenQuestion);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn jsonl_clear_truncates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mem.jsonl");
        let s = JsonlMemoryStore::new(&path);
        s.append(SessionMemoryEntry::new(SessionMemoryKind::Decision, "x"))
            .await
            .unwrap();
        s.clear().await.unwrap();
        let listed = s.list().await.unwrap();
        assert!(listed.is_empty());
    }

    #[test]
    fn extract_memories_from_analysis_recognizes_kinds() {
        let analysis = "
- DECISION: Switch to spawn_blocking for fs4 locks.
- observation: Tests deadlock under current_thread runtime.
- Constraint: Must keep fs4 sync API only.
- Open Question: Should we add notify-based watcher in batch O?
- A trailing unmarked bullet about something else.
";
        let entries = extract_memories_from_analysis(analysis, &[]);
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].kind, SessionMemoryKind::Decision);
        assert_eq!(entries[1].kind, SessionMemoryKind::Observation);
        assert_eq!(entries[2].kind, SessionMemoryKind::Constraint);
        assert_eq!(entries[3].kind, SessionMemoryKind::OpenQuestion);
        // Default for unmarked is Observation.
        assert_eq!(entries[4].kind, SessionMemoryKind::Observation);
        assert!(entries[0].text.contains("Switch to spawn_blocking"));
    }

    #[test]
    fn extract_memories_drops_empty_bodies() {
        let analysis = "- DECISION:\n- OBSERVATION: real text\n- ";
        let entries = extract_memories_from_analysis(analysis, &[]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "real text");
    }

    #[test]
    fn extract_memories_attaches_source_uuids() {
        let uuids = vec![Uuid::new_v4(), Uuid::new_v4()];
        let entries = extract_memories_from_analysis("- DECISION: yes", &uuids);
        assert_eq!(entries[0].source_message_uuids, uuids);
    }

    #[test]
    fn extract_memories_recognizes_requirement() {
        let analysis = "\
- REQUIREMENT: commit messages must never include co-author lines.
- requirement: use 王小明 as the placeholder name in fixtures.";
        let entries = extract_memories_from_analysis(analysis, &[]);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, SessionMemoryKind::Requirement);
        assert_eq!(entries[1].kind, SessionMemoryKind::Requirement);
        assert!(entries[0].text.starts_with("commit messages"));
    }
}
