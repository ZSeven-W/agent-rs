//! File-locked mailbox for inter-agent messaging (Phase 6 / Task 6.1).
//!
//! Each agent owns a JSONL file under
//! `<team_root>/inboxes/<agent_id>.jsonl`. Writers append envelopes via
//! an exclusive `fs4` lock; readers drain (read + truncate) under the
//! same lock so concurrent senders never see partial writes or lose
//! messages.
//!
//! ## File layout
//!
//! ```text
//! line 1: {"schema_version":"v1","agent_version":"0.8.0"}
//! line 2: {"id":"<uuid>","from":"<agent>","to":"<agent>","timestamp_ms":...,"payload":{...}}
//! line 3: ...
//! ```
//!
//! ## Concurrency contract
//!
//! - `send(envelope)` — open + flock(EX) + append + flush + unlock.
//! - `drain()` — open + flock(EX) + read all + rewrite header-only + unlock.
//!
//! All file I/O runs inside [`tokio::task::spawn_blocking`] because
//! `fs4::FileExt::lock_exclusive` is a sync OS syscall that blocks the
//! calling thread. Running it on the blocking pool keeps the tokio
//! runtime worker threads free to make progress on other tasks even
//! while many senders contend for the same lock.

use std::path::{Path, PathBuf};

use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs;
use uuid::Uuid;

pub const MAILBOX_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Error)]
pub enum MailboxError {
    #[error("mailbox io: {0}")]
    Io(#[from] std::io::Error),
    #[error("mailbox json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing schema header")]
    MissingHeader,
    #[error(
        "unsupported schema version: {0} (this build expects {})",
        MAILBOX_SCHEMA_VERSION
    )]
    UnsupportedVersion(String),
    #[error("mailbox: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MailboxHeader {
    pub schema_version: String,
    pub agent_version: String,
}

impl MailboxHeader {
    pub fn current() -> Self {
        Self {
            schema_version: MAILBOX_SCHEMA_VERSION.into(),
            agent_version: crate::VERSION.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MailboxMessage {
    pub id: Uuid,
    pub from: String,
    pub to: String,
    pub timestamp_ms: u64,
    pub payload: serde_json::Value,
}

impl MailboxMessage {
    pub fn new(from: impl Into<String>, to: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            from: from.into(),
            to: to.into(),
            timestamp_ms: now_ms(),
            payload,
        }
    }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// One agent's inbox.
///
/// Multiple writers may concurrently `send()` to the same Mailbox path;
/// the file lock serializes the appends. A single reader drains under
/// the same lock with [`Self::drain`].
#[derive(Debug, Clone)]
pub struct Mailbox {
    path: PathBuf,
}

impl Mailbox {
    /// Construct a Mailbox handle for `<team_root>/inboxes/<agent_id>.jsonl`.
    /// Creates parent directories if missing. Does NOT yet create the
    /// file — that happens lazily on first send/drain so empty-team
    /// queries don't litter the filesystem.
    pub async fn for_agent(
        team_root: impl AsRef<Path>,
        agent_id: &str,
    ) -> Result<Self, MailboxError> {
        let inbox_dir = team_root.as_ref().join("inboxes");
        fs::create_dir_all(&inbox_dir).await?;
        let path = inbox_dir.join(format!("{agent_id}.jsonl"));
        Ok(Self { path })
    }

    /// Construct a Mailbox at `<agent_root>/teams/{team_name}/inboxes/<agent_id>.jsonl`.
    /// Convenience that pairs the team name + product root.
    pub async fn for_team_agent(
        agent_root: impl AsRef<Path>,
        team_name: &str,
        agent_id: &str,
    ) -> Result<Self, MailboxError> {
        let team_root = agent_root.as_ref().join("teams").join(team_name);
        Self::for_agent(team_root, agent_id).await
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append `message` to the inbox. Holds the file's exclusive lock
    /// only for the duration of the open+append+flush.
    pub async fn send(&self, message: &MailboxMessage) -> Result<(), MailboxError> {
        let path = self.path.clone();
        let header_line = serde_json::to_string(&MailboxHeader::current())?;
        let line = serde_json::to_string(message)?;
        spawn_blocking(move || -> Result<(), MailboxError> {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(&path)?;
            file.lock_exclusive()?;
            let len = file.metadata()?.len();
            if len == 0 {
                file.write_all(header_line.as_bytes())?;
                file.write_all(b"\n")?;
            }
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
            file.flush()?;
            // unlock on Drop via fs4's FileExt impl
            Ok(())
        })
        .await
    }

    /// Read all queued messages, truncate the file (re-write header),
    /// and return the collected envelopes. If the file doesn't exist
    /// yet, returns an empty Vec.
    pub async fn drain(&self) -> Result<Vec<MailboxMessage>, MailboxError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let path = self.path.clone();
        let header_line = serde_json::to_string(&MailboxHeader::current())?;
        spawn_blocking(move || -> Result<Vec<MailboxMessage>, MailboxError> {
            use std::io::{Read, Seek, SeekFrom, Write};
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)?;
            file.lock_exclusive()?;

            file.seek(SeekFrom::Start(0))?;
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)?;
            let messages = parse_messages(&contents)?;

            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(header_line.as_bytes())?;
            file.write_all(b"\n")?;
            file.flush()?;
            Ok(messages)
        })
        .await
    }

    /// Inspect without draining. Returns Ok(Vec::new()) if file doesn't
    /// exist. Holds an exclusive lock briefly.
    pub async fn peek(&self) -> Result<Vec<MailboxMessage>, MailboxError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let path = self.path.clone();
        spawn_blocking(move || -> Result<Vec<MailboxMessage>, MailboxError> {
            use std::io::Read;
            let mut file = std::fs::OpenOptions::new().read(true).open(&path)?;
            file.lock_exclusive()?;
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)?;
            parse_messages(&contents)
        })
        .await
    }

    /// Lock-free peek — may see a partial write if a sender is
    /// mid-append. Use only when stale data is acceptable.
    pub async fn peek_unlocked(&self) -> Result<Vec<MailboxMessage>, MailboxError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let bytes = fs::read(&self.path).await?;
        Ok(parse_messages_lenient(&bytes))
    }
}

async fn spawn_blocking<F, T>(f: F) -> Result<T, MailboxError>
where
    F: FnOnce() -> Result<T, MailboxError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| MailboxError::Other(format!("blocking task: {e}")))?
}

fn parse_messages(bytes: &[u8]) -> Result<Vec<MailboxMessage>, MailboxError> {
    if bytes.is_empty() {
        return Err(MailboxError::MissingHeader);
    }
    let s = std::str::from_utf8(bytes)
        .map_err(|e| MailboxError::Other(format!("non-utf8 mailbox: {e}")))?;
    let lines: Vec<&str> = s.split('\n').filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return Err(MailboxError::MissingHeader);
    }
    let header: MailboxHeader = serde_json::from_str(lines[0])?;
    if header.schema_version != MAILBOX_SCHEMA_VERSION {
        return Err(MailboxError::UnsupportedVersion(header.schema_version));
    }
    let mut messages = Vec::with_capacity(lines.len().saturating_sub(1));
    for line in lines.iter().skip(1) {
        let m: MailboxMessage = serde_json::from_str(line)?;
        messages.push(m);
    }
    Ok(messages)
}

/// Lenient parser: tolerates trailing partial line from a concurrent
/// writer. Returns an empty Vec rather than erroring on a malformed
/// last line.
fn parse_messages_lenient(bytes: &[u8]) -> Vec<MailboxMessage> {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return Vec::new();
    };
    let lines: Vec<&str> = s.split('\n').filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let Ok(header) = serde_json::from_str::<MailboxHeader>(lines[0]) else {
        return Vec::new();
    };
    if header.schema_version != MAILBOX_SCHEMA_VERSION {
        return Vec::new();
    }
    let mut out = Vec::new();
    for line in lines.iter().skip(1) {
        if let Ok(m) = serde_json::from_str::<MailboxMessage>(line) {
            out.push(m);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use tempfile::tempdir;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn for_agent_creates_inbox_dir() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_agent(dir.path(), "alice").await.unwrap();
        let parent = mb.path().parent().unwrap();
        assert!(parent.exists());
        assert_eq!(parent.file_name().unwrap(), "inboxes");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn send_then_drain_roundtrip() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_agent(dir.path(), "alice").await.unwrap();
        let m1 = MailboxMessage::new("leader", "alice", serde_json::json!({"task": 1}));
        let m2 = MailboxMessage::new("leader", "alice", serde_json::json!({"task": 2}));
        mb.send(&m1).await.unwrap();
        mb.send(&m2).await.unwrap();

        let drained = mb.drain().await.unwrap();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].id, m1.id);
        assert_eq!(drained[1].id, m2.id);
        let again = mb.drain().await.unwrap();
        assert!(again.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn peek_does_not_consume() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_agent(dir.path(), "alice").await.unwrap();
        let m = MailboxMessage::new("a", "b", serde_json::json!(null));
        mb.send(&m).await.unwrap();
        let peeked1 = mb.peek().await.unwrap();
        let peeked2 = mb.peek().await.unwrap();
        assert_eq!(peeked1.len(), 1);
        assert_eq!(peeked2.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_on_missing_returns_empty() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_agent(dir.path(), "alice").await.unwrap();
        let drained = mb.drain().await.unwrap();
        assert!(drained.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unsupported_version_errors() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_agent(dir.path(), "alice").await.unwrap();
        let bogus_header = serde_json::json!({
            "schema_version": "v999",
            "agent_version": "0.0.1",
        });
        let body = format!("{bogus_header}\n");
        fs::write(mb.path(), body).await.unwrap();
        match mb.peek().await {
            Err(MailboxError::UnsupportedVersion(v)) => assert_eq!(v, "v999"),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_senders_no_data_loss() {
        let dir = tempdir().unwrap();
        let mb = Arc::new(Mailbox::for_agent(dir.path(), "shared").await.unwrap());

        let n = 20;
        let mut tasks = Vec::new();
        for i in 0..n {
            let mb = mb.clone();
            tasks.push(tokio::spawn(async move {
                let m = MailboxMessage::new(
                    format!("worker-{i}"),
                    "shared",
                    serde_json::json!({"i": i}),
                );
                mb.send(&m).await.unwrap();
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        let drained = mb.drain().await.unwrap();
        assert_eq!(
            drained.len(),
            n,
            "expected {n} messages, got {}",
            drained.len()
        );

        let mut seen = std::collections::HashSet::new();
        for m in &drained {
            let i = m.payload["i"].as_u64().unwrap();
            assert!(seen.insert(i), "duplicate i={i}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn for_team_agent_path_includes_team_and_inboxes() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_team_agent(dir.path(), "design-squad", "bob")
            .await
            .unwrap();
        let path = mb.path();
        // Walk components rather than substring-matching the rendered
        // path — Windows uses `\` while Unix uses `/`, so a literal
        // "teams/design-squad/inboxes" would not match on Windows CI.
        let comps: Vec<String> = path
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let teams_idx = comps
            .iter()
            .position(|c| c == "teams")
            .expect("path missing `teams` component");
        assert_eq!(
            comps.get(teams_idx + 1).map(String::as_str),
            Some("design-squad")
        );
        assert_eq!(
            comps.get(teams_idx + 2).map(String::as_str),
            Some("inboxes")
        );
        assert!(path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("bob"));
    }
}
