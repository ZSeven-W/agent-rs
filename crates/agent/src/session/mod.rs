//! JSONL session persistence (Phase 4 / Task 4.2).
//!
//! Schema **v1**, ported per `docs/migration.md`. Storage paths are
//! product-supplied via [`Session::root_for`] /
//! [`Session::openpencil_root`] / [`Session::zode_root`]. The Rust
//! agent does **not** read Zig's legacy `~/.claude/sessions/`.
//!
//! ## File layout
//!
//! ```text
//! line 1: {"schema_version":"v1","agent_version":"0.8.0"}
//! line 2: {"type":"user","header":{...},"content":[...]}
//! line 3: {"type":"assistant","header":{...},"content":[...]}
//! ...
//! ```
//!
//! ## Atomicity
//!
//! `Session::save` writes the full content to a sibling tempfile,
//! `fsync`s it, and atomically renames it over the target path. A
//! partial write of an in-flight save never corrupts the existing
//! session file.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::error::AgentError;
use crate::message::{Message, MessageStore};

pub const SCHEMA_VERSION: &str = "v1";

/// First-line metadata that prefixes every session file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionHeader {
    pub schema_version: String,
    pub agent_version: String,
}

impl SessionHeader {
    pub fn current() -> Self {
        Self {
            schema_version: SCHEMA_VERSION.into(),
            agent_version: crate::VERSION.to_string(),
        }
    }
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("session io: {0}")]
    Io(#[from] std::io::Error),
    #[error("session json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing schema header line — file is empty or malformed")]
    MissingHeader,
    #[error(
        "unsupported schema version: {0} (this build supports {})",
        SCHEMA_VERSION
    )]
    UnsupportedVersion(String),
    #[error("session: {0}")]
    Other(String),
}

impl From<SessionError> for AgentError {
    fn from(e: SessionError) -> Self {
        AgentError::Other(format!("session: {e}"))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Session;

impl Session {
    /// Returns `~/.{product}/agent/sessions/` for an arbitrary
    /// product name. Cross-product friendly — both OpenPencil and
    /// Zode (and any future consumer) call this with their own
    /// product slug. Returns `None` if `$HOME` (Unix) /
    /// `%USERPROFILE%` (Windows) cannot be resolved.
    pub fn root_for(product: &str) -> Option<PathBuf> {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        Some(PathBuf::from(home).join(format!(".{product}/agent/sessions")))
    }

    /// Convenience for OpenPencil: `~/.openpencil/agent/sessions/`.
    /// Equivalent to `Session::root_for("openpencil")`.
    pub fn openpencil_root() -> Option<PathBuf> {
        Self::root_for("openpencil")
    }

    /// Convenience for Zode: `~/.zode/agent/sessions/`.
    /// Equivalent to `Session::root_for("zode")`.
    pub fn zode_root() -> Option<PathBuf> {
        Self::root_for("zode")
    }

    /// Write the full message history of `store` to `path` atomically.
    /// Schema header is the first line; each subsequent line is one
    /// JSON-encoded `Message`.
    pub async fn save(path: impl AsRef<Path>, store: &MessageStore) -> Result<(), SessionError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        let tmp = path.with_extension("jsonl.tmp");

        let header = SessionHeader::current();
        let header_json = serde_json::to_string(&header)?;

        let mut file = fs::File::create(&tmp).await?;
        file.write_all(header_json.as_bytes()).await?;
        file.write_all(b"\n").await?;
        for msg in store.iter() {
            let line = serde_json::to_string(msg)?;
            file.write_all(line.as_bytes()).await?;
            file.write_all(b"\n").await?;
        }
        file.flush().await?;
        file.sync_all().await?;
        drop(file);

        fs::rename(&tmp, path).await?;
        Ok(())
    }

    /// Append `messages` to an existing session file — O(new messages)
    /// instead of the O(total) full rewrite [`save`] does. Used when a turn
    /// only APPENDED to the store (the common case), so a long session's
    /// per-turn I/O stops being quadratic.
    ///
    /// Safe-by-construction: if the file is missing, has no header, or is
    /// shorter than `expected_existing` messages (someone else rewrote it,
    /// or a compaction changed the prefix), this refuses to append and
    /// returns `Ok(false)` so the caller falls back to a full [`save`].
    /// Returns `Ok(true)` when the append happened.
    pub async fn append(
        path: impl AsRef<Path>,
        messages: &[Message],
        expected_existing: usize,
    ) -> Result<bool, SessionError> {
        let path = path.as_ref();
        // Verify the file's current message count matches what the caller
        // believes is already persisted — otherwise appending would produce
        // a corrupt (misordered / duplicated) transcript.
        let actual = match count_messages(path).await {
            Ok(Some(n)) => n,
            Ok(None) => return Ok(false), // missing / headerless → full save
            Err(e) => return Err(e),
        };
        if actual != expected_existing {
            return Ok(false);
        }
        if messages.is_empty() {
            return Ok(true); // nothing to append; file already correct
        }
        // Append is not atomic like the tmp+rename full save, but it only
        // ever adds trailing lines: a torn append leaves already-persisted
        // history intact, and the next full save heals a partial tail.
        let mut file = fs::OpenOptions::new().append(true).open(path).await?;
        for msg in messages {
            let line = serde_json::to_string(msg)?;
            file.write_all(line.as_bytes()).await?;
            file.write_all(b"\n").await?;
        }
        file.flush().await?;
        file.sync_all().await?;
        Ok(true)
    }

    /// Read a session file, validate the schema header, and rebuild a
    /// `MessageStore` with all messages in original order. Unknown
    /// schema → [`SessionError::UnsupportedVersion`].
    pub async fn load(path: impl AsRef<Path>) -> Result<MessageStore, SessionError> {
        let file = fs::File::open(path.as_ref()).await?;
        let mut reader = BufReader::new(file);

        let mut header_line = String::new();
        let header_bytes = reader.read_line(&mut header_line).await?;
        if header_bytes == 0 {
            return Err(SessionError::MissingHeader);
        }
        let header_trim = header_line.trim_end();
        if header_trim.is_empty() {
            return Err(SessionError::MissingHeader);
        }
        let header: SessionHeader = serde_json::from_str(header_trim)?;
        if header.schema_version != SCHEMA_VERSION {
            return Err(SessionError::UnsupportedVersion(header.schema_version));
        }

        let mut store = MessageStore::new();
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = reader.read_line(&mut buf).await?;
            if n == 0 {
                break;
            }
            let line = buf.trim_end();
            if line.is_empty() {
                continue;
            }
            let msg: Message = serde_json::from_str(line)?;
            store.push(msg).map_err(|e| match e {
                AgentError::DuplicateUuid(uuid) => {
                    SessionError::Other(format!("duplicate uuid {uuid} in session file"))
                }
                other => SessionError::Other(other.to_string()),
            })?;
        }
        Ok(store)
    }
}

/// Count the message (non-header, non-blank) lines in a session file.
/// `Ok(None)` when the file is missing or has no valid header — the caller
/// treats that as "can't append, do a full save". Cheap: reads line
/// boundaries without deserializing message bodies.
async fn count_messages(path: &Path) -> Result<Option<usize>, SessionError> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    if reader.read_line(&mut header).await? == 0 || header.trim_end().is_empty() {
        return Ok(None);
    }
    // Validate it's a schema header, not a stray body line.
    if serde_json::from_str::<SessionHeader>(header.trim_end()).is_err() {
        return Ok(None);
    }
    let mut count = 0usize;
    let mut buf = String::new();
    loop {
        buf.clear();
        if reader.read_line(&mut buf).await? == 0 {
            break;
        }
        if !buf.trim_end().is_empty() {
            count += 1;
        }
    }
    Ok(Some(count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Header, Message};
    use tempfile::tempdir;

    fn user(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    fn assistant(text: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[tokio::test]
    async fn save_and_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("s.jsonl");

        let mut store = MessageStore::new();
        let m1 = user("hi");
        let m2 = assistant("hello");
        store.push(m1.clone()).unwrap();
        store.push(m2.clone()).unwrap();

        Session::save(&path, &store).await.unwrap();

        let loaded = Session::load(&path).await.unwrap();
        assert_eq!(loaded.len(), 2);
        let mut iter = loaded.iter();
        assert_eq!(iter.next().unwrap(), &m1);
        assert_eq!(iter.next().unwrap(), &m2);
    }

    #[tokio::test]
    async fn append_extends_an_existing_session() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut store = MessageStore::new();
        let m1 = user("first");
        store.push(m1.clone()).unwrap();
        Session::save(&path, &store).await.unwrap();

        // Append two more, claiming 1 already persisted.
        let m2 = assistant("second");
        let m3 = user("third");
        let ok = Session::append(&path, &[m2.clone(), m3.clone()], 1)
            .await
            .unwrap();
        assert!(ok, "append should succeed when the count matches");

        let loaded = Session::load(&path).await.unwrap();
        let msgs: Vec<_> = loaded.iter().cloned().collect();
        assert_eq!(msgs, vec![m1, m2, m3], "append preserves order");
    }

    #[tokio::test]
    async fn append_refuses_on_count_mismatch() {
        // A wrong watermark (e.g. a compaction rewrote the prefix) must be
        // refused so the caller falls back to a full rewrite — never corrupt.
        let dir = tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut store = MessageStore::new();
        store.push(user("a")).unwrap();
        store.push(assistant("b")).unwrap();
        Session::save(&path, &store).await.unwrap(); // 2 persisted

        // Claim only 1 exists → refuse.
        let ok = Session::append(&path, &[user("c")], 1).await.unwrap();
        assert!(!ok, "count mismatch must refuse the append");
        // File unchanged (still 2 messages).
        assert_eq!(Session::load(&path).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn append_to_missing_file_is_refused() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nope.jsonl");
        let ok = Session::append(&path, &[user("x")], 0).await.unwrap();
        assert!(!ok, "missing file → full save, not append");
    }

    #[tokio::test]
    async fn load_unsupported_version_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("v999.jsonl");
        let bogus = serde_json::json!({
            "schema_version": "v999",
            "agent_version": "0.0.1",
        });
        let body = format!("{bogus}\n");
        tokio::fs::write(&path, body).await.unwrap();

        match Session::load(&path).await {
            Err(SessionError::UnsupportedVersion(v)) => assert_eq!(v, "v999"),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn load_empty_file_errors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        tokio::fs::write(&path, "").await.unwrap();
        match Session::load(&path).await {
            Err(SessionError::MissingHeader) => {}
            other => panic!("expected MissingHeader, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn save_creates_parent_dir() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/deeper/s.jsonl");
        let store = MessageStore::new();
        Session::save(&path, &store).await.unwrap();
        assert!(path.exists());
    }

    #[tokio::test]
    async fn save_atomic_rename_preserves_existing_on_failure() {
        // Sanity-only: a successful save with overwrite leaves no tmp.
        let dir = tempdir().unwrap();
        let path = dir.path().join("atomic.jsonl");
        let mut store = MessageStore::new();
        store.push(user("first")).unwrap();
        Session::save(&path, &store).await.unwrap();
        // Modify and resave.
        store.push(assistant("second")).unwrap();
        Session::save(&path, &store).await.unwrap();
        // Tmp file should be gone (atomic rename).
        let tmp = path.with_extension("jsonl.tmp");
        assert!(!tmp.exists(), "tmp file leaked: {}", tmp.display());

        let loaded = Session::load(&path).await.unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[tokio::test]
    async fn load_skips_blank_trailing_lines() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trailing.jsonl");
        let header = serde_json::to_string(&SessionHeader::current()).unwrap();
        let m = user("hi");
        let m_json = serde_json::to_string(&m).unwrap();
        let body = format!("{header}\n{m_json}\n\n\n");
        tokio::fs::write(&path, body).await.unwrap();
        let loaded = Session::load(&path).await.unwrap();
        assert_eq!(loaded.len(), 1);
    }
}
