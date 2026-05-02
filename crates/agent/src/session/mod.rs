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
