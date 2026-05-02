//! Files API client trait and shared types.
//!
//! agent-rs ships the trait + error / metadata types here; concrete
//! transports (Anthropic Files API, OpenAI Files API, S3-style
//! pre-signed URL stores) implement [`FilesClient`] in their own
//! module. The Anthropic-specific impl lives at
//! [`crate::attachments::anthropic_client::AnthropicFilesClient`]
//! when the `anthropic` feature is enabled.
//!
//! # Why a trait
//!
//! Hosts that target a single provider can wire up the matching
//! impl directly. Hosts that swap providers (or add custom storage —
//! e.g., a private intranet file store) inject their own impl.
//! Tests use [`InMemoryFilesClient`] to simulate uploads without
//! network I/O.

use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;

use super::ImageMime;

/// Metadata returned by [`FilesClient::upload`].
///
/// `id` is the provider-side handle that hosts pass to
/// [`crate::message::ImageSource::File`] /
/// [`crate::message::DocumentSource::File`]. The remaining fields
/// are best-effort: providers that don't expose them set them to
/// `None`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UploadedFile {
    pub id: String,
    pub size_bytes: u64,
    pub mime: String,
    /// RFC 3339 timestamp from the provider, if surfaced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// RFC 3339 timestamp at which the provider will GC the file.
    /// Anthropic currently retains for 30 days; OpenAI varies by
    /// endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    /// Filename echoed by the provider, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Errors from the Files API. Discriminated so hosts can react —
/// `Quota` and `TooLarge` typically map to a user-visible "your
/// upload was rejected" notice, while `Network` / `Auth` warrant a
/// retry-with-backoff or re-auth.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FilesError {
    #[error("file too large: {size_bytes} bytes (limit {limit_bytes})")]
    TooLarge { size_bytes: u64, limit_bytes: u64 },
    #[error("provider quota exceeded")]
    Quota,
    #[error("invalid mime type '{mime}'")]
    InvalidMime { mime: String },
    #[error("authentication failed")]
    Auth,
    #[error("file not found: {id}")]
    NotFound { id: String },
    #[error("network error: {0}")]
    Network(String),
    #[error("provider error: {0}")]
    Other(String),
}

/// Anthropic's Files API hard ceiling for any single upload as
/// published at <https://docs.anthropic.com/en/api/files-create>
/// (verified 2026-05-02). The wire response uses 413 if exceeded.
/// Treat this as the *current* upper bound — Anthropic has bumped
/// it before; hosts running a long-lived deployment should
/// occasionally reverify.
pub const ANTHROPIC_FILES_MAX_BYTES: u64 = 500 * 1024 * 1024;

/// Pluggable Files API client. Implementations are typically thin
/// HTTP wrappers — see
/// [`crate::attachments::anthropic_client::AnthropicFilesClient`]
/// for the canonical example.
///
/// `Debug` is required because the crate-wide
/// `missing_debug_implementations` lint enforces it; impls holding
/// secrets (API keys) should redact them in their manual `Debug`.
#[async_trait]
pub trait FilesClient: Send + Sync + std::fmt::Debug {
    /// Upload a file. `mime` is the wire-level mime string
    /// (`image/png`, `application/pdf`, …). `filename` is optional;
    /// providers use it for catalogs / billing line items.
    async fn upload(
        &self,
        bytes: Vec<u8>,
        mime: &str,
        filename: Option<&str>,
    ) -> Result<UploadedFile, FilesError>;

    /// Delete a previously-uploaded file. `Ok(())` on success or
    /// when the file was already gone (idempotent — providers
    /// frequently 404 on already-deleted files; we treat that as
    /// success so retry loops are safe).
    async fn delete(&self, file_id: &str) -> Result<(), FilesError>;
}

/// In-memory `FilesClient` impl for tests. Stores bytes keyed by a
/// monotonically-increasing id; tracks call counts so test
/// assertions can verify the host actually uploaded.
#[derive(Debug, Default)]
pub struct InMemoryFilesClient {
    inner: Mutex<InMemoryState>,
}

#[derive(Debug, Default)]
struct InMemoryState {
    next_id: u64,
    files: BTreeMap<String, StoredFile>,
    upload_calls: u64,
    delete_calls: u64,
}

#[derive(Debug, Clone)]
struct StoredFile {
    bytes: Vec<u8>,
    /// Captured for diagnostics + parity with what a real provider
    /// would return; not currently inspected in tests, but cheap to
    /// hold and useful when stepping through with a debugger.
    #[allow(dead_code)]
    mime: String,
    #[allow(dead_code)]
    filename: Option<String>,
}

impl InMemoryFilesClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of `upload` calls observed.
    pub fn upload_count(&self) -> u64 {
        self.inner
            .lock()
            .map(|g| g.upload_calls)
            .unwrap_or_default()
    }

    /// Number of `delete` calls observed.
    pub fn delete_count(&self) -> u64 {
        self.inner
            .lock()
            .map(|g| g.delete_calls)
            .unwrap_or_default()
    }

    /// Retrieve the bytes previously uploaded under `file_id`.
    pub fn get_bytes(&self, file_id: &str) -> Option<Vec<u8>> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.files.get(file_id).map(|f| f.bytes.clone()))
    }
}

#[async_trait]
impl FilesClient for InMemoryFilesClient {
    async fn upload(
        &self,
        bytes: Vec<u8>,
        mime: &str,
        filename: Option<&str>,
    ) -> Result<UploadedFile, FilesError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| FilesError::Other("InMemoryFilesClient mutex poisoned".to_string()))?;
        guard.upload_calls = guard.upload_calls.saturating_add(1);
        let id = format!("file_mem_{}", guard.next_id);
        guard.next_id = guard.next_id.saturating_add(1);
        let size_bytes = bytes.len() as u64;
        let stored_filename = filename.map(|s| s.to_string());
        guard.files.insert(
            id.clone(),
            StoredFile {
                bytes,
                mime: mime.to_string(),
                filename: stored_filename.clone(),
            },
        );
        Ok(UploadedFile {
            id,
            size_bytes,
            mime: mime.to_string(),
            created_at: None,
            expires_at: None,
            filename: stored_filename,
        })
    }

    async fn delete(&self, file_id: &str) -> Result<(), FilesError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| FilesError::Other("InMemoryFilesClient mutex poisoned".to_string()))?;
        guard.delete_calls = guard.delete_calls.saturating_add(1);
        guard.files.remove(file_id);
        Ok(())
    }
}

/// Resolve the wire-level mime string for an [`ImageMime`].
/// Convenience indirection so call-sites that already hold
/// `ImageMime` don't have to call `.as_str()` plus deal with the
/// `&'static str` borrow.
pub fn image_mime_string(mime: ImageMime) -> String {
    mime.as_str().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_round_trip() {
        let client = InMemoryFilesClient::new();
        let up = client
            .upload(vec![1, 2, 3], "image/png", Some("a.png"))
            .await
            .unwrap();
        assert_eq!(up.size_bytes, 3);
        assert_eq!(up.mime, "image/png");
        assert_eq!(up.filename.as_deref(), Some("a.png"));
        assert_eq!(client.upload_count(), 1);
        assert_eq!(client.get_bytes(&up.id).unwrap(), vec![1, 2, 3]);
        client.delete(&up.id).await.unwrap();
        assert_eq!(client.delete_count(), 1);
        assert!(client.get_bytes(&up.id).is_none());
    }

    #[tokio::test]
    async fn delete_unknown_id_is_idempotent() {
        let client = InMemoryFilesClient::new();
        // Per FilesClient contract, deleting a missing id is OK.
        client.delete("file_does_not_exist").await.unwrap();
        assert_eq!(client.delete_count(), 1);
    }

    #[test]
    fn files_error_display() {
        let too_large = FilesError::TooLarge {
            size_bytes: 6_000_000,
            limit_bytes: 5_000_000,
        };
        assert!(too_large.to_string().contains("6000000"));
        assert!(too_large.to_string().contains("5000000"));
        let quota = FilesError::Quota;
        assert_eq!(quota.to_string(), "provider quota exceeded");
    }

    #[test]
    fn anthropic_max_is_500_mib() {
        assert_eq!(ANTHROPIC_FILES_MAX_BYTES, 500 * 1024 * 1024);
    }

    #[test]
    fn image_mime_string_round_trips() {
        assert_eq!(image_mime_string(ImageMime::Png), "image/png");
        assert_eq!(image_mime_string(ImageMime::Jpeg), "image/jpeg");
    }

    #[tokio::test]
    async fn upload_count_increments_on_each_call() {
        let client = InMemoryFilesClient::new();
        for i in 0..5 {
            client
                .upload(vec![i as u8], "image/png", None)
                .await
                .unwrap();
        }
        assert_eq!(client.upload_count(), 5);
    }
}
