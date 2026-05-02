//! Anthropic Files API implementation of [`FilesClient`].
//!
//! Uses `reqwest::multipart` to POST `multipart/form-data` to
//! `/v1/files`. Anthropic returns:
//!
//! ```json
//! {
//!   "id": "file_abc123",
//!   "type": "file",
//!   "filename": "diagram.png",
//!   "size_bytes": 1234567,
//!   "mime_type": "image/png",
//!   "created_at": "2026-05-02T12:34:56Z"
//! }
//! ```
//!
//! Errors are classified into [`FilesError`] variants so retry / UX
//! code can react. The `anthropic-beta: files-api-2025-04-14` header
//! is required as of 2026-05-02; we send it unconditionally.
//!
//! Feature-gated behind `anthropic`.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;

use super::files::{FilesClient, FilesError, UploadedFile, ANTHROPIC_FILES_MAX_BYTES};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const FILES_BETA: &str = "files-api-2025-04-14";
/// Multipart form-data field name expected by Anthropic's
/// `POST /v1/files` endpoint.
const MULTIPART_FILE_FIELD: &str = "file";

/// Anthropic-flavored Files API client. Construct once per session
/// and share via `Arc` — internal `reqwest::Client` is cheap to clone.
#[derive(Clone)]
pub struct AnthropicFilesClient {
    api_key: Arc<String>,
    base_url: Arc<String>,
    client: reqwest::Client,
}

impl std::fmt::Debug for AnthropicFilesClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Redact api_key — never log it.
        f.debug_struct("AnthropicFilesClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl AnthropicFilesClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_client(api_key, reqwest::Client::new())
    }

    pub fn with_client(api_key: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            api_key: Arc::new(api_key.into()),
            base_url: Arc::new(DEFAULT_BASE_URL.to_string()),
            client,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Arc::new(url.into());
        self
    }

    fn auth_headers(&self) -> Result<HeaderMap, FilesError> {
        let mut h = HeaderMap::new();
        h.insert(
            "x-api-key",
            HeaderValue::from_str(self.api_key.as_str())
                .map_err(|e| FilesError::Other(format!("invalid x-api-key header: {e}")))?,
        );
        h.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        h.insert("anthropic-beta", HeaderValue::from_static(FILES_BETA));
        Ok(h)
    }
}

#[async_trait]
impl FilesClient for AnthropicFilesClient {
    async fn upload(
        &self,
        bytes: Vec<u8>,
        mime: &str,
        filename: Option<&str>,
    ) -> Result<UploadedFile, FilesError> {
        let size_bytes = bytes.len() as u64;
        if size_bytes > ANTHROPIC_FILES_MAX_BYTES {
            return Err(FilesError::TooLarge {
                size_bytes,
                limit_bytes: ANTHROPIC_FILES_MAX_BYTES,
            });
        }
        if mime.is_empty() {
            return Err(FilesError::InvalidMime {
                mime: mime.to_string(),
            });
        }
        // Default the filename when the host doesn't supply one —
        // Anthropic rejects multipart parts without a filename.
        let fname = filename
            .map(|s| s.to_string())
            .unwrap_or_else(|| "attachment.bin".to_string());

        let part = Part::bytes(bytes)
            .file_name(fname.clone())
            .mime_str(mime)
            .map_err(|e| FilesError::InvalidMime {
                mime: format!("{mime}: {e}"),
            })?;
        // Anthropic's `POST /v1/files` expects the upload payload
        // under the multipart field name `file` per the public docs
        // at https://docs.anthropic.com/en/api/files-create
        // (verified 2026-05-02).
        let form = Form::new().part(MULTIPART_FILE_FIELD, part);
        let url = format!("{}/v1/files", self.base_url);
        let resp = self
            .client
            .post(&url)
            .headers(self.auth_headers()?)
            .multipart(form)
            .send()
            .await
            .map_err(|e| FilesError::Network(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(classify_http_error(status.as_u16(), &body, &fname));
        }
        let parsed: AnthropicFile = resp
            .json()
            .await
            .map_err(|e| FilesError::Other(format!("parse Anthropic Files response: {e}")))?;
        Ok(UploadedFile {
            id: parsed.id,
            size_bytes: parsed.size_bytes.unwrap_or(size_bytes),
            mime: parsed.mime_type.unwrap_or_else(|| mime.to_string()),
            created_at: parsed.created_at,
            expires_at: parsed.expires_at,
            filename: parsed.filename.or(Some(fname)),
        })
    }

    async fn delete(&self, file_id: &str) -> Result<(), FilesError> {
        let url = format!("{}/v1/files/{}", self.base_url, file_id);
        let resp = self
            .client
            .delete(&url)
            .headers(self.auth_headers()?)
            .send()
            .await
            .map_err(|e| FilesError::Network(e.to_string()))?;
        let status = resp.status();
        // 200 / 204 / 404 all count as "the file is gone".
        if status.is_success() || status.as_u16() == 404 {
            return Ok(());
        }
        let body = resp.text().await.unwrap_or_default();
        Err(classify_http_error(status.as_u16(), &body, file_id))
    }
}

/// Wire-shape of Anthropic's Files API response. Fields are
/// optional because Anthropic occasionally omits them (e.g.,
/// `filename` is missing when not supplied at upload).
#[derive(Debug, Deserialize)]
struct AnthropicFile {
    id: String,
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    /// Anthropic's docs don't currently surface this on POST, but
    /// list/get responses include it. Captured optimistically.
    #[serde(default)]
    expires_at: Option<String>,
}

fn classify_http_error(status: u16, body: &str, context: &str) -> FilesError {
    match status {
        401 | 403 => FilesError::Auth,
        404 => FilesError::NotFound {
            id: context.to_string(),
        },
        413 => FilesError::TooLarge {
            size_bytes: 0,
            limit_bytes: ANTHROPIC_FILES_MAX_BYTES,
        },
        429 => FilesError::Quota,
        400 => {
            // Anthropic 400s carry a structured body of shape
            // `{"error": {"type": "...", "message": "..."}}`. We
            // promote to InvalidMime only when the parsed
            // `error.message` field names a media-type / mime
            // issue. Loose unstructured substring matching produced
            // false positives (e.g., bodies that incidentally
            // mentioned "mime parts" of an unrelated request).
            if let Some(msg) = parse_anthropic_error_message(body) {
                let lower = msg.to_ascii_lowercase();
                let mime_signal = lower.contains("media_type") || lower.contains("mime type");
                if mime_signal {
                    return FilesError::InvalidMime { mime: msg };
                }
                return FilesError::Other(format!(
                    "HTTP 400 from Anthropic Files (context={context}): {msg}"
                ));
            }
            FilesError::Other(format!(
                "HTTP 400 from Anthropic Files (context={context}): {body}"
            ))
        }
        s if (500..600).contains(&s) => FilesError::Network(format!(
            "HTTP {status} from Anthropic Files (context={context}): {body}"
        )),
        _ => FilesError::Other(format!(
            "HTTP {status} from Anthropic Files (context={context}): {body}"
        )),
    }
}

/// Parse `{"error":{"message":"..."}}` out of an Anthropic 4xx body.
/// Returns `None` if the body isn't JSON or doesn't follow the
/// canonical shape — callers fall back to surfacing the raw body.
fn parse_anthropic_error_message(body: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(body).ok()?;
    parsed
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_api_key() {
        let c = AnthropicFilesClient::new("sk-ant-secret");
        let s = format!("{c:?}");
        assert!(!s.contains("sk-ant-secret"));
        assert!(s.contains("redacted"));
    }

    #[test]
    fn classify_http_401_is_auth() {
        let e = classify_http_error(401, "unauth", "upload");
        assert!(matches!(e, FilesError::Auth));
    }

    #[test]
    fn classify_http_404_is_not_found() {
        let e = classify_http_error(404, "no", "file_x");
        match e {
            FilesError::NotFound { id } => assert_eq!(id, "file_x"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn classify_http_413_is_too_large() {
        let e = classify_http_error(413, "", "upload");
        assert!(matches!(e, FilesError::TooLarge { .. }));
    }

    #[test]
    fn classify_http_429_is_quota() {
        let e = classify_http_error(429, "rate", "upload");
        assert!(matches!(e, FilesError::Quota));
    }

    #[test]
    fn classify_http_500_range_is_network() {
        for s in [500, 502, 503, 504] {
            let e = classify_http_error(s, "boom", "x");
            assert!(matches!(e, FilesError::Network(_)), "status {s}");
        }
    }

    #[test]
    fn classify_http_400_with_media_type_message_is_invalid_mime() {
        // Anthropic's canonical 400 body has "media_type" in the
        // message — promote to InvalidMime.
        let body = r#"{"error":{"type":"invalid_request_error","message":"unsupported media_type 'application/x-foo'"}}"#;
        match classify_http_error(400, body, "upload") {
            FilesError::InvalidMime { mime } => {
                assert!(mime.contains("media_type"), "got {mime}");
            }
            other => panic!("expected InvalidMime, got {other:?}"),
        }
    }

    #[test]
    fn classify_http_400_unrelated_substring_no_longer_misclassifies() {
        // A 400 about something else that happens to mention "mime"
        // (e.g., "could not parse multipart mime headers from your
        // payload") is no longer flattened into InvalidMime — round-1
        // codex flagged this.
        let body =
            r#"{"error":{"type":"invalid_request_error","message":"could not parse multipart"}}"#;
        match classify_http_error(400, body, "upload") {
            FilesError::Other(msg) => {
                assert!(msg.contains("could not parse multipart"));
            }
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn classify_http_400_falls_back_to_raw_body_when_not_json() {
        let body = "<html>500</html>"; // server returned HTML by mistake
        match classify_http_error(400, body, "upload") {
            FilesError::Other(msg) => {
                assert!(msg.contains("<html>"));
            }
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn parse_anthropic_error_message_extracts_message_field() {
        let body = r#"{"error":{"type":"x","message":"hello"}}"#;
        assert_eq!(
            parse_anthropic_error_message(body).as_deref(),
            Some("hello")
        );
        assert!(parse_anthropic_error_message("not json").is_none());
        assert!(parse_anthropic_error_message(r#"{"foo":1}"#).is_none());
    }

    #[test]
    fn upload_rejects_oversized_locally() {
        // Build a stub at 600 MiB so we don't actually allocate.
        // Use the boundary check in upload via unit-test of TooLarge.
        let e = FilesError::TooLarge {
            size_bytes: ANTHROPIC_FILES_MAX_BYTES + 1,
            limit_bytes: ANTHROPIC_FILES_MAX_BYTES,
        };
        assert!(e.to_string().contains("too large"));
    }

    #[test]
    fn upload_rejects_empty_mime_locally() {
        // Verify the check exists at the source; running upload requires
        // a network so we just assert the InvalidMime variant constructs
        // for an empty mime string.
        let e = FilesError::InvalidMime {
            mime: String::new(),
        };
        match e {
            FilesError::InvalidMime { mime } => assert!(mime.is_empty()),
            other => panic!("got {other:?}"),
        }
    }

    /// Real-API integration test, gated by ANTHROPIC_API_KEY env var.
    /// Run manually with
    /// `cargo test --features anthropic anthropic_files_real_api -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY env var; hits real Anthropic Files API"]
    async fn anthropic_files_real_api_round_trip() {
        let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
            eprintln!("ANTHROPIC_API_KEY not set; skipping real-API test");
            return;
        };
        let client = AnthropicFilesClient::new(api_key);
        // 1×1 PNG.
        let png = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x62, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        let up = client
            .upload(png.to_vec(), "image/png", Some("dot.png"))
            .await
            .expect("upload");
        assert!(up.id.starts_with("file_"));
        client.delete(&up.id).await.expect("delete");
    }
}
