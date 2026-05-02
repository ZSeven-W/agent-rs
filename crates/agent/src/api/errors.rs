//! API error classification + parsing (Tier 1 / claude-code parity).
//!
//! Mirrors `services/api/errors.ts`: turn an opaque provider error
//! body or status code into a structured [`ParsedApiError`] that
//! callers can inspect for retry decisions, user-facing messages, and
//! billing/quota surfacing.
//!
//! Two entry points:
//!
//! - [`parse_api_error`] for the typical shape Anthropic / OpenAI
//!   return: a JSON body with `{"error": {"type", "message"}}`.
//! - [`ApiErrorKind::from_status`] for status-only fallback when the
//!   body is unreadable.

use serde::{Deserialize, Serialize};

/// High-level error category. Picked to be cross-provider — Anthropic
/// and OpenAI use different `error.type` strings, so we map both into
/// the same enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ApiErrorKind {
    /// 401 / 403 — bad or missing API key.
    Authentication,
    /// 400 — request body violates schema.
    InvalidRequest,
    /// 404 — model id unknown or no longer available.
    ModelNotFound,
    /// Model is deprecated and being sunset (special body shape on
    /// Anthropic). Caller should fall back to a current model.
    ModelDeprecated,
    /// 429 — caller exceeded their rate limit / TPM / RPM.
    RateLimit,
    /// 529 (Anthropic-specific) — service is overloaded. Distinct
    /// from RateLimit because it's a server-side condition; retry
    /// budget should be larger.
    Overloaded,
    /// 402 / 403 with billing reason — quota exhausted.
    QuotaExceeded,
    /// Content policy / safety filter rejection.
    ContentPolicy,
    /// File reference invalid (Files API) — e.g., file expired,
    /// wrong account.
    InvalidFile,
    /// 500 / 502 / 503 / 504 — generic transient server error.
    ServerError,
    /// Network-level: DNS, TLS, connection reset, EOF before headers.
    Network,
    /// Anything that doesn't fit a known bucket.
    Unknown,
}

impl ApiErrorKind {
    /// Best-effort classification from HTTP status alone — used when
    /// the body parser fails (binary response, truncated stream).
    pub fn from_status(status: u16) -> Self {
        match status {
            401 | 403 => Self::Authentication,
            400 => Self::InvalidRequest,
            404 => Self::ModelNotFound,
            429 => Self::RateLimit,
            529 => Self::Overloaded,
            500..=504 => Self::ServerError,
            _ => Self::Unknown,
        }
    }

    /// Whether retrying the request is worth attempting. Mirrors the
    /// `RetryClassification` in [`super::retry`] but at the higher
    /// API-error layer.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::RateLimit | Self::Overloaded | Self::ServerError | Self::Network
        )
    }

    /// Whether the caller should rotate credentials / re-prompt the
    /// user before retrying.
    pub fn needs_credential_refresh(self) -> bool {
        matches!(self, Self::Authentication)
    }
}

/// Parsed error envelope. The original `body` slice is preserved so
/// the host UI can show the raw payload for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedApiError {
    pub kind: ApiErrorKind,
    pub status: Option<u16>,
    /// Human-readable message extracted from the JSON envelope when
    /// possible, else the first non-empty line of the body.
    pub message: String,
    /// Original provider error type string (e.g., "rate_limit_error",
    /// "overloaded_error", "invalid_request_error"). Useful for
    /// telemetry segmentation.
    pub provider_type: Option<String>,
    /// Optional retry-after hint in seconds. Anthropic 429s set this.
    pub retry_after_seconds: Option<u32>,
}

/// Parse a provider error response. `status` is the HTTP status code
/// (or `None` if the call failed below the HTTP layer — e.g., DNS
/// error, in which case `body` is the OS error message). `body` is the
/// raw response body.
pub fn parse_api_error(status: Option<u16>, body: &str) -> ParsedApiError {
    if status.is_none() && !body.is_empty() {
        let kind = if body.to_lowercase().contains("dns")
            || body.to_lowercase().contains("connection")
            || body.to_lowercase().contains("tls")
        {
            ApiErrorKind::Network
        } else {
            ApiErrorKind::Unknown
        };
        return ParsedApiError {
            kind,
            status: None,
            message: body.lines().next().unwrap_or(body).to_string(),
            provider_type: None,
            retry_after_seconds: None,
        };
    }

    // Try the canonical { "error": { "type", "message" } } shape.
    if let Ok(envelope) = serde_json::from_str::<ErrorEnvelope>(body) {
        let provider_type = envelope.error.r#type.clone();
        let message = envelope.error.message.unwrap_or_default();
        let kind = classify_provider_type(provider_type.as_deref(), status, &message);
        return ParsedApiError {
            kind,
            status,
            message,
            provider_type,
            retry_after_seconds: None,
        };
    }

    // Fallback: status-only classification.
    let kind = status
        .map(ApiErrorKind::from_status)
        .unwrap_or(ApiErrorKind::Unknown);
    let message = body.lines().next().unwrap_or("").trim().to_string();
    ParsedApiError {
        kind,
        status,
        message,
        provider_type: None,
        retry_after_seconds: None,
    }
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    r#type: Option<String>,
    message: Option<String>,
}

fn classify_provider_type(
    provider_type: Option<&str>,
    status: Option<u16>,
    message: &str,
) -> ApiErrorKind {
    if let Some(t) = provider_type {
        // Some Anthropic deprecation responses come through as
        // `invalid_request_error` with the deprecation hint in the
        // message body — check that first so we don't lose the
        // signal.
        if message.to_lowercase().contains("deprecated") {
            return ApiErrorKind::ModelDeprecated;
        }
        match t {
            "authentication_error" | "permission_error" => return ApiErrorKind::Authentication,
            "invalid_request_error" => return ApiErrorKind::InvalidRequest,
            "not_found_error" => return ApiErrorKind::ModelNotFound,
            "rate_limit_error" => return ApiErrorKind::RateLimit,
            "overloaded_error" => return ApiErrorKind::Overloaded,
            "billing_error" | "quota_exceeded" => return ApiErrorKind::QuotaExceeded,
            "content_policy_violation" | "policy_violation" => return ApiErrorKind::ContentPolicy,
            "invalid_file_error" | "file_not_found_error" => return ApiErrorKind::InvalidFile,
            "api_error" | "internal_server_error" => return ApiErrorKind::ServerError,
            _ => {}
        }
    }
    status
        .map(ApiErrorKind::from_status)
        .unwrap_or(ApiErrorKind::Unknown)
}

/// Set the retry-after hint after parsing, e.g. from a `Retry-After`
/// HTTP header. Caller-side update because header parsing happens
/// outside this module.
pub fn with_retry_after(mut err: ParsedApiError, secs: u32) -> ParsedApiError {
    err.retry_after_seconds = Some(secs);
    err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_anthropic_rate_limit_envelope() {
        let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded for org X"}}"#;
        let parsed = parse_api_error(Some(429), body);
        assert_eq!(parsed.kind, ApiErrorKind::RateLimit);
        assert_eq!(parsed.provider_type.as_deref(), Some("rate_limit_error"));
        assert!(parsed.message.contains("Rate limit exceeded"));
    }

    #[test]
    fn parse_anthropic_overloaded_envelope() {
        let body =
            r#"{"error":{"type":"overloaded_error","message":"Service is currently overloaded"}}"#;
        let parsed = parse_api_error(Some(529), body);
        assert_eq!(parsed.kind, ApiErrorKind::Overloaded);
    }

    #[test]
    fn parse_authentication_error() {
        let body = r#"{"error":{"type":"authentication_error","message":"Invalid API key"}}"#;
        let parsed = parse_api_error(Some(401), body);
        assert_eq!(parsed.kind, ApiErrorKind::Authentication);
        assert!(parsed.kind.needs_credential_refresh());
        assert!(!parsed.kind.is_retryable());
    }

    #[test]
    fn parse_model_deprecated_via_message() {
        let body = r#"{"error":{"type":"invalid_request_error","message":"This model has been deprecated. Please migrate."}}"#;
        let parsed = parse_api_error(Some(400), body);
        assert_eq!(parsed.kind, ApiErrorKind::ModelDeprecated);
    }

    #[test]
    fn fallback_to_status_when_body_unparseable() {
        let parsed = parse_api_error(Some(503), "<html><body>upstream gateway</body></html>");
        assert_eq!(parsed.kind, ApiErrorKind::ServerError);
    }

    #[test]
    fn no_status_with_network_body_is_network() {
        let parsed = parse_api_error(None, "DNS resolution failed for api.anthropic.com");
        assert_eq!(parsed.kind, ApiErrorKind::Network);
    }

    #[test]
    fn no_status_unknown_body_is_unknown() {
        let parsed = parse_api_error(None, "");
        assert_eq!(parsed.kind, ApiErrorKind::Unknown);
    }

    #[test]
    fn retryability_matrix() {
        assert!(ApiErrorKind::RateLimit.is_retryable());
        assert!(ApiErrorKind::Overloaded.is_retryable());
        assert!(ApiErrorKind::ServerError.is_retryable());
        assert!(ApiErrorKind::Network.is_retryable());
        assert!(!ApiErrorKind::Authentication.is_retryable());
        assert!(!ApiErrorKind::InvalidRequest.is_retryable());
        assert!(!ApiErrorKind::ModelNotFound.is_retryable());
        assert!(!ApiErrorKind::QuotaExceeded.is_retryable());
        assert!(!ApiErrorKind::ContentPolicy.is_retryable());
        assert!(!ApiErrorKind::InvalidFile.is_retryable());
        assert!(!ApiErrorKind::Unknown.is_retryable());
    }

    #[test]
    fn with_retry_after_attaches_hint() {
        let body = r#"{"error":{"type":"rate_limit_error","message":"Slow down"}}"#;
        let parsed = with_retry_after(parse_api_error(Some(429), body), 12);
        assert_eq!(parsed.retry_after_seconds, Some(12));
    }
}
