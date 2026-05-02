//! MCP elicitation handler (Tier 1 / claude-code parity).
//!
//! Mirrors `services/mcp/elicitationHandler.ts`. An MCP server may
//! ask the host to elicit user action — typically an OAuth browser
//! login during initial connect, but also user confirmation prompts
//! before destructive operations. This module defines the trait the
//! host implements to surface such prompts in its UI.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// One elicitation request from a server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ElicitationRequest {
    /// Open a URL in the user's default browser (typical OAuth start).
    OpenUrl {
        /// Display message — shown to the user before redirect.
        message: String,
        url: String,
    },
    /// Yes/no confirmation. Returned as `Confirmed { yes: bool }`.
    Confirm {
        message: String,
        /// Optional default if the host UI dismisses without choice.
        #[serde(default)]
        default_yes: bool,
    },
    /// Free-form text input — the server may need an OTP, project
    /// name, etc. that the user is expected to type.
    PromptText {
        message: String,
        /// Whether the input should be treated as a secret (mask in UI).
        #[serde(default)]
        secret: bool,
    },
}

/// Handler-side response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ElicitationResponse {
    /// User dismissed (closed the prompt without acting). For
    /// `OpenUrl` this is "browser was opened". For `Confirm` this is
    /// the default. For `PromptText` this is no-input.
    Dismissed,
    /// User confirmed yes/no.
    Confirmed { yes: bool },
    /// User provided text input.
    Provided { text: String },
}

/// Trait the host implements. The lifecycle manager calls
/// `elicit(request)` whenever an MCP server initiates an elicitation.
#[async_trait]
pub trait ElicitationHandler: std::fmt::Debug + Send + Sync {
    async fn elicit(&self, request: ElicitationRequest) -> ElicitationResponse;
}

/// No-op handler — every elicitation gets [`ElicitationResponse::Dismissed`].
/// Useful in tests and headless servers.
#[derive(Debug, Clone, Default)]
pub struct NoopElicitationHandler;

#[async_trait]
impl ElicitationHandler for NoopElicitationHandler {
    async fn elicit(&self, _request: ElicitationRequest) -> ElicitationResponse {
        ElicitationResponse::Dismissed
    }
}

/// Convenience handler for headless / test runs:
///
/// - `Confirm`         → `Confirmed { yes: true }` (always agree).
/// - `PromptText`      → `Provided { text: "" }` (empty input).
/// - `OpenUrl`         → `Dismissed` — the host owns the actual
///   browser bridge; this handler does NOT spawn a real browser.
///   For tests this is sufficient because the OAuth code lives in
///   [`super::auth`] and the elicitation arm is only the prompt
///   surface.
#[derive(Debug, Clone, Default)]
pub struct AutoConfirmElicitationHandler;

#[async_trait]
impl ElicitationHandler for AutoConfirmElicitationHandler {
    async fn elicit(&self, request: ElicitationRequest) -> ElicitationResponse {
        match request {
            ElicitationRequest::OpenUrl { .. } => ElicitationResponse::Dismissed,
            // Auto-confirm: always say yes regardless of default.
            ElicitationRequest::Confirm { .. } => ElicitationResponse::Confirmed { yes: true },
            ElicitationRequest::PromptText { .. } => ElicitationResponse::Provided {
                text: String::new(),
            },
        }
    }
}

/// Boxed handler reference — what callers store in lifecycle config.
pub type ElicitationHandlerRef = Arc<dyn ElicitationHandler>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_dismisses_all() {
        let h = NoopElicitationHandler;
        for req in [
            ElicitationRequest::OpenUrl {
                message: "go".into(),
                url: "https://x".into(),
            },
            ElicitationRequest::Confirm {
                message: "ok?".into(),
                default_yes: false,
            },
            ElicitationRequest::PromptText {
                message: "name".into(),
                secret: false,
            },
        ] {
            assert_eq!(h.elicit(req).await, ElicitationResponse::Dismissed);
        }
    }

    #[tokio::test]
    async fn auto_confirm_says_yes_to_confirm() {
        let h = AutoConfirmElicitationHandler;
        let resp = h
            .elicit(ElicitationRequest::Confirm {
                message: "ok?".into(),
                default_yes: false,
            })
            .await;
        assert_eq!(resp, ElicitationResponse::Confirmed { yes: true });
    }

    #[tokio::test]
    async fn auto_confirm_provides_empty_text() {
        let h = AutoConfirmElicitationHandler;
        let resp = h
            .elicit(ElicitationRequest::PromptText {
                message: "name".into(),
                secret: false,
            })
            .await;
        assert_eq!(
            resp,
            ElicitationResponse::Provided {
                text: String::new()
            }
        );
    }

    #[test]
    fn elicitation_request_roundtrip() {
        for req in [
            ElicitationRequest::OpenUrl {
                message: "go".into(),
                url: "https://x".into(),
            },
            ElicitationRequest::Confirm {
                message: "ok?".into(),
                default_yes: true,
            },
            ElicitationRequest::PromptText {
                message: "x".into(),
                secret: true,
            },
        ] {
            let json = serde_json::to_string(&req).unwrap();
            let parsed: ElicitationRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, req);
        }
    }
}
