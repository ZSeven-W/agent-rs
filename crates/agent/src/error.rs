//! Crate-wide error type.
//!
//! Phase 1 surface. Variants will be added in later phases (`#[non_exhaustive]`
//! lets us extend without breaking SemVer).

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentError {
    #[error("aborted: {0}")]
    Aborted(String),

    #[error("provider {provider}: {message}")]
    Provider { provider: String, message: String },

    #[error("invalid message: {0}")]
    InvalidMessage(String),

    #[error("duplicate uuid: {0}")]
    DuplicateUuid(uuid::Uuid),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl AgentError {
    /// Convenience constructor for the `Other` catch-all.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    /// Convenience constructor for the `Provider` variant.
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
        }
    }
}
