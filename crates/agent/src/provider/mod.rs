//! LLM provider abstraction (Phase 1 / Task 1.3).
//!
//! Three provider impls land in Phase 2 / Phase 5:
//! - **Anthropic** — hand-rolled (reqwest + eventsource-stream) per
//!   `notes/2026-05-01-sdk-maturity-research.md`. Phase 2.
//! - **OpenAI-compat** — `async-openai` crate. Phase 5. Covers OpenAI,
//!   DeepSeek, Moonshot, OpenRouter, LM Studio, Groq, etc.
//! - **Ollama** — `ollama-rs` crate. Phase 5.
//!
//! All providers translate their wire-level protocol into the same
//! [`crate::stream::Event`] vocabulary. Capabilities (tool_use, prompt
//! caching, extended thinking) are reported up-front via
//! [`ProviderCapabilities`] so the QueryEngine can adapt request shape.

mod types;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "openai")]
pub mod openai_compat;

#[cfg(feature = "openai")]
pub use openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider, OpenAiDialect};

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "ollama")]
pub use ollama::{OllamaConfig, OllamaProvider};

pub use types::{
    Provider, ProviderCapabilities, StreamRequest, ThinkingConfig, ToolChoice, ToolDefinition,
    ToolDefinitionError,
};

/// Shared HTTP client for streaming providers: bounded connect + idle-read
/// timeouts so a black-holed endpoint can't hang a turn forever, while
/// leaving the TOTAL request unbounded (SSE streams legitimately run for
/// minutes). Falls back to a default client if the builder ever fails.
#[cfg(any(feature = "anthropic", feature = "openai"))]
pub(crate) fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .read_timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_default()
}
