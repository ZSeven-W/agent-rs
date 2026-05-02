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
