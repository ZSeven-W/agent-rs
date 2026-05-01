//! Convenience re-exports.
//!
//! `use agent::prelude::*;` to bring in the most-commonly-used types.

pub use crate::abort::AbortController;
pub use crate::error::AgentError;
pub use crate::file_cache::FileStateCache;
pub use crate::hook::HookRunner;
pub use crate::message::{
    ContentBlock, Header, ImageSource, Message, MessageStore, ToolResultContent,
};
pub use crate::permission::PermissionManager;
pub use crate::provider::{Provider, ProviderCapabilities, StreamRequest, ThinkingConfig};
#[cfg(feature = "anthropic")]
pub use crate::provider::AnthropicProvider;
pub use crate::query::QueryEngine;
pub use crate::stream::{Event, EventStream, ResultData};
pub use crate::tool::{Tool, ToolRegistry, ToolUseContext};
pub use crate::VERSION;
