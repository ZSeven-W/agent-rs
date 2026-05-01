//! Convenience re-exports.
//!
//! `use agent::prelude::*;` to bring in the most-commonly-used types.

pub use crate::abort::AbortController;
pub use crate::error::AgentError;
pub use crate::file_cache::FileStateCache;
pub use crate::hook::{HookEvent, HookHandler, HookOutcome, HookRunner, RustHookHandler};
pub use crate::message::{
    ContentBlock, Header, ImageSource, Message, MessageStore, ToolResultContent,
};
pub use crate::permission::{
    AsyncToolPermissionCheck, ExternalQueue, ExternalQueueError, ExternalQueueReceiver,
    ExternalRequest, PermissionManager,
};
#[cfg(feature = "anthropic")]
pub use crate::provider::AnthropicProvider;
pub use crate::provider::{Provider, ProviderCapabilities, StreamRequest, ThinkingConfig};
pub use crate::query::QueryEngine;
pub use crate::stream::{Event, EventStream, RequestedToolUse, ResultData, ToolExecutor};
pub use crate::tool::{Tool, ToolRegistry, ToolUseContext};
pub use crate::VERSION;
