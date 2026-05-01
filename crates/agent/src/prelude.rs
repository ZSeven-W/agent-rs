//! Convenience re-exports.
//!
//! `use agent::prelude::*;` to bring in the most-commonly-used types.

pub use crate::abort::AbortController;
pub use crate::error::AgentError;
pub use crate::file_cache::FileStateCache;
pub use crate::message::{
    ContentBlock, Header, ImageSource, Message, MessageStore, ToolResultContent,
};
pub use crate::provider::{Provider, ProviderCapabilities, StreamRequest, ThinkingConfig};
pub use crate::stream::{Event, EventStream, ResultData};
pub use crate::VERSION;
