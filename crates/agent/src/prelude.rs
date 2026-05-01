//! Convenience re-exports.
//!
//! `use agent::prelude::*;` to bring in the most-commonly-used types.

pub use crate::error::AgentError;
pub use crate::message::{
    ContentBlock, Header, ImageSource, Message, MessageStore, ToolResultContent,
};
pub use crate::stream::{Event, EventStream, ResultData};
pub use crate::VERSION;
