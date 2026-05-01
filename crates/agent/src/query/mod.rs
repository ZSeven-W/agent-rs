//! Single-turn query engine (Phase 2 / Task 2.3).
//!
//! [`QueryEngine`] composes a [`Provider`](crate::provider::Provider), a
//! [`ToolRegistry`](crate::tool::ToolRegistry), and a
//! [`MessageStore`](crate::message::MessageStore) to run an LLM turn end
//! to end. Phase 2 ships only the happy path — tool dispatch, permission
//! evaluation, and hook firing land in Phase 3+.

mod engine;
mod loop_;

pub use engine::QueryEngine;
pub use loop_::{Phase, QueryLoop, QueryLoopBuilder, Transition};
