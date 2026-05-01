//! Streaming event channel — the unified observability surface (Phase 1 / Task 1.2).
//!
//! Providers (Anthropic / OpenAI-compat / Ollama) translate their wire-level
//! protocols into [`Event`]s; downstream consumers (graphical chrome, TUI,
//! HTTP/SSE bridges, etc.) render the same event vocabulary. This is the
//! "event stream as primary observability channel" pattern lifted from
//! Claude Code's reference implementation (see
//! `notes/2026-05-01-claude-code-feature-reference.md`).

mod event;
mod tool_executor;

pub use event::{Event, EventStream, ResultData};
pub use tool_executor::{RequestedToolUse, ToolExecutor};
