//! Application state store (Tier 2 / claude-code parity).
//!
//! Mirrors `services/state/AppStateStore.ts`. The host (OpenPencil,
//! Zode) needs a single source of truth for transient runtime state
//! that gets read by every UI surface and the agent loop:
//!
//! - Active session id + working directory.
//! - Agent mode (default / accept-edits / bypass / plan).
//! - Currently-running task / tool, if any.
//! - Configuration snapshot (model, effort, output config).
//! - Queued user messages waiting to be sent.
//! - Last error surface / dismissable banner.
//!
//! The store is a typed snapshot + subscriber pattern: writers
//! produce a new immutable [`AppState`], subscribers see the change
//! via a tokio broadcast channel. Selectors let consumers subscribe
//! to a slice of state (model name, mode, etc.) without re-rendering
//! on every unrelated change.
//!
//! Does NOT persist — host's responsibility to write to JSONL or
//! its own settings file. This module is purely the in-process
//! coordination point.

pub mod selector;
pub mod store;

pub use selector::{Selector, SelectorChange};
pub use store::{AgentMode, AppState, AppStateStore, ConfigSnapshot, QueuedMessage};
