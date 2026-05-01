//! Hook system — typed event registry with shell-script + Rust-closure
//! handlers (Phase 3 / Task 3.3).
//!
//! See `runner` for the [`HookHandler`] trait and the [`HookRunner`]
//! registry; see `event` for the [`HookEvent`] vocabulary (24+
//! variants ported from the Claude Code reference + the Zig
//! placeholder).

mod event;
mod runner;

pub use event::HookEvent;
pub use runner::{HookHandler, HookOutcome, HookRunner, RustHookHandler, ScriptHookHandler};
