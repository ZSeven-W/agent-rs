//! Swarm / Teams primitives (Phase 6).
//!
//! Multi-agent collaboration on the same machine: a "team" is a set of
//! [`SubAgent`] instances that exchange messages via per-agent
//! [`Mailbox`] files (file-locked JSONL) and request human approval via
//! [`permission_sync`]. Backends (in-process / tmux / iterm2) decide
//! how each agent process is spawned.
//!
//! Storage root: `~/.openpencil/agent/teams/{team_id}/` per the Phase 0
//! migration decision (see `docs/migration.md`). The Rust agent does
//! NOT touch the legacy Zig `~/.claude/teams/...` paths.
//!
//! Feature-gated behind `swarm`.

#![allow(clippy::result_large_err)]

#[cfg(feature = "swarm")]
mod mailbox;

#[cfg(feature = "swarm")]
pub use mailbox::{Mailbox, MailboxError, MailboxHeader, MailboxMessage, MAILBOX_SCHEMA_VERSION};
