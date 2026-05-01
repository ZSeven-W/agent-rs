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
mod backends;
#[cfg(feature = "swarm")]
mod coordinator;
#[cfg(feature = "swarm")]
mod mailbox;
#[cfg(feature = "swarm")]
mod permission_sync;
#[cfg(feature = "swarm")]
mod sub_agent;
#[cfg(feature = "swarm")]
mod task;
#[cfg(feature = "swarm")]
mod team;

#[cfg(feature = "swarm")]
pub use backends::{
    Backend, BackendError, BackendHandle, InProcessBackend, Iterm2Backend, RunnerFn,
    SpawnSpec, TmuxBackend,
};

#[cfg(feature = "swarm")]
pub use coordinator::Coordinator;
#[cfg(feature = "swarm")]
pub use mailbox::{Mailbox, MailboxError, MailboxHeader, MailboxMessage, MAILBOX_SCHEMA_VERSION};
#[cfg(feature = "swarm")]
pub use permission_sync::{
    PendingRequest, PermissionSync, PermissionSyncError, ResolvedResponse,
};
#[cfg(feature = "swarm")]
pub use sub_agent::SubAgent;
#[cfg(feature = "swarm")]
pub use task::{SwarmTask, SwarmTaskPriority, SwarmTaskStatus};
#[cfg(feature = "swarm")]
pub use team::{MemberSpec, Team, TeamError};
