//! Generic coding tool pack for the agent-rs runtime.
//!
//! Hosts that build coding agents (Zode, third-party CLI agents,
//! IDE assistants) typically want the same primitives: read a
//! file, write a file, edit a string, run a shell command, grep
//! for a pattern, glob for paths. This crate ships those tools as
//! `agent::tool::Tool` impls so any host can wire them up by
//! calling [`register_default`] (or a per-tool builder) on their
//! `ToolRegistry`.
//!
//! # Why a separate crate
//!
//! The core `agent` crate stays product-agnostic and slim: it
//! defines the `Tool` trait + registry, but ships zero concrete
//! tool implementations. Concrete tools belong here because they
//! pull non-trivial deps:
//!
//! - `regex` for [`grep`] (~5 MB compiled).
//! - `ignore` for [`glob`] gitignore-aware traversal.
//! - `shell-words` for safe argv splitting in [`bash`].
//! - `reqwest` for [`web_fetch`].
//!
//! Each is feature-gated so a host that only wants file CRUD
//! doesn't pay for the others. `default = ["fs", "search"]` covers
//! the common "read-mostly" case.
//!
//! # Safety classification
//!
//! Every tool overrides `Tool::safety_class` per the
//! [`agent::tool::SafetyClass`] lattice:
//!
//! | Tool      | Class         | Notes                                  |
//! |-----------|---------------|----------------------------------------|
//! | FileRead  | `ReadOnly`    | No side effects.                       |
//! | ListDir   | `ReadOnly`    |                                        |
//! | Grep      | `ReadOnly`    |                                        |
//! | Glob      | `ReadOnly`    |                                        |
//! | WebFetch  | `ReadOnly`    | Network read, no mutation.             |
//! | TodoWrite | `Mutating`    | Mutates in-memory state only.          |
//! | FileWrite | `Mutating`    | Creates / overwrites.                  |
//! | FileEdit  | `Mutating`    | In-place edit.                         |
//! | Mkdir     | `Mutating`    |                                        |
//! | Move      | `Mutating`    | File system mutation.                  |
//! | Bash      | `Mutating`    | Caller can gate Destructive shell      |
//! |           |               | shapes via PermissionMatcher.          |
//! | Remove    | `Destructive` | Irreversible.                          |
//!
//! Hosts compose these with [`agent::permission::PermissionMatcher`]
//! rules to add path / command allowlists per their threat model.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod discovery;
pub mod policy;

#[cfg(feature = "fs")]
pub mod fs;

#[cfg(feature = "search")]
pub mod search;

pub use discovery::ToolSearchTool;
pub use policy::{PolicyError, WorkspacePolicy};

#[cfg(feature = "fs")]
pub use fs::{
    FileEditTool, FileReadTool, FileWriteTool, ListDirTool, MkdirTool, MoveTool, RemoveTool,
};

#[cfg(feature = "search")]
pub use search::{GlobTool, GrepTool};

use std::sync::Arc;

use agent::tool::{Tool, ToolRegistry};

/// Register every enabled tool against a `ToolRegistry` using a
/// shared [`WorkspacePolicy`]. Convenience for hosts that want the
/// "everything I asked for" bundle without naming each tool.
///
/// Tools are inserted under their canonical names (`FileRead`,
/// `FileWrite`, …). Re-registering replaces.
pub fn register_default(registry: &mut ToolRegistry, policy: Arc<WorkspacePolicy>) {
    #[cfg(feature = "fs")]
    {
        registry.register(Arc::new(FileReadTool::new(policy.clone())) as Arc<dyn Tool>);
        registry.register(Arc::new(FileWriteTool::new(policy.clone())) as Arc<dyn Tool>);
        registry.register(Arc::new(FileEditTool::new(policy.clone())) as Arc<dyn Tool>);
        registry.register(Arc::new(ListDirTool::new(policy.clone())) as Arc<dyn Tool>);
        registry.register(Arc::new(MkdirTool::new(policy.clone())) as Arc<dyn Tool>);
        registry.register(Arc::new(MoveTool::new(policy.clone())) as Arc<dyn Tool>);
        registry.register(Arc::new(RemoveTool::new(policy.clone())) as Arc<dyn Tool>);
    }
    #[cfg(feature = "search")]
    {
        registry.register(Arc::new(GrepTool::new(policy.clone())) as Arc<dyn Tool>);
        registry.register(Arc::new(GlobTool::new(policy.clone())) as Arc<dyn Tool>);
    }
    let _ = policy; // suppress unused when no features enabled
    let _ = registry;
}
