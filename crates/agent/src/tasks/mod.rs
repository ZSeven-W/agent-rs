//! Task system (Tier 3 / claude-code parity).
//!
//! Mirrors `services/tasks/` from claude-code. Distinct from
//! [`crate::swarm::task`] (which models a single execution attempt
//! by a sub-agent worker) — this module is the higher-level
//! planning + state-tracking surface a host UI uses to render the
//! task list, persist it across sessions, and gate dependencies.
//!
//! Task lifecycle:
//!
//! ```text
//! Pending → InProgress → Completed
//!     │         │
//!     ▼         ▼
//!  Blocked   Canceled
//! ```
//!
//! Dependencies: a task in `blocked_by: [t1, t2]` enters
//! `InProgress` only when both `t1` and `t2` are `Completed`.

pub mod graph;
pub mod task;

pub use graph::{TaskGraph, TaskGraphError};
pub use task::{PlannedTask, TaskId, TaskStatus};
