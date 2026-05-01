//! Sub-agent identity + mailbox binding (Phase 6 / Task 6.3).
//!
//! A [`SubAgent`] is the metadata + messaging surface for one
//! participant in a [`super::Team`]. Each sub-agent owns:
//!
//! - a stable `id` (used as the inbox file basename),
//! - a `role` string (free-text — "leader", "qa-worker", "writer"),
//! - a [`super::Mailbox`] for receiving messages,
//! - a [`super::permission_sync::PermissionSync`] handle (shared with
//!   teammates) for cross-agent approval,
//! - an optional `runner_handle` slot that the [`super::Coordinator`]
//!   fills with the spawned task's abort handle (set when started).
//!
//! Deliberately **does not** own a [`crate::query::QueryEngine`] —
//! the QueryEngine is product-supplied (OpenPencil / Zode each wire
//! their own provider/tools). The Coordinator drives the QueryEngine
//! by passing user messages to it and forwarding events.

use std::sync::Arc;

use crate::abort::AbortController;

use super::mailbox::Mailbox;
use super::permission_sync::PermissionSync;

#[derive(Debug, Clone)]
pub struct SubAgent {
    pub id: String,
    pub role: String,
    pub mailbox: Mailbox,
    pub permission_sync: Arc<PermissionSync>,
    /// Filled by the Coordinator when this sub-agent's runner task
    /// starts; cancelling triggers abort.
    pub abort: AbortController,
}

impl SubAgent {
    pub fn new(
        id: impl Into<String>,
        role: impl Into<String>,
        mailbox: Mailbox,
        permission_sync: Arc<PermissionSync>,
    ) -> Self {
        Self {
            id: id.into(),
            role: role.into(),
            mailbox,
            permission_sync,
            abort: AbortController::new(),
        }
    }

    /// Cancels the sub-agent's runner. Idempotent.
    pub fn stop(&self) {
        self.abort.abort_with_reason(format!("stop {}", self.id));
    }

    pub fn is_stopped(&self) -> bool {
        self.abort.is_aborted()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::tempdir;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn new_subagent_has_role_and_running_state() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_agent(dir.path(), "alice").await.unwrap();
        let ps = Arc::new(PermissionSync::new(dir.path()).await.unwrap());
        let agent = SubAgent::new("alice", "qa-worker", mb, ps);
        assert_eq!(agent.id, "alice");
        assert_eq!(agent.role, "qa-worker");
        assert!(!agent.is_stopped());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stop_is_idempotent() {
        let dir = tempdir().unwrap();
        let mb = Mailbox::for_agent(dir.path(), "alice").await.unwrap();
        let ps = Arc::new(PermissionSync::new(dir.path()).await.unwrap());
        let agent = SubAgent::new("alice", "qa", mb, ps);
        agent.stop();
        agent.stop(); // idempotent
        assert!(agent.is_stopped());
        assert_eq!(agent.abort.reason().as_deref(), Some("stop alice"));
    }
}
