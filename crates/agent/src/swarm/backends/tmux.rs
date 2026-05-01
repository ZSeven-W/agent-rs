//! Tmux backend — STUB. Real impl defers per plan Task 6.4 follow-up
//! (needs headless tmux available in CI + a clean tracing protocol
//! to replay against the legacy Zig recordings).
//!
//! Real impl outline (for reference):
//! 1. `tokio::process::Command::new("tmux").args(["new-session", "-d",
//!    "-s", session_name])` to create a detached session.
//! 2. `tmux send-keys -t {session}.0 "{cwd_cd_command}; {agent_runner}"
//!    Enter` to start the agent in the pane.
//! 3. Wire a heartbeat file in the cwd that the agent updates so the
//!    coordinator can confirm liveness; cancellation deletes the
//!    session.

use async_trait::async_trait;

use super::{Backend, BackendError, BackendHandle, SpawnSpec};

#[derive(Debug, Default, Clone, Copy)]
pub struct TmuxBackend;

#[async_trait]
impl Backend for TmuxBackend {
    fn id(&self) -> &str {
        "tmux"
    }

    async fn spawn(&self, _spec: SpawnSpec) -> Result<BackendHandle, BackendError> {
        Err(BackendError::NotImplemented(
            "tmux backend defers full implementation — see Phase 6 Task 6.4 follow-up".into(),
        ))
    }
}
