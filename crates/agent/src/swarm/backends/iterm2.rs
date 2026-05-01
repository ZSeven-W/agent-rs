//! iTerm2 backend — STUB. macOS only. Real impl defers per plan Task
//! 6.4 follow-up (requires osascript-based AppleScript launcher and
//! agent-side heartbeat protocol).
//!
//! Real impl outline (for reference):
//! 1. Build an osascript program that opens a new iTerm2 tab in the
//!    target window, cds to `spec.cwd`, and runs the agent runner
//!    binary path + arguments.
//! 2. Use `tokio::process::Command::new("osascript").args(["-e", ...])`
//!    to invoke; capture the tab id from the script output.
//! 3. On stop(), send another osascript that closes the tab by id.

use async_trait::async_trait;

use super::{Backend, BackendError, BackendHandle, SpawnSpec};

#[derive(Debug, Default, Clone, Copy)]
pub struct Iterm2Backend;

#[async_trait]
impl Backend for Iterm2Backend {
    fn id(&self) -> &str {
        "iterm2"
    }

    async fn spawn(&self, _spec: SpawnSpec) -> Result<BackendHandle, BackendError> {
        Err(BackendError::NotImplemented(
            "iterm2 backend defers full implementation — see Phase 6 Task 6.4 follow-up".into(),
        ))
    }
}
