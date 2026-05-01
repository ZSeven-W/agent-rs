//! In-process backend — spawns a `tokio::task` in the current process.

use async_trait::async_trait;

use crate::abort::AbortController;

use super::{Backend, BackendError, BackendHandle, SpawnSpec};

#[derive(Debug, Default, Clone, Copy)]
pub struct InProcessBackend;

impl InProcessBackend {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Backend for InProcessBackend {
    fn id(&self) -> &str {
        "in_process"
    }

    async fn spawn(&self, spec: SpawnSpec) -> Result<BackendHandle, BackendError> {
        let abort = AbortController::new();
        let abort_for_runner = abort.clone();
        let runner = spec.runner.clone();
        let agent_id = spec.agent_id.clone();
        let join = tokio::spawn(async move {
            let fut = (runner)(abort_for_runner);
            fut.await;
        });
        let _ = spec.cwd; // cwd is informational for in-process; the runner closure decides
        Ok(BackendHandle {
            agent_id,
            abort,
            join: Some(join),
        })
    }
}
