//! Backend abstraction for spawning sub-agents (Phase 6 / Task 6.4).
//!
//! Three backends:
//! - [`InProcess`] — spawns a `tokio::task` in the current process.
//!   Real implementation; suitable for headless / library use.
//! - [`Tmux`] — opens a tmux pane and runs the agent there. **Stub.**
//!   Full implementation deferred (needs headless tmux available in
//!   CI; tracked in plan Task 6.4 follow-up).
//! - [`Iterm2`] — opens an iTerm2 tab on macOS via osascript. **Stub.**
//!   Same deferral; the trait + struct exist so consumers can plug in
//!   their own AppleScript launcher.

mod in_process;
mod iterm2;
mod tmux;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;
use tokio::task::JoinHandle;

use crate::abort::AbortController;

pub use in_process::InProcessBackend;
pub use iterm2::Iterm2Backend;
pub use tmux::TmuxBackend;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend not implemented: {0}")]
    NotImplemented(String),
    #[error("backend io: {0}")]
    Io(#[from] std::io::Error),
    #[error("backend: {0}")]
    Other(String),
}

/// One running sub-agent. The handle exposes the abort controller used
/// to cancel the runner; for in-process backends it also wraps the
/// JoinHandle.
#[derive(Debug)]
pub struct BackendHandle {
    pub agent_id: String,
    pub abort: AbortController,
    /// In-process: Some(handle). Out-of-process (tmux/iterm2): None.
    pub join: Option<JoinHandle<()>>,
}

impl BackendHandle {
    /// Cancel the sub-agent. Idempotent.
    pub fn stop(&self) {
        self.abort
            .abort_with_reason(format!("backend stop {}", self.agent_id));
    }

    /// For in-process backends, wait for the runner task to finish.
    /// For out-of-process backends, returns Ok(()) immediately
    /// (caller is expected to use the abort/heartbeat path).
    pub async fn join(self) -> Result<(), BackendError> {
        if let Some(handle) = self.join {
            handle
                .await
                .map_err(|e| BackendError::Other(format!("join: {e}")))?;
        }
        Ok(())
    }
}

/// Per-agent runner — what the backend should execute. Receives an
/// [`AbortController`] cloned from the handle so the runner can
/// observe cancellation.
pub type RunnerFn = Arc<
    dyn Fn(AbortController) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

/// Spec for spawning one agent. `cwd` is the working directory the
/// runner should resolve against.
#[derive(Clone)]
pub struct SpawnSpec {
    pub agent_id: String,
    pub cwd: PathBuf,
    pub runner: RunnerFn,
}

impl std::fmt::Debug for SpawnSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnSpec")
            .field("agent_id", &self.agent_id)
            .field("cwd", &self.cwd)
            .field("runner", &"<async closure>")
            .finish()
    }
}

#[async_trait]
pub trait Backend: Send + Sync + std::fmt::Debug {
    fn id(&self) -> &str;

    /// Spawn the agent. Returns a handle that lets the caller cancel
    /// or join the runner.
    async fn spawn(&self, spec: SpawnSpec) -> Result<BackendHandle, BackendError>;
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    fn ran_once() -> (RunnerFn, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let runner: RunnerFn = Arc::new(move |abort: AbortController| {
            let counter = counter_clone.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::SeqCst);
                let _ = abort; // unused
            })
        });
        (runner, counter)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_process_spawn_runs_runner() {
        let (runner, counter) = ran_once();
        let backend = InProcessBackend::new();
        let spec = SpawnSpec {
            agent_id: "alice".into(),
            cwd: std::env::temp_dir(),
            runner,
        };
        let handle = backend.spawn(spec).await.unwrap();
        assert_eq!(handle.agent_id, "alice");
        handle.join().await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_process_abort_observable_in_runner() {
        let observed_abort = Arc::new(AtomicUsize::new(0));
        let observed_clone = observed_abort.clone();
        let runner: RunnerFn = Arc::new(move |abort: AbortController| {
            let observed = observed_clone.clone();
            Box::pin(async move {
                abort.cancelled().await;
                observed.fetch_add(1, Ordering::SeqCst);
            })
        });
        let backend = InProcessBackend::new();
        let handle = backend
            .spawn(SpawnSpec {
                agent_id: "bob".into(),
                cwd: std::env::temp_dir(),
                runner,
            })
            .await
            .unwrap();
        // Give the spawn a moment to land on the runner.
        tokio::time::sleep(Duration::from_millis(20)).await;
        handle.stop();
        handle.join().await.unwrap();
        assert_eq!(observed_abort.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tmux_backend_returns_not_implemented() {
        let backend = TmuxBackend;
        let result = backend
            .spawn(SpawnSpec {
                agent_id: "x".into(),
                cwd: std::env::temp_dir(),
                runner: Arc::new(|_| Box::pin(async {})),
            })
            .await;
        assert!(matches!(result, Err(BackendError::NotImplemented(_))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn iterm2_backend_returns_not_implemented() {
        let backend = Iterm2Backend;
        let result = backend
            .spawn(SpawnSpec {
                agent_id: "x".into(),
                cwd: std::env::temp_dir(),
                runner: Arc::new(|_| Box::pin(async {})),
            })
            .await;
        assert!(matches!(result, Err(BackendError::NotImplemented(_))));
    }
}
