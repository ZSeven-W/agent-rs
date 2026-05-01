//! HookRunner + HookHandler trait + Rust closure / shell script handlers.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use super::event::HookEvent;
use crate::error::AgentError;

/// Outcome of running a single hook against a single event.
///
/// Mirrors the Claude Code shell-script hook protocol:
/// - exit 0 → [`HookOutcome::Ok`] (proceed normally)
/// - exit 2 → [`HookOutcome::Block`] (abort the action that triggered the hook)
/// - other exit code → [`HookOutcome::Warn`] (log + proceed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookOutcome {
    Ok,
    Block,
    Warn(i32),
}

impl HookOutcome {
    pub fn from_exit_code(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            2 => Self::Block,
            other => Self::Warn(other),
        }
    }
}

/// One hook implementation. Implementors come from either user Rust
/// closures or external shell scripts.
#[async_trait]
pub trait HookHandler: Send + Sync + std::fmt::Debug {
    async fn handle(&self, event: &HookEvent) -> HookOutcome;
}

/// Adapter that wraps a synchronous Rust closure into a [`HookHandler`].
pub struct RustHookHandler {
    name: String,
    func: Box<dyn Fn(&HookEvent) -> HookOutcome + Send + Sync>,
}

impl RustHookHandler {
    pub fn new(
        name: impl Into<String>,
        f: impl Fn(&HookEvent) -> HookOutcome + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            func: Box::new(f),
        }
    }
}

impl std::fmt::Debug for RustHookHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustHookHandler")
            .field("name", &self.name)
            .field("func", &"<closure>")
            .finish()
    }
}

#[async_trait]
impl HookHandler for RustHookHandler {
    async fn handle(&self, event: &HookEvent) -> HookOutcome {
        (self.func)(event)
    }
}

/// Adapter that runs an external shell script as a hook. The event is
/// serialized to JSON and written to the script's stdin; the script's
/// exit code drives the [`HookOutcome`].
#[derive(Debug, Clone)]
pub struct ScriptHookHandler {
    pub script: PathBuf,
}

impl ScriptHookHandler {
    pub fn new(script: impl Into<PathBuf>) -> Self {
        Self {
            script: script.into(),
        }
    }

    async fn run(&self, event: &HookEvent) -> Result<i32, AgentError> {
        use tokio::io::AsyncWriteExt;
        use tokio::process::Command;

        let json = serde_json::to_vec(event)?;
        let mut child = Command::new(&self.script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(&json).await?;
            stdin.shutdown().await.ok();
        }
        let status = child.wait().await?;
        Ok(status.code().unwrap_or(-1))
    }
}

#[async_trait]
impl HookHandler for ScriptHookHandler {
    async fn handle(&self, event: &HookEvent) -> HookOutcome {
        match self.run(event).await {
            Ok(code) => HookOutcome::from_exit_code(code),
            Err(err) => {
                tracing::warn!(
                    script = %self.script.display(),
                    event = event.name(),
                    error = %err,
                    "hook script failed to execute; treating as warn",
                );
                HookOutcome::Warn(-1)
            }
        }
    }
}

/// Registry of hooks. `run(event)` invokes every registered handler in
/// insertion order; the first `Block` outcome short-circuits and returns
/// `Block`. Non-zero non-block exits log a warning and continue.
#[derive(Debug, Default, Clone)]
pub struct HookRunner {
    handlers: Vec<Arc<dyn HookHandler>>,
}

impl HookRunner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, handler: Arc<dyn HookHandler>) {
        self.handlers.push(handler);
    }

    pub fn with(mut self, handler: Arc<dyn HookHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Fire `event` through every registered handler in order. Returns
    /// `Block` as soon as any handler blocks; `Warn(code)` if any handler
    /// warned but none blocked; `Ok` otherwise.
    pub async fn run(&self, event: &HookEvent) -> HookOutcome {
        let mut last_warn: Option<i32> = None;
        for h in &self.handlers {
            match h.handle(event).await {
                HookOutcome::Block => return HookOutcome::Block,
                HookOutcome::Warn(code) => {
                    tracing::warn!(
                        event = event.name(),
                        exit_code = code,
                        "hook returned non-zero non-block",
                    );
                    last_warn = Some(code);
                }
                HookOutcome::Ok => {}
            }
        }
        match last_warn {
            Some(code) => HookOutcome::Warn(code),
            None => HookOutcome::Ok,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    fn rust_hook(name: &str, outcome: HookOutcome) -> Arc<RustHookHandler> {
        let outcome_clone = outcome;
        Arc::new(RustHookHandler::new(name, move |_| outcome_clone))
    }

    #[test]
    fn outcome_from_exit_code_mapping() {
        assert_eq!(HookOutcome::from_exit_code(0), HookOutcome::Ok);
        assert_eq!(HookOutcome::from_exit_code(2), HookOutcome::Block);
        assert_eq!(HookOutcome::from_exit_code(7), HookOutcome::Warn(7));
        assert_eq!(HookOutcome::from_exit_code(-1), HookOutcome::Warn(-1));
    }

    #[tokio::test]
    async fn empty_runner_returns_ok() {
        let r = HookRunner::new();
        let event = HookEvent::OnSessionStart;
        assert_eq!(r.run(&event).await, HookOutcome::Ok);
    }

    #[tokio::test]
    async fn block_short_circuits() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let counting = Arc::new(RustHookHandler::new("counter", move |_| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            HookOutcome::Ok
        }));

        let r = HookRunner::new()
            .with(rust_hook("blocker", HookOutcome::Block))
            .with(counting);

        let event = HookEvent::OnSessionStart;
        assert_eq!(r.run(&event).await, HookOutcome::Block);
        // The counting hook should NOT have run because the blocker came first.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn warn_continues_and_returns_warn() {
        let r = HookRunner::new()
            .with(rust_hook("a", HookOutcome::Ok))
            .with(rust_hook("b", HookOutcome::Warn(7)))
            .with(rust_hook("c", HookOutcome::Ok));
        let event = HookEvent::OnSessionStart;
        assert_eq!(r.run(&event).await, HookOutcome::Warn(7));
    }

    #[tokio::test]
    async fn handler_receives_event_with_name() {
        let captured = Arc::new(std::sync::Mutex::new(String::new()));
        let captured_c = captured.clone();
        let h = Arc::new(RustHookHandler::new("capture", move |e| {
            *captured_c.lock().unwrap() = e.name().to_string();
            HookOutcome::Ok
        }));
        let r = HookRunner::new().with(h);
        let event = HookEvent::BeforeToolUse {
            tool: "Bash".into(),
            input: serde_json::json!({}),
        };
        r.run(&event).await;
        assert_eq!(*captured.lock().unwrap(), "before_tool_use");
    }
}
