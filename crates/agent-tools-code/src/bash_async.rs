//! Background shells: `BashRun` + `BashOutput` + `KillShell`.
//!
//! Mirrors Claude Code's split between one-shot Bash (already
//! exposed as [`crate::BashTool`] under `feature = "shell"`) and
//! the async shell trio: spawn → poll output later → optionally
//! kill. The model uses these for long-running commands (build /
//! watcher / tail) where blocking the tool dispatcher for minutes
//! is a non-starter.
//!
//! All three tools share a single [`BashSessionRegistry`]
//! (`Arc<RwLock<HashMap<id, BashSession>>>`). Hosts construct one
//! registry per session and pass it to all three tools, so the
//! model's `BashRun` returns a `shell_id` that subsequent
//! `BashOutput` / `KillShell` calls can address.
//!
//! Output capture follows the same ring-buffer + tail-preservation
//! shape as [`crate::BashTool`]: each stream gets a
//! `VecDeque<u8>` cap, a non-blocking reader thread drains
//! stdout/stderr into it, and `BashOutput` returns whatever has
//! accumulated since the last poll. Old output is NOT replayed —
//! `BashOutput` is read-then-clear, so the model's window stays
//! bounded even on a chatty long-running build.
//!
//! Process hygiene matches `BashTool`: on Unix the child is started
//! in a new session, so it has no controlling TTY and its pid is also
//! the process-group id that `KillShell` can `/bin/kill -9 -<pgid>`.
//! On Windows the supervisor owns the `Child` through termination and uses
//! `taskkill /T` as a best-effort descendant fallback.
//!
//! Hosts that don't enable both this and the `shell` feature get
//! the BashTool's existing one-shot semantics. The async trio is
//! orthogonal — `bash-async` doesn't depend on `shell`.

use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::sync::{watch, Mutex as AsyncMutex, RwLock};
use tokio::task::JoinHandle;
use tokio::time::timeout;

use crate::policy::{PolicyError, WorkspacePolicy};

/// Hard cap on tail-buffer size per stream. Same as `BashTool`.
const PER_STREAM_CAP: usize = 1024 * 1024;
/// Hard cap on the number of concurrent background sessions per
/// registry. Stops a runaway model from spawning thousands of
/// processes.
const MAX_SESSIONS_PER_REGISTRY: usize = 32;
/// Default `BashOutput` poll wait. The tool returns whatever has
/// accumulated immediately; this is the maximum extra wait when
/// the buffer is currently empty (so the model can poll once and
/// catch new output without spinning).
const DEFAULT_OUTPUT_WAIT_MS: u64 = 0;
const MAX_OUTPUT_WAIT_MS: u64 = 10_000;
const TERMINATION_GRACE: Duration = Duration::from_secs(2);

#[derive(Debug)]
struct BashSession {
    /// Original command (for logging / introspection).
    command: String,
    /// PID captured at spawn for the process-group kill fallback.
    pid: Option<u32>,
    /// Tail buffers for stdout/stderr + truncation flags. Bounded
    /// at `PER_STREAM_CAP` regardless of stream length.
    stdout: Arc<AsyncMutex<TailBuffer>>,
    stderr: Arc<AsyncMutex<TailBuffer>>,
    /// Latest exit status; `None` while the process is still
    /// running.
    exit: Arc<AsyncMutex<Option<std::process::ExitStatus>>>,
    running: Arc<AtomicBool>,
    terminate: watch::Sender<bool>,
    supervisor: Option<JoinHandle<()>>,
}

impl BashSession {
    fn request_stop(&self) {
        if self.running.load(Ordering::Acquire) {
            self.terminate.send_replace(true);
            kill_process_tree(self.pid);
        }
    }

    async fn stop(mut self) {
        self.request_stop();
        if let Some(mut supervisor) = self.supervisor.take() {
            if timeout(TERMINATION_GRACE, &mut supervisor).await.is_err() {
                supervisor.abort();
                let _ = supervisor.await;
            }
        }
    }
}

impl Drop for BashSession {
    fn drop(&mut self) {
        self.request_stop();
        if let Some(supervisor) = self.supervisor.take() {
            // Aborting drops the owned Child and process-tree guard inside the
            // supervisor. This is the synchronous registry-drop fallback.
            supervisor.abort();
        }
    }
}

#[derive(Debug)]
struct ProcessTreeGuard {
    pid: Option<u32>,
    armed: bool,
}

impl ProcessTreeGuard {
    fn new(pid: Option<u32>) -> Self {
        Self { pid, armed: true }
    }

    fn kill(&self) {
        kill_process_tree(self.pid);
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ProcessTreeGuard {
    fn drop(&mut self) {
        if self.armed {
            self.kill();
        }
    }
}

#[cfg(unix)]
fn kill_process_tree(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    let group = format!("-{pid}");
    let _ = std::process::Command::new("/bin/kill")
        .args(["-KILL", &group])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(windows)]
fn kill_process_tree(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(not(any(unix, windows)))]
fn kill_process_tree(_pid: Option<u32>) {}

#[derive(Debug, Default)]
struct TailBuffer {
    bytes: VecDeque<u8>,
    truncated: bool,
}

impl TailBuffer {
    fn push_chunk(&mut self, chunk: &[u8]) {
        for &b in chunk {
            if self.bytes.len() == PER_STREAM_CAP {
                self.bytes.pop_front();
                self.truncated = true;
            }
            self.bytes.push_back(b);
        }
    }
    /// Drain everything currently held; returns the buffered bytes
    /// and whether the buffer had truncated since the last drain.
    fn drain(&mut self) -> (Vec<u8>, bool) {
        let bytes: Vec<u8> = self.bytes.drain(..).collect();
        let truncated = std::mem::take(&mut self.truncated);
        (bytes, truncated)
    }
}

/// Shared registry of running background shells. Cheap to clone
/// (`Arc`); hosts hand the same handle to `BashRun` /
/// `BashOutput` / `KillShell`.
#[derive(Debug, Clone, Default)]
pub struct BashSessionRegistry {
    inner: Arc<RwLock<HashMap<String, BashSession>>>,
}

impl BashSessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

fn policy_to_agent_err(e: PolicyError) -> AgentError {
    AgentError::other(format!("policy: {e}"))
}

// =====================================================================
// BashRun
// =====================================================================

/// Spawn a long-running shell command and return immediately with
/// a `shell_id` the model can poll via `BashOutput` / kill via
/// `KillShell`.
#[derive(Debug)]
pub struct BashRunTool {
    policy: Arc<WorkspacePolicy>,
    registry: BashSessionRegistry,
}

impl BashRunTool {
    pub fn new(policy: Arc<WorkspacePolicy>, registry: BashSessionRegistry) -> Self {
        Self { policy, registry }
    }
}

#[derive(Debug, Deserialize)]
struct BashRunInput {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
}

#[async_trait]
impl Tool for BashRunTool {
    fn name(&self) -> &str {
        "BashRun"
    }
    fn description(&self) -> &str {
        "Spawn a long-running shell command in the background. Returns a `shell_id` to poll via BashOutput and kill via KillShell."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command. Runs via /bin/sh -c (Unix) / cmd /C (Windows)."},
                "cwd": {"type": "string"}
            },
            "required": ["command"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(&self, ctx: &ToolUseContext, input: Value) -> Result<Value, AgentError> {
        let parsed: BashRunInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("BashRun invalid input: {e}")))?;
        if parsed.command.trim().is_empty() {
            return Err(AgentError::other("BashRun command must be non-empty"));
        }

        let cwd = match parsed.cwd.as_deref() {
            Some(p) => self.policy.resolve(p, true).map_err(policy_to_agent_err)?,
            None => self.policy.cwd.clone(),
        };

        // Acquire the only awaitable resource before spawning. From spawn
        // through insertion there are no suspension points, so cancellation
        // cannot strand an unregistered waiter or process.
        let abort = ctx.abort.clone();
        let mut sessions = tokio::select! {
            biased;
            _ = abort.cancelled() => {
                return Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "aborted".into()),
                ));
            }
            sessions = self.registry.inner.write() => sessions,
        };
        if sessions.len() >= MAX_SESSIONS_PER_REGISTRY {
            return Err(AgentError::other(format!(
                "BashRun: registry capped at {MAX_SESSIONS_PER_REGISTRY} concurrent sessions; kill an existing one first"
            )));
        }
        abort.mark_side_effect_risk();

        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(&parsed.command);
            c
        } else {
            let mut c = Command::new("/bin/sh");
            c.arg("-c").arg(&parsed.command);
            c
        };
        cmd.current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        crate::process::detach_from_controlling_tty(&mut cmd);

        let mut child = cmd
            .spawn()
            .map_err(|e| AgentError::other(format!("BashRun spawn failed: {e}")))?;
        let pid = child.id();
        let mut registration_guard = ProcessTreeGuard::new(pid);

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::other("BashRun missing stdout pipe"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AgentError::other("BashRun missing stderr pipe"))?;

        let session_id = format!("bash_{}", random_id());
        let stdout_buf: Arc<AsyncMutex<TailBuffer>> =
            Arc::new(AsyncMutex::new(TailBuffer::default()));
        let stderr_buf: Arc<AsyncMutex<TailBuffer>> =
            Arc::new(AsyncMutex::new(TailBuffer::default()));
        let exit_slot: Arc<AsyncMutex<Option<std::process::ExitStatus>>> =
            Arc::new(AsyncMutex::new(None));
        let running = Arc::new(AtomicBool::new(true));
        let (terminate, terminate_rx) = watch::channel(false);

        // Reader tasks — drain each pipe into its tail buffer.
        spawn_reader(stdout, stdout_buf.clone());
        spawn_reader(stderr, stderr_buf.clone());

        let supervisor =
            spawn_supervisor(child, pid, terminate_rx, exit_slot.clone(), running.clone());

        let session = BashSession {
            command: parsed.command.clone(),
            pid,
            stdout: stdout_buf,
            stderr: stderr_buf,
            exit: exit_slot,
            running,
            terminate,
            supervisor: Some(supervisor),
        };

        sessions.insert(session_id.clone(), session);
        registration_guard.disarm();
        // A registered background shell intentionally survives this tool
        // future. Scheduled turns must never be retried or recur while that
        // detached external work remains unresolved.
        abort.mark_unresolved_external_work();

        Ok(json!({
            "shell_id": session_id,
            "command": parsed.command,
            "cwd": cwd.display().to_string(),
            "pid": pid,
        }))
    }
}

fn spawn_supervisor(
    mut child: Child,
    pid: Option<u32>,
    mut terminate: watch::Receiver<bool>,
    exit: Arc<AsyncMutex<Option<std::process::ExitStatus>>>,
    running: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut process_guard = ProcessTreeGuard::new(pid);
        let status = tokio::select! {
            status = child.wait() => status.ok(),
            _ = termination_requested(&mut terminate) => {
                process_guard.kill();
                let _ = child.kill().await;
                timeout(TERMINATION_GRACE, child.wait()).await.ok().and_then(Result::ok)
            }
        };

        // Publish terminal state before disarming the drop guard. A concurrent
        // registry drop then either sees `running` and kills, or relies on this
        // still-armed guard to finish the same cleanup.
        running.store(false, Ordering::Release);
        process_guard.kill();
        process_guard.disarm();
        if let Some(status) = status {
            *exit.lock().await = Some(status);
        }
    })
}

async fn termination_requested(terminate: &mut watch::Receiver<bool>) {
    loop {
        if *terminate.borrow_and_update() {
            return;
        }
        if terminate.changed().await.is_err() {
            return;
        }
    }
}

fn spawn_reader<R>(mut reader: R, buf: Arc<AsyncMutex<TailBuffer>>)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut tmp = [0u8; 16 * 1024];
        loop {
            match reader.read(&mut tmp).await {
                Ok(0) => break,
                Ok(n) => {
                    buf.lock().await.push_chunk(&tmp[..n]);
                }
                Err(_) => break,
            }
        }
    });
}

fn random_id() -> String {
    // Nanos-since-epoch + a process-global counter so back-to-back
    // calls (e.g. a test that inserts 32 ids in a tight loop) can't
    // collide on systems where the clock resolution is coarser than
    // one call's worth of work. Without the counter, two `now()`
    // calls within the same nanosecond produce the same id and the
    // HashMap dedupes them silently.
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let s = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{n:x}_{s:x}")
}

// =====================================================================
// BashOutput
// =====================================================================

/// Drain accumulated output for a previously-started shell.
/// Read-then-clear: subsequent calls only see new output.
#[derive(Debug)]
pub struct BashOutputTool {
    registry: BashSessionRegistry,
    compress_output: bool,
}

impl BashOutputTool {
    pub fn new(registry: BashSessionRegistry) -> Self {
        Self::with_compress_output(registry, true)
    }

    pub fn with_compress_output(registry: BashSessionRegistry, compress_output: bool) -> Self {
        Self {
            registry,
            compress_output,
        }
    }
}

#[derive(Debug, Deserialize)]
struct BashOutputInput {
    shell_id: String,
    /// If the buffer is empty, wait up to this many milliseconds
    /// for new output before returning. Default 0 (immediate
    /// return). Capped at 10 s.
    #[serde(default)]
    wait_ms: Option<u64>,
}

#[async_trait]
impl Tool for BashOutputTool {
    fn name(&self) -> &str {
        "BashOutput"
    }
    fn description(&self) -> &str {
        "Drain stdout/stderr accumulated since the last poll for a background shell. Returns {stdout, stderr, running, exit_code, signal} and clears the buffer."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "shell_id": {"type": "string"},
                "wait_ms": {"type": "integer", "minimum": 0, "maximum": MAX_OUTPUT_WAIT_MS}
            },
            "required": ["shell_id"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    async fn call(&self, ctx: &ToolUseContext, input: Value) -> Result<Value, AgentError> {
        let parsed: BashOutputInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("BashOutput invalid input: {e}")))?;
        let wait_ms = parsed
            .wait_ms
            .unwrap_or(DEFAULT_OUTPUT_WAIT_MS)
            .min(MAX_OUTPUT_WAIT_MS);

        let session = {
            let g = self.registry.inner.read().await;
            let s = g.get(&parsed.shell_id).ok_or_else(|| {
                AgentError::other(format!(
                    "BashOutput: no shell with id '{}'",
                    parsed.shell_id
                ))
            })?;
            // Clone only observable state; the supervisor remains owned by
            // the registry entry.
            (
                s.stdout.clone(),
                s.stderr.clone(),
                s.exit.clone(),
                s.running.clone(),
                s.command.clone(),
            )
        };
        let (stdout_buf, stderr_buf, exit_slot, running_flag, command) = session;

        // First pass — drain whatever's there now.
        let mut stdout_bytes;
        let mut stdout_trunc;
        let mut stderr_bytes;
        let mut stderr_trunc;
        {
            let (b, t) = stdout_buf.lock().await.drain();
            stdout_bytes = b;
            stdout_trunc = t;
        }
        {
            let (b, t) = stderr_buf.lock().await.drain();
            stderr_bytes = b;
            stderr_trunc = t;
        }

        // If both empty AND the process is still running, wait.
        if stdout_bytes.is_empty() && stderr_bytes.is_empty() && wait_ms > 0 {
            let exit_now = exit_slot.lock().await.is_some();
            if !exit_now {
                let abort = ctx.abort.clone();
                let wait_dur = Duration::from_millis(wait_ms);
                tokio::select! {
                    biased;
                    _ = abort.cancelled() => {
                        return Err(AgentError::Aborted(
                            abort.reason().unwrap_or_else(|| "aborted".into()),
                        ));
                    }
                    _ = poll_until_data(&stdout_buf, &stderr_buf, &exit_slot, wait_dur) => {}
                }
                {
                    let (b, t) = stdout_buf.lock().await.drain();
                    stdout_bytes.extend(b);
                    stdout_trunc |= t;
                }
                {
                    let (b, t) = stderr_buf.lock().await.drain();
                    stderr_bytes.extend(b);
                    stderr_trunc |= t;
                }
            }
        }

        let exit = *exit_slot.lock().await;
        let running = running_flag.load(Ordering::Acquire);

        let stdout_raw = String::from_utf8_lossy(&stdout_bytes).into_owned();
        let stderr_raw = String::from_utf8_lossy(&stderr_bytes).into_owned();
        let (stdout_str, stdout_capped) =
            crate::shell::model_stdout(&command, &stdout_raw, self.compress_output);
        let (stderr_str, stderr_capped) = crate::shell::cap_for_model(&stderr_raw);
        stdout_trunc |= stdout_capped;
        stderr_trunc |= stderr_capped;

        Ok(json!({
            "shell_id": parsed.shell_id,
            "command": command,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "stdout_truncated": stdout_trunc,
            "stderr_truncated": stderr_trunc,
            "running": running,
            "exit_code": exit.and_then(|s| s.code()),
            "signal": exit.and_then(signal_of_status),
        }))
    }
}

async fn poll_until_data(
    stdout: &Arc<AsyncMutex<TailBuffer>>,
    stderr: &Arc<AsyncMutex<TailBuffer>>,
    exit: &Arc<AsyncMutex<Option<std::process::ExitStatus>>>,
    deadline_after: Duration,
) {
    let start = tokio::time::Instant::now();
    let _ = timeout(deadline_after, async {
        loop {
            let s_empty = stdout.lock().await.bytes.is_empty();
            let e_empty = stderr.lock().await.bytes.is_empty();
            let exited = exit.lock().await.is_some();
            if !s_empty || !e_empty || exited {
                return;
            }
            // Sleep just long enough to keep this responsive without
            // pinning the CPU. 25 ms is the sweet spot for a tool
            // that's polled by a model rather than a user.
            tokio::time::sleep(Duration::from_millis(25)).await;
            if start.elapsed() >= deadline_after {
                return;
            }
        }
    })
    .await;
}

#[cfg(unix)]
fn signal_of_status(status: std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn signal_of_status(_status: std::process::ExitStatus) -> Option<i32> {
    None
}

// =====================================================================
// KillShell
// =====================================================================

/// Terminate a background shell. The owned supervisor closes the process tree
/// and reaps the direct child before this call returns.
#[derive(Debug)]
pub struct KillShellTool {
    registry: BashSessionRegistry,
}

impl KillShellTool {
    pub fn new(registry: BashSessionRegistry) -> Self {
        Self { registry }
    }
}

#[derive(Debug, Deserialize)]
struct KillShellInput {
    shell_id: String,
}

#[async_trait]
impl Tool for KillShellTool {
    fn name(&self) -> &str {
        "KillShell"
    }
    fn description(&self) -> &str {
        "Terminate a background shell started by BashRun. Removes the session from the registry. No-op if the shell already exited."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"shell_id": {"type": "string"}},
            "required": ["shell_id"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(&self, _ctx: &ToolUseContext, input: Value) -> Result<Value, AgentError> {
        let parsed: KillShellInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("KillShell invalid input: {e}")))?;
        let removed = self.registry.inner.write().await.remove(&parsed.shell_id);
        let session = removed.ok_or_else(|| {
            AgentError::other(format!("KillShell: no shell with id '{}'", parsed.shell_id))
        })?;

        session.stop().await;

        Ok(json!({
            "shell_id": parsed.shell_id,
            "killed": true,
        }))
    }
}

#[cfg(test)]
#[path = "bash_async_tests.rs"]
mod tests;
