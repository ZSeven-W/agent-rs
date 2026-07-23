//! `Bash` — run a shell command with captured stdout/stderr.
//!
//! Modeled on Claude Code's Bash tool. The tool spawns
//! `/bin/sh -c <command>` (or `cmd /C` on Windows) so the model can
//! pass full shell expressions including pipes, redirects, and
//! command chaining. Output is captured, not streamed; long-running
//! commands hit the timeout and get SIGKILL'd.
//!
//! Safety classification: `Mutating`. Hosts that want to block
//! destructive shell shapes (e.g., `rm -rf`, `sudo`, `dd`) compose
//! `agent::permission::PermissionMatcher` rules over the
//! `/command` field. Examples in the host README of agent-tools-code.
//!
//! Timeouts cap runtime at 60s by default — change via the
//! `timeout_secs` input field. There's also a hard ceiling of 10
//! minutes regardless of caller request.
//!
//! Output capture is **per-stream** capped at 1 MiB — stdout and
//! stderr each get their own ring buffer, so a verbose stderr
//! doesn't crowd out stdout. We use a streaming ring buffer
//! (`VecDeque<u8>`) so a process that prints gigabytes never grows
//! our buffer beyond the cap; the **tail** is preserved (the part
//! the model usually cares about) and `*_truncated` flags surface
//! when truncation happened.

#![cfg_attr(all(not(feature = "shell"), feature = "bash-async"), allow(dead_code))]

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use agent::abort::TurnActivity;
use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::time::timeout;

use crate::policy::{PolicyError, WorkspacePolicy};

/// Default timeout if the caller doesn't specify. Matches Claude
/// Code's per-call default. Tools that need longer should pass an
/// explicit `timeout_secs`.
const DEFAULT_TIMEOUT_SECS: u64 = 60;
/// More forgiving default for recognized network/build commands (clone,
/// install, download, compile), which routinely exceed 60s. Still overridable
/// via `timeout_secs` and still bounded by [`MAX_TIMEOUT_SECS`].
const LONG_RUNNING_TIMEOUT_SECS: u64 = 5 * 60;
/// Hard ceiling regardless of caller request. Stops a runaway
/// command from pinning a runtime worker for an hour.
const MAX_TIMEOUT_SECS: u64 = 10 * 60;
const PROCESS_TREE_POLL: Duration = Duration::from_millis(20);
const PROCESS_TREE_CLEANUP_GRACE: Duration = Duration::from_secs(5);

/// Pick the default timeout when the caller didn't set `timeout_secs`. Network
/// and build commands get [`LONG_RUNNING_TIMEOUT_SECS`] so a normal `git clone`
/// or `npm install` doesn't fail at 60s; everything else keeps the snappy
/// 60s default that surfaces hangs quickly.
fn default_timeout_for(command: &str) -> u64 {
    const SLOW: &[&str] = &[
        "git clone",
        "git fetch",
        "git pull",
        "git submodule",
        "git lfs",
        "npm install",
        "npm ci",
        "npm i ",
        "pnpm install",
        "pnpm i",
        "yarn",
        "cargo build",
        "cargo install",
        "cargo fetch",
        "cargo update",
        "cargo test",
        "pip install",
        "pip3 install",
        "poetry install",
        "uv pip",
        "uv sync",
        "go mod download",
        "go install",
        "go build",
        "go get",
        "bundle install",
        "gem install",
        "brew install",
        "brew upgrade",
        "apt install",
        "apt-get install",
        "dnf install",
        "yum install",
        "docker build",
        "docker pull",
        "docker compose",
        "make ",
        "cmake ",
        "curl ",
        "wget ",
        "gradle",
        "mvn ",
        "./gradlew",
    ];
    let c = command.to_ascii_lowercase();
    if SLOW.iter().any(|p| c.contains(p)) {
        LONG_RUNNING_TIMEOUT_SECS
    } else {
        DEFAULT_TIMEOUT_SECS
    }
}

fn broad_git_add(command: &str) -> bool {
    let words = match shell_words::split(command) {
        Ok(words) => words,
        Err(_) => command.split_whitespace().map(str::to_string).collect(),
    };
    for segment in words.split(|w| matches!(w.as_str(), "&&" | "||" | ";")) {
        if git_add_is_broad(segment) {
            return true;
        }
    }
    false
}

fn git_add_is_broad(words: &[String]) -> bool {
    let Some(mut i) = words
        .iter()
        .position(|w| w.eq_ignore_ascii_case("git"))
        .map(|idx| idx + 1)
    else {
        return false;
    };
    while i < words.len() {
        let word = words[i].as_str();
        if word.eq_ignore_ascii_case("add") {
            return git_add_args_are_broad(&words[i + 1..]);
        }
        match word {
            "-C" | "-c" | "--git-dir" | "--work-tree" if i + 1 < words.len() => i += 2,
            _ if word.starts_with("--git-dir=")
                || word.starts_with("--work-tree=")
                || word.starts_with("-c") =>
            {
                i += 1
            }
            _ => return false,
        }
    }
    false
}

fn git_add_args_are_broad(args: &[String]) -> bool {
    let mut broad_flag = false;
    let mut specific_path = false;
    for arg in args {
        let arg = arg.as_str();
        if matches!(arg, "&&" | "||" | ";") {
            break;
        }
        if matches!(arg, "-A" | "--all") {
            broad_flag = true;
            continue;
        }
        if matches!(arg, "." | "./" | ":/") {
            return true;
        }
        if arg == "--" || arg.starts_with('-') {
            continue;
        }
        specific_path = true;
    }
    broad_flag && !specific_path
}
/// Per-stream capture cap. Above this, output truncates; the tail
/// (the last `MAX_OUTPUT_BYTES` bytes of the stream) is what we
/// surface, since panic / error messages are usually at the end.
/// Implemented via a streaming ring buffer so the process can emit
/// gigabytes without growing our memory beyond the cap.
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
/// Model-facing output cap (per stream). This is much smaller than
/// `MAX_OUTPUT_BYTES`, which is only an in-memory guard. Head + tail are kept
/// so a command that dumps a huge file cannot fill the model transcript.
const MODEL_CAP_BYTES: usize = 8 * 1024;
const MODEL_HEAD_BYTES: usize = 5 * 1024;
const MODEL_TAIL_BYTES: usize = 3 * 1024;

#[derive(Debug)]
pub struct BashTool {
    policy: Arc<WorkspacePolicy>,
    compress_output: bool,
}

impl BashTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self::with_compress_output(policy, true)
    }

    pub fn with_compress_output(policy: Arc<WorkspacePolicy>, compress_output: bool) -> Self {
        Self {
            policy,
            compress_output,
        }
    }
}

#[derive(Debug, Deserialize)]
struct BashInput {
    command: String,
    /// Working directory. Resolved against the workspace policy. If
    /// omitted, runs in `policy.cwd`.
    #[serde(default)]
    cwd: Option<String>,
    /// Per-call timeout in seconds. Capped at [`MAX_TIMEOUT_SECS`].
    #[serde(default)]
    timeout_secs: Option<u64>,
}

fn policy_to_agent_err(e: PolicyError) -> AgentError {
    AgentError::other(format!("policy: {e}"))
}

fn floor_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

pub(crate) fn cap_for_model(s: &str) -> (String, bool) {
    if s.len() <= MODEL_CAP_BYTES {
        return (s.to_string(), false);
    }
    let head_budget = floor_char_boundary(s, MODEL_HEAD_BYTES);
    let head_end = s[..head_budget]
        .rfind('\n')
        .map(|i| i + 1)
        .unwrap_or_else(|| floor_char_boundary(s, MODEL_HEAD_BYTES));
    let tail_from = ceil_char_boundary(s, s.len().saturating_sub(MODEL_TAIL_BYTES));
    let tail_start = s[..tail_from]
        .rfind('\n')
        .map(|i| i + 1)
        .unwrap_or(tail_from);
    if tail_start <= head_end {
        return (s.to_string(), false);
    }
    let elided = tail_start - head_end;
    let out = format!(
        "{}\n... output truncated: {} bytes elided (total {} bytes). Re-run through \
         a filter (grep/head/tail) or use FileRead(offset,limit) / Grep instead of \
         dumping whole files.\n{}",
        &s[..head_end],
        elided,
        s.len(),
        &s[tail_start..],
    );
    (out, true)
}

pub(crate) fn model_stdout(command: &str, raw: &str, compress: bool) -> (String, bool) {
    if compress {
        if let Some(compressed) = crate::compress_command(command, raw) {
            let body = if compressed.note.is_empty() {
                compressed.text
            } else {
                format!("{}\n{}", compressed.text, compressed.note)
            };
            let (capped, _) = cap_for_model(&body);
            return (capped, true);
        }
    }
    cap_for_model(raw)
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "Bash"
    }
    fn description(&self) -> &str {
        "Run a shell command. Captures stdout/stderr/exit_code. Runs non-interactively (no prompts). Per-call timeout: 60s default, 300s for clone/install/build/download commands, max 600s — pass timeout_secs to override. Output capped at 1 MiB."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command. Runs via `/bin/sh -c` (Unix) or `cmd /C` (Windows)."},
                "cwd": {"type": "string", "description": "Working dir (default: workspace cwd)."},
                "timeout_secs": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECS}
            },
            "required": ["command"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(
        &self,
        ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: BashInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("Bash invalid input: {e}")))?;
        if parsed.command.trim().is_empty() {
            return Err(AgentError::other("Bash command must be non-empty"));
        }
        if broad_git_add(&parsed.command) {
            return Err(AgentError::other(
                "Bash blocked broad git staging (`git add -A`, `git add --all`, or `git add .`). Stage explicit paths with `git add -- <paths>` or use GitCommit with `paths` so unrelated files such as node_modules, dist, or IDE metadata are not included.",
            ));
        }
        let timeout_secs = parsed
            .timeout_secs
            .unwrap_or_else(|| default_timeout_for(&parsed.command))
            .clamp(1, MAX_TIMEOUT_SECS);
        let cwd = match parsed.cwd.as_deref() {
            Some(p) => self.policy.resolve(p, true).map_err(policy_to_agent_err)?,
            None => self.policy.cwd.clone(),
        };

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

        // Run non-interactively. stdin is already /dev/null, but git and ssh
        // read prompts from the TTY, so an unknown host key or missing
        // credential would hang the command until the timeout (the classic
        // "git clone hangs forever"). These make such cases fail fast with a
        // clear error the agent can act on, instead of burning the budget.
        cmd.env("GIT_TERMINAL_PROMPT", "0")
            .env(
                "GIT_SSH_COMMAND",
                "ssh -oBatchMode=yes -oStrictHostKeyChecking=accept-new",
            )
            .env("GCM_INTERACTIVE", "never");

        crate::process::detach_from_controlling_tty(&mut cmd);

        let mut child = cmd
            .spawn()
            .map_err(|e| AgentError::other(format!("Bash spawn failed: {e}")))?;

        // Arm descendant cleanup before doing anything that can return early.
        // `kill_on_drop` covers only the direct child; this guard supplies the
        // platform tree fallback for aborts, timeouts, hard future drops, and
        // successful leaders that leave background children behind.
        let child_pid = child.id();
        let abort = ctx.abort.clone();
        let mut process_tree_guard = ProcessTreeGuard::new(child_pid, abort.activity());

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::other("Bash spawn missing stdout pipe"))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| AgentError::other("Bash spawn missing stderr pipe"))?;

        let stdout_activity = abort.activity();
        let stderr_activity = stdout_activity.clone();

        enum Completion {
            Finished(Result<(Vec<u8>, bool, Vec<u8>, bool, std::process::ExitStatus), AgentError>),
            Aborted,
            TimedOut,
        }

        // Wait for either: command finishes, timeout fires, or
        // host abort fires. The `kill_on_drop(true)` flag kills the
        // direct child when the future is dropped; the tree guard also kills
        // the Unix process group or invokes Windows `taskkill /T`. Commands
        // that intentionally need a surviving process must use `BashRun`.
        let completion = {
            let exec = async {
                let read_out = read_capped(&mut stdout, MAX_OUTPUT_BYTES, &stdout_activity);
                let read_err = read_capped(&mut stderr, MAX_OUTPUT_BYTES, &stderr_activity);
                let (a, b, status) = tokio::join!(read_out, read_err, child.wait());
                let (out_bytes, out_truncated) =
                    a.map_err(|e| AgentError::other(format!("Bash stdout read failed: {e}")))?;
                let (err_bytes, err_truncated) =
                    b.map_err(|e| AgentError::other(format!("Bash stderr read failed: {e}")))?;
                let status =
                    status.map_err(|e| AgentError::other(format!("Bash wait failed: {e}")))?;
                Ok((out_bytes, out_truncated, err_bytes, err_truncated, status))
            };
            tokio::pin!(exec);
            tokio::select! {
                biased;
                _ = abort.cancelled() => Completion::Aborted,
                _ = tokio::time::sleep(Duration::from_secs(timeout_secs)) => Completion::TimedOut,
                result = &mut exec => Completion::Finished(result),
            }
        };

        let (stdout_bytes, stdout_truncated, stderr_bytes, stderr_truncated, status) =
            match completion {
                Completion::Finished(result) => {
                    process_tree_guard.cleanup_after_leader_exit().await;
                    result?
                }
                Completion::Aborted => {
                    process_tree_guard.terminate_and_reap(&mut child).await;
                    return Err(AgentError::Aborted(
                        abort.reason().unwrap_or_else(|| "aborted".into()),
                    ));
                }
                Completion::TimedOut => {
                    process_tree_guard.terminate_and_reap(&mut child).await;
                    return Err(AgentError::other(format!(
                        "Bash command timed out after {timeout_secs}s. no shell_id was created for this foreground Bash call; BashOutput can only poll commands started with BashRun. For long-running commands, start them with BashRun and then poll that shell_id with BashOutput."
                    )));
                }
            };

        let stdout_str = format_capture(stdout_bytes, stdout_truncated);
        let stderr_str = format_capture(stderr_bytes, stderr_truncated);
        let (stdout_str, stdout_capped) =
            model_stdout(&parsed.command, &stdout_str, self.compress_output);
        let (stderr_str, stderr_capped) = cap_for_model(&stderr_str);
        let stdout_truncated = stdout_truncated || stdout_capped;
        let stderr_truncated = stderr_truncated || stderr_capped;

        Ok(json!({
            "exit_code": status.code(),
            "signal": signal_of(&status),
            "stdout": stdout_str,
            "stderr": stderr_str,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "cwd": cwd.display().to_string(),
        }))
    }
}

/// Stream-read up to `cap` bytes from `reader`, keeping the tail
/// when the stream is longer than the cap. Memory stays bounded at
/// `cap` regardless of how much the process emits — gigabytes of
/// `yes` won't OOM the agent.
///
/// `cap == 0` is treated as "discard everything but track that the
/// stream was non-empty", so the buffer can never grow.
async fn read_capped<R>(
    reader: &mut R,
    cap: usize,
    activity: &TurnActivity,
) -> std::io::Result<(Vec<u8>, bool)>
where
    R: AsyncRead + Unpin,
{
    let mut tmp = [0u8; 16 * 1024];
    if cap == 0 {
        let mut total: u64 = 0;
        loop {
            let n = reader.read(&mut tmp).await?;
            if n == 0 {
                break;
            }
            activity.pulse();
            total = total.saturating_add(n as u64);
        }
        return Ok((Vec::new(), total > 0));
    }
    let mut tail: VecDeque<u8> = VecDeque::with_capacity(cap.min(64 * 1024));
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        activity.pulse();
        total = total.saturating_add(n as u64);
        for &b in &tmp[..n] {
            if tail.len() == cap {
                tail.pop_front();
            }
            tail.push_back(b);
        }
    }
    let truncated = total > cap as u64;
    Ok((tail.into_iter().collect(), truncated))
}

/// When tail-truncation cuts mid-codepoint, `String::from_utf8_lossy`
/// inserts a U+FFFD at the prefix. Trim the leading bytes to the
/// next valid UTF-8 boundary so output starts on a real character.
///
/// In well-formed UTF-8 every codepoint is at most 4 bytes, so at
/// most 3 leading continuation bytes (top two bits == `10`) can
/// follow a cut. We scan up to 4 bytes; pathological non-UTF-8
/// streams may still surface U+FFFDs via `from_utf8_lossy`, which
/// is acceptable — the function only promises a best-effort fix
/// for clean UTF-8 inputs that happened to be cut mid-character.
fn trim_to_utf8_boundary(bytes: Vec<u8>) -> Vec<u8> {
    for (i, &b) in bytes.iter().take(4).enumerate() {
        if b & 0b1100_0000 != 0b1000_0000 {
            return if i == 0 { bytes } else { bytes[i..].to_vec() };
        }
    }
    bytes
}

fn format_capture(bytes: Vec<u8>, truncated: bool) -> String {
    let bytes = if truncated {
        trim_to_utf8_boundary(bytes)
    } else {
        bytes
    };
    let body = String::from_utf8_lossy(&bytes);
    if truncated {
        format!("[output truncated; tail preserved]\n{body}")
    } else {
        body.into_owned()
    }
}

struct ProcessTreeGuard {
    pid: Option<u32>,
    activity: TurnActivity,
    armed: bool,
}

impl ProcessTreeGuard {
    fn new(pid: Option<u32>, activity: TurnActivity) -> Self {
        Self {
            pid,
            activity,
            armed: true,
        }
    }

    async fn cleanup_after_leader_exit(&mut self) {
        let signal_succeeded = kill_process_tree(self.pid);
        let tree_exit_proven = wait_for_process_tree_exit(self.pid, signal_succeeded).await;
        self.complete_cleanup(tree_exit_proven);
    }

    async fn terminate_and_reap(&mut self, child: &mut tokio::process::Child) {
        let signal_succeeded = kill_process_tree(self.pid);
        let _ = child.start_kill();
        let direct_child_reaped = matches!(
            timeout(PROCESS_TREE_CLEANUP_GRACE, child.wait()).await,
            Ok(Ok(_))
        );
        let tree_exit_proven = wait_for_process_tree_exit(self.pid, signal_succeeded).await;
        self.complete_cleanup(direct_child_reaped && signal_succeeded && tree_exit_proven);
    }

    fn complete_cleanup(&mut self, proven: bool) {
        if !proven {
            self.activity.mark_unresolved_external_work();
        }
        self.armed = false;
    }
}

impl Drop for ProcessTreeGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = kill_process_tree(self.pid);
            // A synchronous drop can signal the tree but cannot wait for an
            // exit proof. Keep scheduler replay fail-closed.
            self.activity.mark_unresolved_external_work();
        }
    }
}

/// Send SIGKILL to the entire process group rooted at `pid`. Failures are
/// reported to the guard so unprovable cleanup latches the turn as unresolved
/// without shadowing the caller's original error.
#[cfg(unix)]
fn kill_process_tree(pid: Option<u32>) -> bool {
    let Some(pid) = pid else {
        return false;
    };
    matches!(
        crate::process::kill_process_group(pid),
        Ok(crate::process::ProcessGroupSignal::Delivered)
    )
}

/// Best-effort Windows descendant cleanup. `kill_on_drop` only terminates the
/// shell itself, whereas `/T` walks and terminates its child process tree.
#[cfg(windows)]
fn kill_process_tree(pid: Option<u32>) -> bool {
    let Some(pid) = pid else { return false };
    let pid = pid.to_string();
    match std::process::Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(s) if s.success() => true,
        Ok(s) => {
            tracing::debug!(pid, status = ?s, "taskkill /T returned non-zero");
            false
        }
        Err(e) => {
            tracing::debug!(pid, error = %e, "taskkill /T spawn failed");
            false
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn kill_process_tree(_pid: Option<u32>) -> bool {
    false
}

async fn wait_for_process_tree_exit(pid: Option<u32>, signal_succeeded: bool) -> bool {
    #[cfg(unix)]
    {
        let _ = signal_succeeded;
        let deadline = tokio::time::Instant::now() + PROCESS_TREE_CLEANUP_GRACE;
        loop {
            match unix_process_tree_state(pid) {
                ProcessTreeState::Gone => return true,
                ProcessTreeState::Unknown => return false,
                ProcessTreeState::Alive if tokio::time::Instant::now() >= deadline => {
                    return false;
                }
                ProcessTreeState::Alive => tokio::time::sleep(PROCESS_TREE_POLL).await,
            }
        }
    }

    #[cfg(windows)]
    return signal_succeeded;

    #[cfg(not(any(unix, windows)))]
    {
        let _ = (pid, signal_succeeded);
        false
    }
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessTreeState {
    Alive,
    Gone,
    Unknown,
}

#[cfg(unix)]
fn unix_process_tree_state(pid: Option<u32>) -> ProcessTreeState {
    let Some(pid) = pid else {
        return ProcessTreeState::Unknown;
    };
    match crate::process::process_group_state(pid) {
        Ok(crate::process::ProcessGroupState::Alive) => ProcessTreeState::Alive,
        Ok(crate::process::ProcessGroupState::Gone) => ProcessTreeState::Gone,
        Err(_) => ProcessTreeState::Unknown,
    }
}

#[cfg(unix)]
fn signal_of(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn signal_of(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

#[cfg(test)]
#[path = "shell_tests.rs"]
mod tests;
