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

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

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

#[derive(Debug)]
pub struct BashTool {
    policy: Arc<WorkspacePolicy>,
}

impl BashTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
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

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::other("Bash spawn missing stdout pipe"))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| AgentError::other("Bash spawn missing stderr pipe"))?;

        // Capture the child pid for the kill-group fallback below
        // (kill_on_drop only kills the direct child, not the group).
        #[cfg(unix)]
        let child_pid: Option<u32> = child.id();

        let abort = ctx.abort.clone();
        let exec = async move {
            let read_out = read_capped(&mut stdout, MAX_OUTPUT_BYTES);
            let read_err = read_capped(&mut stderr, MAX_OUTPUT_BYTES);
            let (a, b) = tokio::join!(read_out, read_err);
            let (out_bytes, out_truncated) =
                a.map_err(|e| AgentError::other(format!("Bash stdout read failed: {e}")))?;
            let (err_bytes, err_truncated) =
                b.map_err(|e| AgentError::other(format!("Bash stderr read failed: {e}")))?;
            let status = child
                .wait()
                .await
                .map_err(|e| AgentError::other(format!("Bash wait failed: {e}")))?;
            Ok::<(Vec<u8>, bool, Vec<u8>, bool, std::process::ExitStatus), AgentError>((
                out_bytes,
                out_truncated,
                err_bytes,
                err_truncated,
                status,
            ))
        };

        // Wait for either: command finishes, timeout fires, or
        // host abort fires. The `kill_on_drop(true)` flag kills the
        // direct child when the future is dropped; on Unix, we
        // also `kill -9 -<pgid>` to terminate the whole process
        // group so descendants don't outlive the timeout.
        let (stdout_bytes, stdout_truncated, stderr_bytes, stderr_truncated, status) = tokio::select! {
            biased;
            _ = abort.cancelled() => {
                #[cfg(unix)]
                kill_process_group(child_pid);
                return Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "aborted".into()),
                ));
            }
            result = timeout(Duration::from_secs(timeout_secs), exec) => {
                match result {
                    Ok(Ok(quintuple)) => quintuple,
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        #[cfg(unix)]
                        kill_process_group(child_pid);
                        return Err(AgentError::other(format!(
                            "Bash command timed out after {timeout_secs}s. no shell_id was created for this foreground Bash call; BashOutput can only poll commands started with BashRun. For long-running commands, start them with BashRun and then poll that shell_id with BashOutput."
                        )));
                    }
                }
            }
        };

        let stdout_str = format_capture(stdout_bytes, stdout_truncated);
        let stderr_str = format_capture(stderr_bytes, stderr_truncated);

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
async fn read_capped<R>(reader: &mut R, cap: usize) -> std::io::Result<(Vec<u8>, bool)>
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

/// Send SIGKILL to the entire process group rooted at `pid` via
/// `/bin/kill -9 -<pid>`. We shell out instead of pulling `nix` /
/// `libc` because (a) it works without an `unsafe` block, and (b)
/// `kill(1)` is on every POSIX target. Best-effort: failures are
/// logged at `debug` and swallowed so the caller's error path isn't
/// shadowed.
#[cfg(unix)]
fn kill_process_group(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    // Negative PID = process group with that leader.
    let arg = format!("-{pid}");
    match std::process::Command::new("/bin/kill")
        .arg("-9")
        .arg(&arg)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(s) if s.success() => {}
        Ok(s) => tracing::debug!(pgid = pid, status = ?s, "/bin/kill -9 returned non-zero"),
        Err(e) => tracing::debug!(pgid = pid, error = %e, "/bin/kill -9 spawn failed"),
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
mod tests {
    use super::*;
    use agent::abort::AbortController;
    use std::num::NonZeroUsize;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn long_running_commands_get_a_forgiving_default() {
        // Network/build commands get the longer default...
        for cmd in [
            "git clone git@github.com:x/y.git",
            "cd /tmp && npm install",
            "cargo build --release",
            "pip install requests",
            "curl -fsSL https://example.com/install.sh | sh",
        ] {
            assert_eq!(default_timeout_for(cmd), LONG_RUNNING_TIMEOUT_SECS, "{cmd}");
        }
        // ...while ordinary commands keep the snappy 60s default.
        for cmd in ["ls -la", "echo hi", "grep foo bar.txt", "cat README.md"] {
            assert_eq!(default_timeout_for(cmd), DEFAULT_TIMEOUT_SECS, "{cmd}");
        }
    }

    #[test]
    fn broad_git_add_detection_blocks_only_unscoped_staging() {
        for cmd in [
            "git add -A",
            "git add --all",
            "git add .",
            "cd repo && git add -A && git commit -m x",
            "git -C repo add --all",
        ] {
            assert!(broad_git_add(cmd), "{cmd}");
        }

        for cmd in [
            "git add -- src/lib.rs",
            "git add -A src/lib.rs",
            "git add -u",
            "git status --short",
            "echo 'git add -A'",
        ] {
            assert!(!broad_git_add(cmd), "{cmd}");
        }
    }

    #[tokio::test]
    async fn bash_rejects_broad_git_staging() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let err = tool
            .call(&ctx(), json!({"command": "git add -A"}))
            .await
            .expect_err("broad staging should be blocked before spawn");
        let msg = err.to_string();
        assert!(msg.contains("blocked broad git staging"), "{msg}");
        assert!(msg.contains("GitCommit"), "{msg}");
        assert!(msg.contains("paths"), "{msg}");
    }

    fn ctx() -> ToolUseContext {
        ToolUseContext {
            cwd: std::env::current_dir().unwrap(),
            abort: AbortController::new(),
            file_cache: Arc::new(agent::file_cache::FileStateCache::new(
                NonZeroUsize::new(8).unwrap(),
                1024 * 1024,
            )),
            permissions: Arc::new(agent::permission::PermissionManager::new()),
            hooks: Arc::new(agent::hook::HookRunner::new()),
            task_depth: 0,
        }
    }

    fn policy_for(dir: &Path) -> Arc<WorkspacePolicy> {
        WorkspacePolicy::new(dir).unwrap().into_arc()
    }

    #[tokio::test]
    async fn bash_runs_and_captures_stdout() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert_eq!(out["exit_code"], 0);
        assert!(out["stdout"].as_str().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn bash_captures_nonzero_exit() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"command": "exit 7"}))
            .await
            .unwrap();
        assert_eq!(out["exit_code"], 7);
    }

    #[tokio::test]
    async fn bash_captures_stderr_separately() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"command": "echo to-stderr 1>&2"}))
            .await
            .unwrap();
        assert!(out["stderr"].as_str().unwrap().contains("to-stderr"));
        assert!(out["stdout"].as_str().unwrap().is_empty());
    }

    // `pwd` on Windows runners is Git-for-Windows' MSYS binary, which
    // emits `/d/a/...`-style paths that `std::fs::canonicalize` can't
    // parse as native Windows paths. The cwd-plumbing logic is OS-
    // independent — coverage on Unix is sufficient.
    #[cfg(unix)]
    #[tokio::test]
    async fn bash_runs_in_policy_cwd_by_default() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool.call(&ctx(), json!({"command": "pwd"})).await.unwrap();
        // Canonicalized tempdir on macOS goes through /private/...,
        // so just check that the resolved cwd was used.
        let cwd_in_output = out["stdout"].as_str().unwrap().trim();
        assert!(
            std::fs::canonicalize(cwd_in_output)
                .unwrap()
                .ends_with(dir.path().file_name().unwrap()),
            "got {cwd_in_output}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_runs_without_a_controlling_tty() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool
            .call(
                &ctx(),
                json!({"command": "if (: >/dev/tty) 2>/dev/null; then echo HAS_TTY; else echo NO_TTY; fi"}),
            )
            .await
            .unwrap();
        assert_eq!(out["stdout"].as_str().unwrap().trim(), "NO_TTY");
    }

    // Same Windows/MSYS pwd issue as above — gate cfg(unix).
    #[cfg(unix)]
    #[tokio::test]
    async fn bash_explicit_cwd_validated_through_policy() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"command": "pwd", "cwd": "sub"}))
            .await
            .unwrap();
        assert!(out["stdout"].as_str().unwrap().trim().ends_with("sub"));
    }

    #[tokio::test]
    async fn bash_rejects_cwd_outside_workspace() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let err = tool
            .call(
                &ctx(),
                json!({"command": "pwd", "cwd": outside.path().to_str().unwrap()}),
            )
            .await
            .expect_err("should fail");
        assert!(err.to_string().contains("policy"));
    }

    #[tokio::test]
    async fn bash_empty_command_rejected() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let err = tool
            .call(&ctx(), json!({"command": "   "}))
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("non-empty"));
    }

    // `sleep` isn't a Windows `cmd` builtin and may not be on PATH in
    // restricted CI, even though Git-for-Windows runners happen to ship
    // one. Gate to keep behavior deterministic.
    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_kills_long_running_command() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let err = tool
            .call(&ctx(), json!({"command": "sleep 30", "timeout_secs": 1}))
            .await
            .expect_err("timeout");
        assert!(err.to_string().contains("timed out"), "got {err}");
        assert!(err.to_string().contains("no shell_id"), "got {err}");
        assert!(err.to_string().contains("BashRun"), "got {err}");
        assert!(err.to_string().contains("BashOutput"), "got {err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_aborts_on_ctx_abort() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let c = ctx();
        c.abort.abort_with_reason("user cancelled");
        let err = tool
            .call(&c, json!({"command": "sleep 30"}))
            .await
            .expect_err("aborted");
        assert!(matches!(err, AgentError::Aborted(_)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_pipe_chain_works() {
        // Deterministic version: `printf` with explicit \n
        // produces three lines on every POSIX shell.
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"command": "printf 'a\\nb\\nc\\n' | wc -l"}))
            .await
            .unwrap();
        let stdout = out["stdout"].as_str().unwrap().trim();
        assert_eq!(stdout, "3");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_truncates_huge_stdout_with_tail_preserved() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        // Print 2 MiB of x's followed by "TAIL_MARKER". Uses POSIX
        // coreutils — gated cfg(unix) to keep Windows CI happy.
        let cmd = format!(
            "head -c {} /dev/zero | tr '\\0' 'x' && echo TAIL_MARKER",
            2 * 1024 * 1024
        );
        let out = tool.call(&ctx(), json!({"command": cmd})).await.unwrap();
        assert_eq!(out["stdout_truncated"], true);
        assert!(
            out["stdout"].as_str().unwrap().contains("TAIL_MARKER"),
            "tail should be preserved"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_kills_process_group_on_timeout() {
        // Spawn a shell that backgrounds a long-sleeping child, then
        // exits the foreground sleep early. With process-group kill
        // the background child also dies; without it the descendant
        // would survive past the timeout.
        // We can't easily observe "process killed externally" from
        // the test, but we CAN observe that the timeout error fires
        // — confirming the kill loop ran without panicking.
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let err = tool
            .call(
                &ctx(),
                json!({"command": "sleep 30 & sleep 30", "timeout_secs": 1}),
            )
            .await
            .expect_err("timeout");
        assert!(err.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn bash_classified_mutating() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        assert_eq!(tool.safety_class(), SafetyClass::Mutating);
    }

    #[tokio::test]
    async fn read_capped_zero_cap_discards_but_flags_truncated() {
        // cap == 0 must NOT grow the buffer regardless of input size.
        let mut data: &[u8] = b"abcdefghij";
        let (out, truncated) = read_capped(&mut data, 0).await.unwrap();
        assert!(out.is_empty());
        assert!(truncated);
        // Empty stream with cap 0 → not truncated.
        let mut empty: &[u8] = b"";
        let (out, truncated) = read_capped(&mut empty, 0).await.unwrap();
        assert!(out.is_empty());
        assert!(!truncated);
    }

    #[test]
    fn trim_to_utf8_boundary_drops_continuation_prefix() {
        // 4-byte codepoint U+1F600 = F0 9F 98 80. Cut after first byte
        // and prepend continuations: tail starts at B2/B3-style bytes.
        let bytes = vec![0x9F, 0x98, 0x80, b'a', b'b'];
        let trimmed = trim_to_utf8_boundary(bytes);
        // After trimming continuations, first byte should be ASCII 'a'.
        assert_eq!(trimmed, b"ab");
        // ASCII tail unchanged.
        assert_eq!(trim_to_utf8_boundary(b"hello".to_vec()), b"hello");
        // Multi-byte start byte (0xC3 = 2-byte) preserved.
        assert_eq!(trim_to_utf8_boundary(vec![0xC3, 0xA9]), vec![0xC3, 0xA9]);
    }

    #[tokio::test]
    async fn bash_caps_caller_supplied_timeout() {
        // Even if the caller asks for 999_999 seconds, we cap.
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new(policy_for(dir.path()));
        let out = tool
            .call(
                &ctx(),
                json!({"command": "echo ok", "timeout_secs": 999_999}),
            )
            .await
            .unwrap();
        assert_eq!(out["exit_code"], 0);
    }
}
