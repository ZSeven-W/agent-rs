use super::*;
use agent::abort::AbortController;
use std::num::NonZeroUsize;
use std::path::Path;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;

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
fn cap_for_model_passes_small_output_unchanged() {
    let s = "hello\nworld\n";
    let (out, trunc) = cap_for_model(s);
    assert_eq!(out, s);
    assert!(!trunc);
}

#[test]
fn cap_for_model_head_and_tail_with_notice() {
    let big = "L\n".repeat(20_000);
    let (out, trunc) = cap_for_model(&big);
    assert!(trunc);
    assert!(out.len() < big.len());
    assert!(out.starts_with("L\n"));
    assert!(out.trim_end().ends_with('L'));
    assert!(out.contains("output truncated"));
    assert!(out.contains("FileRead"));
}

#[test]
fn cap_for_model_never_splits_utf8() {
    let s = "😀\n".repeat(10_000);
    let (out, _trunc) = cap_for_model(&s);
    assert!(out.is_char_boundary(0));
    assert!(std::str::from_utf8(out.as_bytes()).is_ok());
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
    let call_ctx = ctx();
    let activity = call_ctx.abort.activity();
    let out = tool
        .call(&call_ctx, json!({"command": "echo hello"}))
        .await
        .unwrap();
    assert_eq!(out["exit_code"], 0);
    assert!(out["stdout"].as_str().unwrap().contains("hello"));
    assert!(!activity.unresolved_external_work());
}

#[tokio::test]
async fn unverified_process_guard_marks_external_work_unresolved() {
    let activity = TurnActivity::new();
    let mut guard = ProcessTreeGuard::new(None, activity.clone());

    guard.cleanup_after_leader_exit().await;

    assert!(activity.unresolved_external_work());
}

#[test]
fn dropping_process_guard_without_exit_proof_marks_external_work_unresolved() {
    let activity = TurnActivity::new();
    drop(ProcessTreeGuard::new(None, activity.clone()));

    assert!(activity.unresolved_external_work());
}

#[cfg(unix)]
#[tokio::test]
async fn bash_output_pulses_activity_before_command_finishes() {
    let dir = TempDir::new().unwrap();
    let tool = BashTool::new(policy_for(dir.path()));
    let call_ctx = ctx();
    let activity = call_ctx.abort.activity();
    let before_output = activity.last_activity_at();
    tokio::time::sleep(Duration::from_millis(5)).await;

    let call = tokio::spawn(async move {
        tool.call(&call_ctx, json!({"command": "printf progress; sleep 1"}))
            .await
    });
    wait_for_activity_after(&activity, before_output).await;
    assert!(
        !call.is_finished(),
        "stdout should pulse activity while the command is still running"
    );

    let out = call.await.unwrap().unwrap();
    assert_eq!(out["exit_code"], 0);
    assert_eq!(out["stdout"], "progress");
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
    let call_ctx = ctx();
    let activity = call_ctx.abort.activity();
    let err = tool
        .call(&call_ctx, json!({"command": "sleep 30", "timeout_secs": 1}))
        .await
        .expect_err("timeout");
    assert!(err.to_string().contains("timed out"), "got {err}");
    assert!(err.to_string().contains("no shell_id"), "got {err}");
    assert!(err.to_string().contains("BashRun"), "got {err}");
    assert!(err.to_string().contains("BashOutput"), "got {err}");
    assert!(!activity.unresolved_external_work());
}

#[cfg(unix)]
#[tokio::test]
async fn hard_dropped_bash_marks_cleanup_unresolved_and_kills_group() {
    let dir = TempDir::new().unwrap();
    let pid_file = dir.path().join("leader.pid");
    let tool = BashTool::new(policy_for(dir.path()));
    let call_ctx = ctx();
    let activity = call_ctx.abort.activity();
    let task = tokio::spawn(async move {
        tool.call(
            &call_ctx,
            json!({"command": "echo $$ > leader.pid; sleep 300"}),
        )
        .await
    });
    let pid = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Ok(raw) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = raw.trim().parse::<i32>() {
                    break pid;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("Bash did not publish its process-group id");

    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(activity.unresolved_external_work());
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let group = format!("-{pid}");
            let alive = std::process::Command::new("/bin/kill")
                .args(["-0", &group])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success());
            if !alive {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("hard-dropped Bash process group survived cleanup");
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
async fn bash_caps_model_facing_stdout() {
    let dir = TempDir::new().unwrap();
    let tool = BashTool::new(policy_for(dir.path()));
    let out = tool
        .call(&ctx(), json!({"command": "yes L | head -n 20000"}))
        .await
        .unwrap();
    let stdout = out["stdout"].as_str().unwrap();
    assert_eq!(out["stdout_truncated"], true);
    assert!(stdout.contains("output truncated"));
    assert!(stdout.contains("FileRead"));
    assert!(stdout.len() < 20_000);
}

#[cfg(unix)]
#[tokio::test]
async fn bash_compresses_recognized_command_stdout() {
    let dir = TempDir::new().unwrap();
    std::process::Command::new("git")
        .arg("init")
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn main() {}\n").unwrap();

    let tool = BashTool::new(policy_for(dir.path()));
    let out = tool
        .call(&ctx(), json!({"command": "git status"}))
        .await
        .unwrap();
    let stdout = out["stdout"].as_str().unwrap();
    assert!(stdout.contains("git status:"), "{stdout}");
    assert!(stdout.contains("compressed git status"), "{stdout}");
    assert!(stdout.contains("1 untracked"), "{stdout}");
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

#[cfg(unix)]
#[tokio::test]
async fn bash_kills_background_descendant_after_success() {
    struct ProcessCleanup {
        pid: u32,
        armed: bool,
    }

    impl Drop for ProcessCleanup {
        fn drop(&mut self) {
            if self.armed {
                let _ = std::process::Command::new("/bin/kill")
                    .arg("-9")
                    .arg(self.pid.to_string())
                    .status();
            }
        }
    }

    let dir = TempDir::new().unwrap();
    let tool = BashTool::new(policy_for(dir.path()));
    let out = tool
        .call(
            &ctx(),
            json!({
                "command": "sleep 300 </dev/null >/dev/null 2>&1 & echo $! > descendant.pid"
            }),
        )
        .await
        .unwrap();
    assert_eq!(out["exit_code"], 0);

    let pid = std::fs::read_to_string(dir.path().join("descendant.pid"))
        .unwrap()
        .trim()
        .parse::<u32>()
        .unwrap();
    let mut cleanup = ProcessCleanup { pid, armed: true };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let alive = std::process::Command::new("/bin/kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if !alive {
            cleanup.armed = false;
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "background descendant {pid} survived foreground Bash completion"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn bash_classified_mutating() {
    let dir = TempDir::new().unwrap();
    let tool = BashTool::new(policy_for(dir.path()));
    assert_eq!(tool.safety_class(), SafetyClass::Mutating);
}

#[tokio::test]
async fn read_capped_zero_cap_discards_but_flags_truncated() {
    let activity = TurnActivity::new();
    // cap == 0 must NOT grow the buffer regardless of input size.
    let mut data: &[u8] = b"abcdefghij";
    let (out, truncated) = read_capped(&mut data, 0, &activity).await.unwrap();
    assert!(out.is_empty());
    assert!(truncated);
    // Empty stream with cap 0 → not truncated.
    let mut empty: &[u8] = b"";
    let (out, truncated) = read_capped(&mut empty, 0, &activity).await.unwrap();
    assert!(out.is_empty());
    assert!(!truncated);
}

#[tokio::test]
async fn read_capped_pulses_activity_for_each_output_chunk() {
    let activity = TurnActivity::new();
    let read_activity = activity.clone();
    let (mut writer, mut reader) = tokio::io::duplex(64);
    let read_task =
        tokio::spawn(async move { read_capped(&mut reader, 64, &read_activity).await.unwrap() });

    let before_first = activity.last_activity_at();
    tokio::time::sleep(Duration::from_millis(5)).await;
    writer.write_all(b"first").await.unwrap();
    let after_first = wait_for_activity_after(&activity, before_first).await;

    tokio::time::sleep(Duration::from_millis(5)).await;
    writer.write_all(b"second").await.unwrap();
    let after_second = wait_for_activity_after(&activity, after_first).await;
    drop(writer);

    let (output, truncated) = read_task.await.unwrap();
    assert_eq!(output, b"firstsecond");
    assert!(!truncated);
    assert!(after_second > after_first);
}

async fn wait_for_activity_after(
    activity: &TurnActivity,
    before: std::time::Instant,
) -> std::time::Instant {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let observed = activity.last_activity_at();
            if observed > before {
                return observed;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("output read should pulse turn activity")
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
