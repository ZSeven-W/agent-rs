use super::*;
use agent::abort::AbortController;
use agent::file_cache::FileStateCache;
use agent::hook::HookRunner;
use agent::permission::PermissionManager;
use std::num::NonZeroUsize;
use std::path::Path;
use tempfile::TempDir;

fn ctx() -> ToolUseContext {
    ToolUseContext {
        cwd: std::env::current_dir().unwrap(),
        abort: AbortController::new(),
        file_cache: Arc::new(FileStateCache::new(
            NonZeroUsize::new(8).unwrap(),
            1024 * 1024,
        )),
        permissions: Arc::new(PermissionManager::new()),
        hooks: Arc::new(HookRunner::new()),
        task_depth: 0,
    }
}

fn policy_for(dir: &Path) -> Arc<WorkspacePolicy> {
    WorkspacePolicy::new(dir).unwrap().into_arc()
}

fn inert_session(command: &str, stdout: Arc<AsyncMutex<TailBuffer>>) -> BashSession {
    let (terminate, _) = watch::channel(false);
    BashSession {
        command: command.to_string(),
        pid: None,
        stdout,
        stderr: Arc::new(AsyncMutex::new(TailBuffer::default())),
        exit: Arc::new(AsyncMutex::new(None)),
        running: Arc::new(AtomicBool::new(false)),
        terminate,
        supervisor: None,
    }
}

#[cfg(unix)]
async fn wait_for_pid(path: &Path) -> u32 {
    for _ in 0..100 {
        if let Ok(raw) = tokio::fs::read_to_string(path).await {
            if let Ok(pid) = raw.trim().parse() {
                return pid;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("child pid was not written to {}", path.display());
}

#[cfg(unix)]
async fn assert_process_stopped(pid: u32) {
    for _ in 0..100 {
        let alive = std::process::Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if !alive {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("process {pid} survived background-shell cleanup");
}

#[cfg(unix)]
#[tokio::test]
async fn run_then_output_then_kill() {
    let dir = TempDir::new().unwrap();
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    let out = BashOutputTool::new(registry.clone());
    let kill = KillShellTool::new(registry.clone());

    let started = run
        .call(
            &ctx(),
            json!({"command": "i=0; while true; do echo tick$i; i=$((i+1)); sleep 0.05; done"}),
        )
        .await
        .unwrap();
    let shell_id = started["shell_id"].as_str().unwrap().to_string();

    let polled = out
        .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 500}))
        .await
        .unwrap();
    let stdout = polled["stdout"].as_str().unwrap();
    assert!(stdout.contains("tick"), "got stdout: {stdout:?}");
    assert_eq!(polled["running"], true);

    let polled2 = out
        .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 500}))
        .await
        .unwrap();
    assert!(polled2["stdout"].as_str().unwrap().contains("tick"));

    let killed = kill
        .call(&ctx(), json!({"shell_id": &shell_id}))
        .await
        .unwrap();
    assert_eq!(killed["killed"], true);

    let err = out
        .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 0}))
        .await
        .expect_err("should be gone");
    assert!(err.to_string().contains("no shell"));
}

#[cfg(unix)]
#[tokio::test]
async fn output_caps_model_facing_stdout() {
    let registry = BashSessionRegistry::new();
    let shell_id = "model-cap-test".to_string();
    let stdout = Arc::new(AsyncMutex::new(TailBuffer::default()));
    stdout.lock().await.push_chunk(&vec![b'L'; 20_000]);
    registry.inner.write().await.insert(
        shell_id.clone(),
        inert_session("synthetic-large-output", stdout),
    );
    let out = BashOutputTool::new(registry);

    let polled = out
        .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 0}))
        .await
        .unwrap();

    let stdout = polled["stdout"].as_str().unwrap();
    assert_eq!(polled["stdout_truncated"], true);
    assert!(stdout.contains("output truncated"));
    assert!(stdout.contains("FileRead"));
    assert!(stdout.len() < 20_000);
}

#[cfg(unix)]
#[tokio::test]
async fn output_compresses_recognized_command_stdout() {
    let dir = TempDir::new().unwrap();
    std::process::Command::new("git")
        .arg("init")
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn main() {}\n").unwrap();

    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    let out = BashOutputTool::new(registry);
    let started = run
        .call(&ctx(), json!({"command": "git status"}))
        .await
        .unwrap();
    let shell_id = started["shell_id"].as_str().unwrap().to_string();

    let mut polled = out
        .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 1000}))
        .await
        .unwrap();
    for _ in 0..20 {
        if polled["running"] == false {
            break;
        }
        polled = out
            .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 1000}))
            .await
            .unwrap();
    }

    let stdout = polled["stdout"].as_str().unwrap();
    assert!(stdout.contains("git status:"), "{stdout}");
    assert!(stdout.contains("compressed git status"), "{stdout}");
    assert!(stdout.contains("1 untracked"), "{stdout}");
}

#[tokio::test]
async fn run_rejects_empty_command() {
    let dir = TempDir::new().unwrap();
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry);
    let err = run
        .call(&ctx(), json!({"command": "  "}))
        .await
        .expect_err("empty");
    assert!(err.to_string().contains("non-empty"));
}

#[tokio::test]
async fn run_honors_root_abort_before_spawn() {
    let dir = TempDir::new().unwrap();
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry);
    let context = ctx();
    context.abort.abort_with_reason("watchdog stop");

    let error = run
        .call(&context, json!({"command": "echo should-not-run"}))
        .await
        .unwrap_err();

    assert!(matches!(error, AgentError::Aborted(reason) if reason == "watchdog stop"));
}

#[cfg(unix)]
#[tokio::test]
async fn registered_shell_latches_unresolved_external_work() {
    let dir = TempDir::new().unwrap();
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    let kill = KillShellTool::new(registry);
    let context = ctx();

    let started = run
        .call(&context, json!({"command": "sleep 30"}))
        .await
        .unwrap();

    assert!(context.abort.activity().unresolved_external_work());
    kill.call(
        &ctx(),
        json!({"shell_id": started["shell_id"].as_str().unwrap()}),
    )
    .await
    .unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn hard_cancel_while_waiting_for_registry_lock_never_spawns() {
    let dir = TempDir::new().unwrap();
    let marker = dir.path().join("spawned");
    let registry = BashSessionRegistry::new();
    let held = registry.inner.write().await;
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    let command = format!("touch '{}'", marker.display());
    let task = tokio::spawn(async move { run.call(&ctx(), json!({"command": command})).await });

    tokio::task::yield_now().await;
    task.abort();
    let _ = task.await;
    drop(held);
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert!(!marker.exists());
    assert!(registry.is_empty().await);
}

#[cfg(unix)]
#[tokio::test]
async fn dropping_the_registry_kills_registered_process_groups() {
    let dir = TempDir::new().unwrap();
    let pid_file = dir.path().join("descendant.pid");
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    let command = format!(
        "sleep 30 & child=$!; printf %s \"$child\" > '{}'; wait",
        pid_file.display()
    );
    run.call(&ctx(), json!({"command": command})).await.unwrap();
    let pid = wait_for_pid(&pid_file).await;

    drop(run);
    drop(registry);

    assert_process_stopped(pid).await;
}

#[tokio::test]
async fn output_unknown_shell_errors() {
    let registry = BashSessionRegistry::new();
    let out = BashOutputTool::new(registry);
    let err = out
        .call(&ctx(), json!({"shell_id": "bogus"}))
        .await
        .expect_err("not found");
    assert!(err.to_string().contains("no shell"));
}

#[tokio::test]
async fn kill_unknown_shell_errors() {
    let registry = BashSessionRegistry::new();
    let kill = KillShellTool::new(registry);
    let err = kill
        .call(&ctx(), json!({"shell_id": "bogus"}))
        .await
        .expect_err("not found");
    assert!(err.to_string().contains("no shell"));
}

#[cfg(unix)]
#[tokio::test]
async fn output_surfaces_exit_code_after_short_command() {
    let dir = TempDir::new().unwrap();
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    let out = BashOutputTool::new(registry.clone());

    let started = run
        .call(&ctx(), json!({"command": "echo done && exit 7"}))
        .await
        .unwrap();
    let shell_id = started["shell_id"].as_str().unwrap().to_string();

    let polled = out
        .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 1000}))
        .await
        .unwrap();
    if polled["running"] == false {
        assert_eq!(polled["exit_code"], 7);
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let polled2 = out
        .call(&ctx(), json!({"shell_id": &shell_id, "wait_ms": 0}))
        .await
        .unwrap();
    assert_eq!(polled2["running"], false);
    assert_eq!(polled2["exit_code"], 7);
    assert!(
        polled["stdout"].as_str().unwrap().contains("done")
            || polled2["stdout"].as_str().unwrap().contains("done")
    );
}

#[tokio::test]
async fn registry_caps_concurrent_sessions() {
    let dir = TempDir::new().unwrap();
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    for _ in 0..MAX_SESSIONS_PER_REGISTRY {
        registry.inner.write().await.insert(
            random_id(),
            inert_session("noop", Arc::new(AsyncMutex::new(TailBuffer::default()))),
        );
    }
    let err = run
        .call(&ctx(), json!({"command": "echo hi"}))
        .await
        .expect_err("over cap");
    assert!(err.to_string().contains("capped"));
}

#[tokio::test]
async fn run_classified_mutating_output_read_only_kill_mutating() {
    let dir = TempDir::new().unwrap();
    let registry = BashSessionRegistry::new();
    let run = BashRunTool::new(policy_for(dir.path()), registry.clone());
    let out = BashOutputTool::new(registry.clone());
    let kill = KillShellTool::new(registry);
    assert_eq!(run.safety_class(), SafetyClass::Mutating);
    assert_eq!(out.safety_class(), SafetyClass::ReadOnly);
    assert_eq!(kill.safety_class(), SafetyClass::Mutating);
}
