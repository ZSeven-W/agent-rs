//! Receipt-order tool executor (Phase 3 / Task 3.5).
//!
//! When the LLM emits multiple `tool_use` events in one assistant turn,
//! the engine dispatches them concurrently — but the consumer must see
//! `ToolResult` events in **receipt order** (the order the LLM produced
//! them), not in completion order. This module exposes
//! [`ToolExecutor::dispatch`], which wraps `futures::StreamExt::buffered`
//! to provide that semantic.
//!
//! Each dispatched tool runs concurrently up to a configurable max; each
//! `ToolResult` event is emitted in the original Vec order regardless of
//! how the underlying tasks finished.

use std::sync::Arc;
use std::time::Duration;

use futures::channel::mpsc;
use futures::stream::StreamExt;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::stream::{Event, EventStream, TaskEventStream};
use crate::tool::{SafetyClass, ToolRegistry, ToolUseContext};

/// One requested tool call coming out of an LLM assistant turn.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestedToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolExecutor;

/// Give tools that observe `ctx.abort` time to run their own cleanup before
/// the final drop-cancels fallback aborts their Tokio task. This is a global
/// deadline for the whole batch, not a per-tool delay.
const ABORT_DRAIN_GRACE: Duration = Duration::from_secs(7);

/// A mutating tool that is dropped before returning may already have handed
/// work to an OS thread, actor, browser extension, or remote server. Local
/// worker-count quiescence cannot prove that work stopped, so scheduler hosts
/// must keep the turn fail-closed until a human reviews it.
struct UnresolvedEffectOnDrop {
    abort: AbortController,
    armed: bool,
}

impl UnresolvedEffectOnDrop {
    fn new(abort: AbortController) -> Self {
        Self { abort, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for UnresolvedEffectOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.abort.mark_unresolved_external_work();
        }
    }
}

impl ToolExecutor {
    /// Dispatch every tool use concurrently against `registry` using
    /// `ctx` as the shared per-call context. The returned stream
    /// yields one `Event::ToolResult` per request in the **receipt
    /// order** of `tool_uses` (Vec order), regardless of which
    /// underlying invocation completes first.
    ///
    /// `max_concurrent` caps the number of tools that run at the same
    /// time. Pass `usize::MAX` for unbounded concurrency. A common
    /// default is `8`; expensive tools should clamp lower.
    ///
    /// Tool not found in `registry` → emits `ToolResult { ok: false,
    /// output: { "error": "tool 'X' not registered" } }` rather than
    /// failing the stream — the LLM may recover.
    pub fn dispatch(
        tool_uses: Vec<RequestedToolUse>,
        registry: Arc<ToolRegistry>,
        ctx: ToolUseContext,
        max_concurrent: usize,
    ) -> Box<dyn EventStream> {
        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();
        if tool_uses.is_empty() {
            // Drop tx to signal end-of-stream to receiver.
            return Box::new(rx);
        }

        let forward_work = ctx.abort.activity().track_worker();
        let forward = tokio::spawn(async move {
            let _work = forward_work;
            forward(tool_uses, registry, ctx, max_concurrent, tx).await;
        });
        Box::new(TaskEventStream::new(rx, forward, "tool dispatch"))
    }
}

async fn forward(
    tool_uses: Vec<RequestedToolUse>,
    registry: Arc<ToolRegistry>,
    ctx: ToolUseContext,
    max_concurrent: usize,
    tx: mpsc::UnboundedSender<Result<Event, AgentError>>,
) {
    let abort = ctx.abort.clone();
    let registry = registry.clone();
    let ctx = Arc::new(ctx);

    let stream = futures::stream::iter(tool_uses).map(move |req| {
        let registry = registry.clone();
        let ctx = ctx.clone();
        let panic_id = req.id.clone();
        let tool_work = ctx.abort.activity().track_worker();
        // Each tool runs on its OWN spawned task: previously all buffered
        // futures were polled cooperatively on this one task, so a
        // CPU-bound or blocking-sync tool starved every concurrent peer.
        let handle = tokio::spawn(async move {
            let _work = tool_work;
            let RequestedToolUse { id, name, input } = req;
            match registry.get(&name) {
                Some(tool) => {
                    ctx.abort.pulse();
                    let mut unresolved = if !matches!(tool.safety_class(), SafetyClass::ReadOnly) {
                        // Mark before entering user/tool code. A panic, abort,
                        // or lost result must still fail closed for replay.
                        ctx.abort.mark_side_effect_risk();
                        Some(UnresolvedEffectOnDrop::new(ctx.abort.clone()))
                    } else {
                        None
                    };
                    let result = tool.call(&ctx, input).await;
                    if let Some(unresolved) = &mut unresolved {
                        // Returning from the tool is the adapter's declaration
                        // that its mutation reached a terminal response. A
                        // dropped/panicked call keeps the stronger latch set.
                        unresolved.disarm();
                    }
                    ctx.abort.pulse();
                    match result {
                        Ok(output) => Event::ToolResult {
                            id,
                            ok: true,
                            output,
                        },
                        Err(err) => Event::ToolResult {
                            id,
                            ok: false,
                            output: serde_json::json!({ "error": err.to_string() }),
                        },
                    }
                }
                None => Event::ToolResult {
                    id,
                    ok: false,
                    output: serde_json::json!({
                        "error": format!("tool '{name}' not registered"),
                    }),
                },
            }
        });
        async move {
            // Kill the spawned task if this wrapper is dropped (the abort
            // path drops the buffered stream) — preserves the old
            // drop-cancels-tools semantics for ill-behaved tools that
            // ignore `ctx.abort`.
            struct KillOnDrop(Option<tokio::task::JoinHandle<Event>>);
            impl Drop for KillOnDrop {
                fn drop(&mut self) {
                    if let Some(h) = self.0.take() {
                        h.abort();
                    }
                }
            }
            let mut guard = KillOnDrop(Some(handle));
            let joined = guard.0.as_mut().expect("handle present").await;
            guard.0 = None; // completed — nothing left to kill
            joined.unwrap_or_else(|join_err| Event::ToolResult {
                id: panic_id,
                ok: false,
                output: serde_json::json!({
                    "error": format!("tool task failed: {join_err}"),
                }),
            })
        }
    });

    let mut buffered = stream.buffered(max_concurrent.max(1));
    let mut abort_deadline = None;
    loop {
        if let Some(deadline) = abort_deadline {
            match tokio::time::timeout_at(deadline, buffered.next()).await {
                Ok(Some(event)) => {
                    if tx.unbounded_send(Ok(event)).is_err() {
                        return;
                    }
                }
                Ok(None) | Err(_) => {
                    let _ = tx.unbounded_send(Err(AgentError::Aborted(
                        abort
                            .reason()
                            .unwrap_or_else(|| "tool dispatch aborted".into()),
                    )));
                    return;
                }
            }
        } else {
            tokio::select! {
                biased;
                _ = abort.cancelled() => {
                    // Keep polling the tool wrappers while their cooperative
                    // abort cleanup runs. The wrapper KillOnDrop remains the
                    // bounded hard-stop fallback when this deadline expires.
                    abort_deadline = Some(tokio::time::Instant::now() + ABORT_DRAIN_GRACE);
                }
                next = buffered.next() => {
                    let Some(event) = next else { break };
                    if tx.unbounded_send(Ok(event)).is_err() {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::StreamExt;

    use super::*;
    use crate::tool::Tool;

    /// A tool whose `call` sleeps for a configurable duration before
    /// returning. Used to scramble completion order.
    #[derive(Debug)]
    struct SleepyTool {
        name: String,
        sleep_ms: u64,
        completed: Arc<AtomicUsize>,
    }

    #[derive(Debug)]
    struct PendingTool {
        started: Arc<tokio::sync::Notify>,
        dropped: Arc<AtomicBool>,
    }

    #[async_trait]
    impl Tool for PendingTool {
        fn name(&self) -> &str {
            "pending"
        }
        fn description(&self) -> &str {
            "never completes"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            struct DropProbe(Arc<AtomicBool>);
            impl Drop for DropProbe {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let _probe = DropProbe(self.dropped.clone());
            self.started.notify_one();
            futures::future::pending::<()>().await;
            unreachable!()
        }
    }

    #[async_trait]
    impl Tool for SleepyTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "test tool that sleeps then returns"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            tokio::time::sleep(Duration::from_millis(self.sleep_ms)).await;
            let order = self.completed.fetch_add(1, Ordering::SeqCst);
            Ok(serde_json::json!({"name": self.name, "completion_order": order}))
        }
    }

    fn build_registry(specs: &[(&str, u64, Arc<AtomicUsize>)]) -> Arc<ToolRegistry> {
        let mut r = ToolRegistry::new();
        for (name, sleep_ms, completed) in specs {
            r.register(Arc::new(SleepyTool {
                name: (*name).into(),
                sleep_ms: *sleep_ms,
                completed: completed.clone(),
            }));
        }
        Arc::new(r)
    }

    #[tokio::test]
    async fn dropping_dispatch_stream_hard_cancels_the_tool_task() {
        let started = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(PendingTool {
            started: started.clone(),
            dropped: dropped.clone(),
        }));
        let ctx = ToolUseContext::new("/tmp");
        let activity = ctx.abort.activity();
        let stream = ToolExecutor::dispatch(
            vec![RequestedToolUse {
                id: "p1".into(),
                name: "pending".into(),
                input: serde_json::json!({}),
            }],
            Arc::new(registry),
            ctx,
            1,
        );
        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("tool started");
        drop(stream);
        tokio::time::timeout(Duration::from_secs(1), activity.wait_for_quiescence())
            .await
            .expect("dispatch reached quiescence");
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(activity.active_workers(), 0);
        assert!(
            activity.unresolved_external_work(),
            "dropping an unclassified tool cannot prove its external work stopped"
        );
    }

    #[tokio::test]
    async fn yields_in_receipt_order_despite_scrambled_completion() {
        // 3 tools, completion order will be [B, C, A] but receipt order
        // (dispatch Vec order) is [A, B, C]. Expect ToolResult events
        // for A, B, C in that exact order.
        let completed = Arc::new(AtomicUsize::new(0));
        let registry = build_registry(&[
            ("A", 50, completed.clone()),
            ("B", 5, completed.clone()),
            ("C", 25, completed.clone()),
        ]);

        let tool_uses = vec![
            RequestedToolUse {
                id: "tu_a".into(),
                name: "A".into(),
                input: serde_json::json!({}),
            },
            RequestedToolUse {
                id: "tu_b".into(),
                name: "B".into(),
                input: serde_json::json!({}),
            },
            RequestedToolUse {
                id: "tu_c".into(),
                name: "C".into(),
                input: serde_json::json!({}),
            },
        ];

        let ctx = ToolUseContext::new("/tmp");
        let mut stream = ToolExecutor::dispatch(tool_uses, registry, ctx, 8);
        let mut emitted_ids = Vec::new();
        while let Some(item) = stream.next().await {
            if let Ok(Event::ToolResult { id, ok, .. }) = item {
                assert!(ok);
                emitted_ids.push(id);
            }
        }
        assert_eq!(emitted_ids, vec!["tu_a", "tu_b", "tu_c"]);
        // Verify completion was actually scrambled (B finished first).
        // Each SleepyTool incremented the same counter — final value 3.
        assert_eq!(completed.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn missing_tool_emits_error_result() {
        let registry = Arc::new(ToolRegistry::new());
        let tool_uses = vec![RequestedToolUse {
            id: "tu_x".into(),
            name: "ghost".into(),
            input: serde_json::json!({}),
        }];
        let ctx = ToolUseContext::new("/tmp");
        let mut stream = ToolExecutor::dispatch(tool_uses, registry, ctx, 4);
        let item = stream.next().await.unwrap().unwrap();
        match item {
            Event::ToolResult { id, ok, output } => {
                assert_eq!(id, "tu_x");
                assert!(!ok);
                assert!(output["error"].as_str().unwrap().contains("'ghost'"));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_input_yields_empty_stream() {
        let registry = Arc::new(ToolRegistry::new());
        let ctx = ToolUseContext::new("/tmp");
        let mut stream = ToolExecutor::dispatch(vec![], registry, ctx, 4);
        assert!(stream.next().await.is_none());
    }

    #[derive(Debug)]
    struct ErroringTool;

    #[async_trait]
    impl Tool for ErroringTool {
        fn name(&self) -> &str {
            "boom"
        }
        fn description(&self) -> &str {
            "always errors"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            Err(AgentError::other("intentional failure"))
        }
    }

    #[tokio::test]
    async fn tool_error_becomes_failed_tool_result() {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(ErroringTool));
        let registry = Arc::new(r);

        let tool_uses = vec![RequestedToolUse {
            id: "tu_e".into(),
            name: "boom".into(),
            input: serde_json::json!({}),
        }];
        let ctx = ToolUseContext::new("/tmp");
        let activity = ctx.abort.activity();
        let mut stream = ToolExecutor::dispatch(tool_uses, registry, ctx, 4);
        let item = stream.next().await.unwrap().unwrap();
        match item {
            Event::ToolResult { id, ok, output } => {
                assert_eq!(id, "tu_e");
                assert!(!ok);
                assert!(output["error"]
                    .as_str()
                    .unwrap()
                    .contains("intentional failure"));
            }
            other => panic!("expected failed ToolResult, got {other:?}"),
        }
        assert!(
            !activity.unresolved_external_work(),
            "a tool that returned reached its declared terminal boundary"
        );
    }

    #[tokio::test]
    async fn abort_terminates_dispatch() {
        let completed = Arc::new(AtomicUsize::new(0));
        let registry = build_registry(&[
            ("slow1", 200, completed.clone()),
            ("slow2", 200, completed.clone()),
        ]);
        let tool_uses = vec![
            RequestedToolUse {
                id: "tu_1".into(),
                name: "slow1".into(),
                input: serde_json::json!({}),
            },
            RequestedToolUse {
                id: "tu_2".into(),
                name: "slow2".into(),
                input: serde_json::json!({}),
            },
        ];
        let ctx = ToolUseContext::new("/tmp");
        let abort = ctx.abort.clone();

        let stream_handle = tokio::spawn(async move {
            let mut stream = ToolExecutor::dispatch(tool_uses, registry, ctx, 4);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        // Cancel before tools complete.
        tokio::time::sleep(Duration::from_millis(30)).await;
        abort.abort_with_reason("user cancel");

        let items = stream_handle.await.unwrap();
        // Should have at least one Aborted error item.
        assert!(
            items
                .iter()
                .any(|i| matches!(i, Err(AgentError::Aborted(_)))),
            "expected Aborted error in stream, got {items:?}"
        );
    }
}
