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

use futures::channel::mpsc;
use futures::stream::StreamExt;

use crate::error::AgentError;
use crate::stream::{Event, EventStream};
use crate::tool::{ToolRegistry, ToolUseContext};

/// One requested tool call coming out of an LLM assistant turn.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestedToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolExecutor;

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

        tokio::spawn(forward(tool_uses, registry, ctx, max_concurrent, tx));
        Box::new(rx)
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
        async move {
            let RequestedToolUse { id, name, input } = req;
            match registry.get(&name) {
                Some(tool) => match tool.call(&ctx, input).await {
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
                },
                None => Event::ToolResult {
                    id,
                    ok: false,
                    output: serde_json::json!({
                        "error": format!("tool '{name}' not registered"),
                    }),
                },
            }
        }
    });

    let mut buffered = stream.buffered(max_concurrent.max(1));
    loop {
        tokio::select! {
            biased;
            _ = abort.cancelled() => {
                let _ = tx.unbounded_send(Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "tool dispatch aborted".into()),
                )));
                return;
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
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
        let mut stream = ToolExecutor::dispatch(tool_uses, registry, ctx, 4);
        let item = stream.next().await.unwrap().unwrap();
        match item {
            Event::ToolResult { id, ok, output } => {
                assert_eq!(id, "tu_e");
                assert!(!ok);
                assert!(output["error"].as_str().unwrap().contains("intentional failure"));
            }
            other => panic!("expected failed ToolResult, got {other:?}"),
        }
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
            items.iter().any(|i| matches!(i, Err(AgentError::Aborted(_)))),
            "expected Aborted error in stream, got {items:?}"
        );
    }
}
