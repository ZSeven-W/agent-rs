//! Full multi-turn query loop with phase machine (Phase 3 / Task 3.4).
//!
//! Walks the conversation through this state graph:
//!
//! ```text
//! Start → Streaming → ToolDispatch → ToolCollecting → YieldingResult → Streaming → ... → Done
//! ```
//!
//! - **Streaming** — call provider, forward TextDelta/Thinking/Usage,
//!   collect ToolUse blocks, track `stop_reason`.
//! - **ToolDispatch** — for each pending tool_use, run permission +
//!   hooks; failed checks emit a synthetic ToolResult (ok=false).
//! - **ToolCollecting** — drive [`ToolExecutor`] (batch I) so the
//!   surviving tool_uses run concurrently and yield in receipt order.
//! - **YieldingResult** — push a User message carrying every
//!   tool_result block, ready to feed back into the next streaming turn.
//! - **Done** — provider's stop_reason was a terminal value (end_turn,
//!   stop_sequence, max_tokens) and no tool_uses remain.

#![allow(clippy::result_large_err)]
// PermissionDecision and the assistant ContentBlock collection are
// intentionally large — they carry rule + reason diagnostics needed by
// the host UI. The multi-turn loop already crosses tokio task
// boundaries via mpsc, so a Box on the Ok variant doesn't materially
// change runtime cost; consistent with `permission/external_queue.rs`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use futures::channel::mpsc;
use futures::StreamExt;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::file_cache::FileStateCache;
use crate::hook::{HookEvent, HookOutcome, HookRunner};
use crate::message::{ContentBlock, Header, Message, MessageStore, ToolResultContent};
use crate::permission::{PermissionDecision, PermissionManager};
use crate::provider::{Provider, StreamRequest};
use crate::stream::{Event, EventStream, RequestedToolUse, ResultData, ToolExecutor};
use crate::tool::{ToolRegistry, ToolUseContext};

/// Loop state at any point during a turn cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Start,
    Streaming,
    ToolDispatch,
    ToolCollecting,
    YieldingResult,
    Done,
}

/// Transitions the loop can fire. Phase 3 batch J ships `ToolUse` and
/// `StopHook`; the remaining variants are reserved for later phases —
/// they exist now (behind `#[non_exhaustive]`) so Phase 4+ work can
/// add behavior without a SemVer break.
///
/// **Reserved variants — not produced by the loop in Phase 3:**
/// - [`Self::CollapseDrain`] — Phase 4: context-window saturation
///   triggers sliding-window trim.
/// - [`Self::ReactiveCompact`] — Phase 4: token-budget-driven
///   transcript summarization.
/// - [`Self::MaxOutputEscalate`] — Phase 5+: provider's `max_tokens`
///   cap hit, split the assistant message.
/// - [`Self::MaxOutputMultiTurn`] — Phase 5+: continuation of
///   [`Self::MaxOutputEscalate`] across turns.
///
/// External code should pattern-match with a catch-all so the
/// reserved variants don't break when implemented.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Transition {
    /// Provider emitted tool_use blocks; dispatch them this turn.
    ToolUse(Vec<RequestedToolUse>),
    /// A hook returned [`HookOutcome::Block`]; abort the turn.
    StopHook { event: String, code: i32 },
    /// **Reserved (Phase 4)** — context window saturated; sliding-window trim.
    CollapseDrain,
    /// **Reserved (Phase 4)** — token budget hit; rewrite transcript to summary.
    ReactiveCompact,
    /// **Reserved (Phase 5+)** — provider's `max_tokens` cap reached; split.
    MaxOutputEscalate,
    /// **Reserved (Phase 5+)** — continuation of [`Self::MaxOutputEscalate`].
    MaxOutputMultiTurn,
}

/// Full multi-turn query driver. Construct with the builder, then call
/// [`run`](Self::run). Returns a stream that yields the same `Event`
/// vocabulary as a single-turn provider — TextDelta / Thinking /
/// ToolUse / ToolResult / Usage / Result — but covers as many provider
/// turns as the conversation requires before reaching `Done`.
#[derive(Debug)]
pub struct QueryLoop {
    pub provider: Arc<dyn Provider>,
    pub tools: Arc<ToolRegistry>,
    pub permissions: Arc<PermissionManager>,
    pub hooks: Arc<HookRunner>,
    pub store: Arc<Mutex<MessageStore>>,
    pub model: String,
    pub system: Option<String>,
    pub max_output_tokens: u32,
    pub max_concurrent_tools: usize,
    /// Hard cap on assistant turns to prevent runaway loops. Defaults
    /// to 16 in [`Self::builder`].
    pub max_iterations: usize,
    /// Working directory threaded into every [`ToolUseContext`].
    pub cwd: PathBuf,
    /// Shared file state cache threaded into every [`ToolUseContext`].
    pub file_cache: Arc<FileStateCache>,
}

impl QueryLoop {
    pub fn builder(provider: Arc<dyn Provider>, model: impl Into<String>) -> QueryLoopBuilder {
        QueryLoopBuilder::new(provider, model)
    }

    /// Run the loop. Returns a stream of `Event`s that completes after
    /// the conversation reaches `Phase::Done`.
    pub async fn run(
        self,
        user_msg: impl Into<String>,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        // Push user message before spawning so callers see it in the
        // store synchronously.
        let user_message = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text {
                text: user_msg.into(),
            }],
        };
        {
            let mut store = self
                .store
                .lock()
                .map_err(|_| AgentError::other("query store lock poisoned"))?;
            store.push(user_message)?;
        }

        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();
        tokio::spawn(drive(self, tx, abort));
        Ok(Box::new(rx))
    }
}

#[derive(Debug)]
pub struct QueryLoopBuilder {
    provider: Arc<dyn Provider>,
    model: String,
    tools: Option<Arc<ToolRegistry>>,
    permissions: Option<Arc<PermissionManager>>,
    hooks: Option<Arc<HookRunner>>,
    store: Option<Arc<Mutex<MessageStore>>>,
    system: Option<String>,
    max_output_tokens: u32,
    max_concurrent_tools: usize,
    max_iterations: usize,
    cwd: PathBuf,
    file_cache: Option<Arc<FileStateCache>>,
}

impl QueryLoopBuilder {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            tools: None,
            permissions: None,
            hooks: None,
            store: None,
            system: None,
            max_output_tokens: 4096,
            max_concurrent_tools: 8,
            max_iterations: 16,
            cwd: PathBuf::from("."),
            file_cache: None,
        }
    }

    pub fn tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn permissions(mut self, p: Arc<PermissionManager>) -> Self {
        self.permissions = Some(p);
        self
    }
    pub fn hooks(mut self, h: Arc<HookRunner>) -> Self {
        self.hooks = Some(h);
        self
    }
    pub fn store(mut self, s: Arc<Mutex<MessageStore>>) -> Self {
        self.store = Some(s);
        self
    }
    pub fn system(mut self, s: impl Into<String>) -> Self {
        self.system = Some(s.into());
        self
    }
    pub fn max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = n;
        self
    }
    pub fn max_concurrent_tools(mut self, n: usize) -> Self {
        self.max_concurrent_tools = n;
        self
    }
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    pub fn cwd(mut self, p: impl Into<PathBuf>) -> Self {
        self.cwd = p.into();
        self
    }
    pub fn file_cache(mut self, c: Arc<FileStateCache>) -> Self {
        self.file_cache = Some(c);
        self
    }

    pub fn build(self) -> QueryLoop {
        let file_cache = self.file_cache.unwrap_or_else(|| {
            Arc::new(FileStateCache::new(
                std::num::NonZeroUsize::new(64).unwrap(),
                8 * 1024 * 1024,
            ))
        });
        QueryLoop {
            provider: self.provider,
            tools: self.tools.unwrap_or_else(|| Arc::new(ToolRegistry::new())),
            permissions: self
                .permissions
                .unwrap_or_else(|| Arc::new(PermissionManager::new())),
            hooks: self.hooks.unwrap_or_else(|| Arc::new(HookRunner::new())),
            store: self
                .store
                .unwrap_or_else(|| Arc::new(Mutex::new(MessageStore::new()))),
            model: self.model,
            system: self.system,
            max_output_tokens: self.max_output_tokens,
            max_concurrent_tools: self.max_concurrent_tools,
            max_iterations: self.max_iterations,
            cwd: self.cwd,
            file_cache,
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

async fn drive(
    qloop: QueryLoop,
    tx: mpsc::UnboundedSender<Result<Event, AgentError>>,
    abort: AbortController,
) {
    let mut iter = 0usize;
    let mut final_result = ResultData {
        stop_reason: None,
        model: None,
        metadata: Default::default(),
    };

    loop {
        if abort.is_aborted() {
            let _ = tx.unbounded_send(Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            )));
            return;
        }

        if iter >= qloop.max_iterations {
            let _ = tx.unbounded_send(Err(AgentError::other(format!(
                "QueryLoop hit max_iterations ({})",
                qloop.max_iterations
            ))));
            return;
        }
        iter += 1;

        // ----- Streaming phase -----
        let messages = match snapshot(&qloop.store) {
            Ok(m) => m,
            Err(e) => {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        };
        let mut req = StreamRequest::new(qloop.model.clone(), messages)
            .with_max_output_tokens(qloop.max_output_tokens);
        if let Some(s) = &qloop.system {
            req = req.with_system(s.clone());
        }

        let upstream = match qloop.provider.stream(req, abort.clone()).await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        };

        let TurnSummary {
            assistant_blocks,
            pending_tool_uses,
            stop_reason,
            model,
        } = match consume_turn(upstream, &tx, &abort).await {
            Ok(s) => s,
            Err(()) => return, // already-emitted error or aborted
        };
        if let Some(m) = model {
            final_result.model = Some(m);
        }
        if let Some(r) = stop_reason.clone() {
            final_result.stop_reason = Some(r);
        }

        // Push assistant message with everything we collected.
        let assistant = Message::Assistant {
            header: child_header(&qloop.store),
            content: assistant_blocks,
        };
        if let Err(e) = push(&qloop.store, assistant) {
            let _ = tx.unbounded_send(Err(e));
            return;
        }

        // No tool_uses → we're done.
        if pending_tool_uses.is_empty() {
            break;
        }

        // ----- ToolDispatch phase: permission + BeforeToolUse hooks -----
        let mut surviving = Vec::with_capacity(pending_tool_uses.len());
        let mut tool_results: Vec<(String, bool, serde_json::Value)> = Vec::new();
        for tu in pending_tool_uses {
            // 1. Permission.
            let decision = qloop.permissions.evaluate(&tu.name, &tu.input, None);
            match &decision {
                PermissionDecision::Allow(_) => {}
                PermissionDecision::Ask(ask) => {
                    let _ = qloop
                        .hooks
                        .run(&HookEvent::OnPermissionRequest {
                            tool: tu.name.clone(),
                            input: tu.input.clone(),
                        })
                        .await;
                    let synthetic = synthetic_tool_result(
                        &tu,
                        false,
                        format!(
                            "Tool '{}' requires manual approval and no external queue is wired (ask: {}).",
                            tu.name, ask.message_text
                        ),
                    );
                    forward_tool_result(&tx, synthetic.clone());
                    tool_results.push((tu.id.clone(), synthetic.0.ok, synthetic.0.output));
                    let _ = qloop
                        .hooks
                        .run(&HookEvent::OnPermissionDenied {
                            tool: tu.name.clone(),
                            reason: ask.message_text.clone(),
                        })
                        .await;
                    continue;
                }
                PermissionDecision::Deny(deny) => {
                    let synthetic = synthetic_tool_result(
                        &tu,
                        false,
                        format!("Tool '{}' denied: {}", tu.name, deny.message_text),
                    );
                    forward_tool_result(&tx, synthetic.clone());
                    tool_results.push((tu.id.clone(), synthetic.0.ok, synthetic.0.output));
                    let _ = qloop
                        .hooks
                        .run(&HookEvent::OnPermissionDenied {
                            tool: tu.name.clone(),
                            reason: deny.message_text.clone(),
                        })
                        .await;
                    continue;
                }
            }

            // 2. BeforeToolUse hook.
            let hook_outcome = qloop
                .hooks
                .run(&HookEvent::BeforeToolUse {
                    tool: tu.name.clone(),
                    input: tu.input.clone(),
                })
                .await;
            if matches!(hook_outcome, HookOutcome::Block) {
                let synthetic = synthetic_tool_result(
                    &tu,
                    false,
                    format!("Tool '{}' blocked by BeforeToolUse hook", tu.name),
                );
                forward_tool_result(&tx, synthetic.clone());
                tool_results.push((tu.id.clone(), synthetic.0.ok, synthetic.0.output));
                continue;
            }

            let _ = qloop
                .hooks
                .run(&HookEvent::OnPermissionAllowed {
                    tool: tu.name.clone(),
                })
                .await;
            surviving.push(tu);
        }

        // ----- ToolCollecting phase: dispatch survivors via executor -----
        if !surviving.is_empty() {
            let ctx = ToolUseContext {
                cwd: qloop.cwd.clone(),
                abort: abort.clone(),
                file_cache: qloop.file_cache.clone(),
                permissions: qloop.permissions.clone(),
                hooks: qloop.hooks.clone(),
            };
            let mut exec_stream = ToolExecutor::dispatch(
                surviving.clone(),
                qloop.tools.clone(),
                ctx,
                qloop.max_concurrent_tools,
            );
            while let Some(item) = exec_stream.next().await {
                match item {
                    Ok(event @ Event::ToolResult { .. }) => {
                        // Capture for the user-message block we'll push next.
                        if let Event::ToolResult { id, ok, output } = &event {
                            tool_results.push((id.clone(), *ok, output.clone()));
                        }
                        // Fire AfterToolUse / PostToolUseFailure hook.
                        if let Some(matching) = surviving.iter().find(|s| match &event {
                            Event::ToolResult { id, .. } => &s.id == id,
                            _ => false,
                        }) {
                            if let Event::ToolResult { ok, output, .. } = &event {
                                if *ok {
                                    let _ = qloop
                                        .hooks
                                        .run(&HookEvent::AfterToolUse {
                                            tool: matching.name.clone(),
                                            input: matching.input.clone(),
                                            output: output.clone(),
                                            ok: *ok,
                                        })
                                        .await;
                                } else {
                                    let _ = qloop
                                        .hooks
                                        .run(&HookEvent::PostToolUseFailure {
                                            tool: matching.name.clone(),
                                            input: matching.input.clone(),
                                            error: output
                                                .get("error")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("unknown")
                                                .into(),
                                        })
                                        .await;
                                }
                            }
                        }
                        if tx.unbounded_send(Ok(event)).is_err() {
                            return;
                        }
                    }
                    Ok(other) => {
                        if tx.unbounded_send(Ok(other)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(e));
                        return;
                    }
                }
            }
        }

        // ----- YieldingResult phase: feed tool_results back to provider -----
        let next_user = Message::User {
            header: child_header(&qloop.store),
            content: tool_results
                .into_iter()
                .map(|(id, ok, output)| ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: ToolResultContent::Text(output.to_string()),
                    is_error: !ok,
                })
                .collect(),
        };
        if let Err(e) = push(&qloop.store, next_user) {
            let _ = tx.unbounded_send(Err(e));
            return;
        }

        // After tool dispatch, the provider needs another turn to
        // observe the tool_results. So we re-loop on `tool_use`,
        // `max_tokens`, `stop_sequence`, or any unrecognized stop
        // reason — only `end_turn` is treated as terminal here.
        // Phase 5+ may add explicit handling for `max_tokens`
        // (MaxOutputEscalate transition) and `stop_sequence` (no
        // continuation expected); for now the conservative choice is
        // to give the provider another shot. Termination is still
        // bounded by `max_iterations`.
        if let Some(reason) = &stop_reason {
            if reason == "end_turn" {
                break;
            }
        }
    }

    // ----- Done phase -----
    let _ = tx.unbounded_send(Ok(Event::Result { data: final_result }));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct TurnSummary {
    assistant_blocks: Vec<ContentBlock>,
    pending_tool_uses: Vec<RequestedToolUse>,
    stop_reason: Option<String>,
    model: Option<String>,
}

async fn consume_turn(
    mut upstream: Box<dyn EventStream>,
    tx: &mpsc::UnboundedSender<Result<Event, AgentError>>,
    abort: &AbortController,
) -> Result<TurnSummary, ()> {
    let mut summary = TurnSummary::default();
    let mut accumulated_text = String::new();
    let mut accumulated_thinking: Option<String> = None;

    while let Some(item) = upstream.next().await {
        if abort.is_aborted() {
            let _ = tx.unbounded_send(Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            )));
            return Err(());
        }
        match item {
            Ok(event) => match event {
                Event::TextDelta { delta } => {
                    accumulated_text.push_str(&delta);
                    if tx
                        .unbounded_send(Ok(Event::TextDelta { delta }))
                        .is_err()
                    {
                        return Err(());
                    }
                }
                Event::Thinking { delta } => {
                    accumulated_thinking
                        .get_or_insert_with(String::new)
                        .push_str(&delta);
                    if tx.unbounded_send(Ok(Event::Thinking { delta })).is_err() {
                        return Err(());
                    }
                }
                Event::ToolUse { id, name, input } => {
                    summary.pending_tool_uses.push(RequestedToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });
                    if tx
                        .unbounded_send(Ok(Event::ToolUse { id, name, input }))
                        .is_err()
                    {
                        return Err(());
                    }
                }
                Event::Result { data } => {
                    summary.stop_reason = data.stop_reason.clone();
                    summary.model = data.model.clone();
                    // Don't forward the per-turn Result yet — the driver
                    // emits a single final Result after the whole loop
                    // settles.
                }
                Event::Usage { .. }
                | Event::Error { .. }
                | Event::ToolResult { .. }
                | Event::Unknown => {
                    if tx.unbounded_send(Ok(event)).is_err() {
                        return Err(());
                    }
                }
            },
            Err(e) => {
                let _ = tx.unbounded_send(Err(e));
                return Err(());
            }
        }
    }

    if !accumulated_text.is_empty() {
        summary
            .assistant_blocks
            .push(ContentBlock::Text { text: accumulated_text });
    }
    if let Some(thinking) = accumulated_thinking {
        summary.assistant_blocks.push(ContentBlock::Thinking {
            thinking,
            signature: None,
        });
    }
    for tu in &summary.pending_tool_uses {
        summary.assistant_blocks.push(ContentBlock::ToolUse {
            id: tu.id.clone(),
            name: tu.name.clone(),
            input: tu.input.clone(),
        });
    }
    Ok(summary)
}

fn snapshot(store: &Arc<Mutex<MessageStore>>) -> Result<Vec<Message>, AgentError> {
    let s = store
        .lock()
        .map_err(|_| AgentError::other("query store lock poisoned"))?;
    Ok(s.iter().cloned().collect())
}

fn push(store: &Arc<Mutex<MessageStore>>, msg: Message) -> Result<(), AgentError> {
    let mut s = store
        .lock()
        .map_err(|_| AgentError::other("query store lock poisoned"))?;
    s.push(msg)
}

fn child_header(store: &Arc<Mutex<MessageStore>>) -> Header {
    let parent = store
        .lock()
        .ok()
        .and_then(|s| s.iter().last().map(|m| m.uuid()));
    match parent {
        Some(p) => Header::child_of(p),
        None => Header::new(),
    }
}

#[derive(Debug, Clone)]
struct SyntheticToolResult(ToolResultEcho);

#[derive(Debug, Clone)]
struct ToolResultEcho {
    id: String,
    ok: bool,
    output: serde_json::Value,
}

fn synthetic_tool_result(
    tu: &RequestedToolUse,
    ok: bool,
    error_text: String,
) -> SyntheticToolResult {
    SyntheticToolResult(ToolResultEcho {
        id: tu.id.clone(),
        ok,
        output: serde_json::json!({"error": error_text}),
    })
}

fn forward_tool_result(
    tx: &mpsc::UnboundedSender<Result<Event, AgentError>>,
    synthetic: SyntheticToolResult,
) {
    let _ = tx.unbounded_send(Ok(Event::ToolResult {
        id: synthetic.0.id,
        ok: synthetic.0.ok,
        output: synthetic.0.output,
    }));
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures::StreamExt;

    use super::*;
    use crate::permission::{PermissionMode, RuleSource};
    use crate::testing::MockProvider;
    use crate::tool::Tool;

    #[derive(Debug)]
    struct EchoTool {
        name: String,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "echo input as output"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(input)
        }
    }

    fn echo_registry(name: &str, calls: Arc<AtomicUsize>) -> Arc<ToolRegistry> {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(EchoTool {
            name: name.into(),
            calls,
        }));
        Arc::new(r)
    }

    #[tokio::test]
    async fn single_turn_text_only() {
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta { delta: "hi ".into() },
            Event::TextDelta { delta: "there".into() },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    model: Some("mock-1".into()),
                    metadata: Default::default(),
                },
            },
        ]]));
        let qloop = QueryLoop::builder(provider, "mock-1")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .build();
        let mut stream = qloop.run("hello", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // 2 TextDelta + 1 final Result.
        assert_eq!(events.len(), 3);
        assert!(matches!(events.last(), Some(Event::Result { .. })));
    }

    #[tokio::test]
    async fn two_turn_tool_use_loop() {
        // Turn 1: assistant emits a tool_use + Result(stop_reason="tool_use").
        // Turn 2: after we feed tool_result back, assistant emits text +
        // Result(stop_reason="end_turn") → loop terminates.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::ToolUse {
                    id: "tu_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"value": 42}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ],
            vec![
                Event::TextDelta {
                    delta: "got 42".into(),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));

        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(
            PermissionManager::new()
                .allow(RuleSource::User, "echo"),
        );

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .build();

        let store = qloop.store.clone();
        let mut stream = qloop.run("call echo", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Echo tool was called exactly once.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Stream emitted ToolUse + ToolResult + TextDelta + Result.
        assert!(events.iter().any(|e| matches!(e, Event::ToolUse { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolResult { ok: true, .. })));
        assert!(events.iter().any(|e| matches!(e, Event::TextDelta { .. })));
        assert!(matches!(events.last(), Some(Event::Result { .. })));

        // Store: User → Assistant(tool_use) → User(tool_result) → Assistant(text).
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert_eq!(snap.len(), 4);
        assert!(matches!(snap[0], Message::User { .. }));
        assert!(matches!(snap[1], Message::Assistant { .. }));
        assert!(matches!(snap[2], Message::User { .. }));
        assert!(matches!(snap[3], Message::Assistant { .. }));
    }

    #[tokio::test]
    async fn permission_deny_synthesizes_failed_tool_result_and_does_not_dispatch() {
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::ToolUse {
                    id: "tu_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ],
            vec![Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            }],
        ]));
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(
            PermissionManager::new()
                .deny(RuleSource::Policy, "echo"),
        );

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Echo tool was NOT called — denied at permission step.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        // Stream still has a synthetic ToolResult ok=false.
        assert!(events.iter().any(|e| matches!(
            e,
            Event::ToolResult {
                ok: false,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn before_tool_use_block_hook_skips_dispatch() {
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::ToolUse {
                    id: "tu_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ],
            vec![Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            }],
        ]));
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(
            PermissionManager::new()
                .allow(RuleSource::User, "echo"),
        );

        let blocking_hook = Arc::new(crate::hook::RustHookHandler::new(
            "blocker",
            |event| match event {
                HookEvent::BeforeToolUse { .. } => HookOutcome::Block,
                _ => HookOutcome::Ok,
            },
        ));
        let mut hooks = HookRunner::new();
        hooks.register(blocking_hook);

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .hooks(Arc::new(hooks))
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        while let Some(_item) = stream.next().await {}
        // Hook blocked dispatch; echo was not called.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn max_iterations_terminates_loop() {
        // Provider scripts an infinite tool_use loop (every turn requests
        // a tool). max_iterations cap should halt the loop with an
        // error.
        let mut turns = Vec::new();
        for i in 0..32 {
            turns.push(vec![
                Event::ToolUse {
                    id: format!("tu_{i}"),
                    name: "echo".into(),
                    input: serde_json::json!({"i": i}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ]);
        }
        let provider = Arc::new(MockProvider::with_turns(turns));
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(
            PermissionManager::new()
                .allow(RuleSource::User, "echo"),
        );

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .max_iterations(3)
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut got_max_err = false;
        while let Some(item) = stream.next().await {
            if let Err(AgentError::Other(msg)) = item {
                if msg.contains("max_iterations") {
                    got_max_err = true;
                }
            }
        }
        assert!(got_max_err, "expected max_iterations error");
    }
}
