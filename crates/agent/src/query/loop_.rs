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
use crate::compact::{
    apply_compaction_to_store, apply_microcompact_to_store, compact_with_hooks,
    estimate_text_tokens, estimate_tokens, promote_to_store, AutoCompactReason, AutoCompactState,
    CompactError, CompactTrigger, CompactWithHooksRequest, MicrocompactConfig,
    PartialCompactDirection, SessionMemoryStore,
};
use crate::error::AgentError;
use crate::file_cache::FileStateCache;
use crate::hook::{HookEvent, HookOutcome, HookRunner};
use crate::message::{ContentBlock, Header, Message, MessageStore, ToolResultContent};
use crate::permission::{PermissionDecision, PermissionManager};
use crate::provider::{Provider, StreamRequest, ToolDefinition};
use crate::stream::{Event, EventStream, RequestedToolUse, ResultData, ToolExecutor};
use crate::tool::{ToolRegistry, ToolUseContext};

/// Default declared context window when the caller has no model-specific
/// information. Matches Anthropic's `claude-3-5-sonnet` family. Callers
/// targeting smaller windows should override via the builder.
pub const DEFAULT_MODEL_MAX_TOKENS: u32 = 200_000;

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
    /// Shared message store. **Single-driver invariant**: while
    /// [`Self::run`] is in progress, only the spawned drive task may
    /// mutate the store. External code may take a read snapshot via
    /// `store.lock()` but MUST NOT push, pop, or otherwise modify the
    /// store, because reactive auto-compaction relies on the snapshot
    /// taken inside `maybe_auto_compact` matching the live store when
    /// the compaction result is applied. Concurrent mutations would
    /// land before the boundary marker and silently escape the
    /// summary. If you need to drive multiple parallel turns over the
    /// same conversation history, construct a new `QueryLoop` per
    /// turn with the same `Arc` and serialize the `run()` calls.
    pub store: Arc<Mutex<MessageStore>>,
    pub model: String,
    pub system: Option<String>,
    pub max_output_tokens: u32,
    pub max_concurrent_tools: usize,
    /// Sub-agent nesting depth of THIS loop (0 = root). Threaded into
    /// every [`ToolUseContext`] so the Task tool's recursion guard holds
    /// across the child loop's `tokio::spawn`.
    pub task_depth: usize,
    /// Optional mid-turn steering inbox: user messages injected while a
    /// multi-step turn runs. Drained at the top of each loop iteration and
    /// appended to the store before the next provider round-trip. Shared
    /// (Arc<Mutex<Receiver>>) because the loop is rebuilt per turn while the
    /// channel persists on the host engine.
    pub steer: Option<Arc<std::sync::Mutex<mpsc::UnboundedReceiver<Vec<ContentBlock>>>>>,
    /// Optional cap on assistant turns. Defaults to `usize::MAX` (UNBOUNDED) in
    /// [`Self::builder`] — the loop already stops naturally on a turn with no
    /// tool calls. Set a finite value only to force a runaway backstop.
    pub max_iterations: usize,
    /// How many times to retry a transient API failure (rate limit / 5xx /
    /// network) when opening the provider stream, with exponential backoff
    /// between attempts. Defaults to 10 in [`Self::builder`].
    pub max_api_retries: u32,
    /// Working directory threaded into every [`ToolUseContext`].
    pub cwd: PathBuf,
    /// Shared file state cache threaded into every [`ToolUseContext`].
    pub file_cache: Arc<FileStateCache>,
    /// Declared context window for the active model. Used by reactive
    /// compaction to compute thresholds. Defaults to
    /// [`DEFAULT_MODEL_MAX_TOKENS`].
    pub model_max_tokens: u32,
    /// Whether reactive auto-compaction is enabled. When `true`, every
    /// turn evaluates [`AutoCompactState`] before streaming and may
    /// rewrite the message store via [`compact_with_hooks`] +
    /// [`apply_compaction_to_store`]. Defaults to `true`.
    pub auto_compact_enabled: bool,
    /// Cross-run auto-compaction state. Latches like `no_progress` and
    /// the `consecutive_failures` circuit breaker survive across
    /// successive [`Self::run`] calls when the host wires the same
    /// `Arc` into every loop instance via
    /// [`QueryLoopBuilder::compact_state`]. Defaults to a fresh, owned
    /// state when not provided (state then resets per-run, matching
    /// pre-Q-δ behavior).
    pub compact_state: Arc<Mutex<AutoCompactState>>,
    /// Sampling temperature passed to the provider. `None` = provider default.
    pub temperature: Option<f32>,
    /// Whether to request provider prompt caching (Anthropic `cache_control`).
    pub use_prompt_cache: bool,
    /// Whether the microcompact ladder step runs before LLM compaction:
    /// on threshold hit, old tool-result payloads are cleared first, and
    /// if that alone drops the total back under the threshold the LLM
    /// compaction is skipped for this turn. Defaults to `false`.
    pub microcompact_enabled: bool,
    /// Host-supplied extra instructions appended to the summarization
    /// prompt of every auto-compaction (see
    /// [`CompactWithHooksRequest::with_custom_instructions`]).
    pub compact_instructions: Option<String>,
    /// Optional durable sink: after a successful auto-compaction the
    /// analysis bullets are promoted here via
    /// [`crate::compact::promote_to_store`]. Errors surface as a Notice
    /// and never fail the compaction.
    pub session_memory: Option<Arc<dyn SessionMemoryStore>>,
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
        self.run_blocks(
            vec![ContentBlock::Text {
                text: user_msg.into(),
            }],
            abort,
        )
        .await
    }

    /// Run the loop with pre-built user content blocks, such as text plus
    /// inline image attachments.
    pub async fn run_blocks(
        self,
        content: Vec<ContentBlock>,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        // Push user message before spawning so callers see it in the
        // store synchronously.
        let user_message = Message::User {
            header: Header::new(),
            content,
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
    task_depth: usize,
    steer: Option<Arc<std::sync::Mutex<mpsc::UnboundedReceiver<Vec<ContentBlock>>>>>,
    max_iterations: usize,
    max_api_retries: u32,
    cwd: PathBuf,
    file_cache: Option<Arc<FileStateCache>>,
    model_max_tokens: u32,
    auto_compact_enabled: bool,
    compact_state: Option<Arc<Mutex<AutoCompactState>>>,
    temperature: Option<f32>,
    use_prompt_cache: bool,
    microcompact_enabled: bool,
    compact_instructions: Option<String>,
    session_memory: Option<Arc<dyn SessionMemoryStore>>,
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
            task_depth: 0,
            steer: None,
            max_iterations: usize::MAX,
            max_api_retries: 10,
            cwd: PathBuf::from("."),
            file_cache: None,
            model_max_tokens: DEFAULT_MODEL_MAX_TOKENS,
            auto_compact_enabled: true,
            compact_state: None,
            temperature: None,
            use_prompt_cache: false,
            microcompact_enabled: false,
            compact_instructions: None,
            session_memory: None,
        }
    }

    /// Sampling temperature. `None` (default) uses the provider's default;
    /// hosts set a low value (e.g. 0) for deterministic coding output.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Request provider prompt caching. For Anthropic-family providers this
    /// adds `cache_control` breakpoints (system + last user message) so the
    /// stable prefix is cached across turns; other providers (OpenAI-compat)
    /// cache automatically and ignore the flag. Default off.
    pub fn use_prompt_cache(mut self, on: bool) -> Self {
        self.use_prompt_cache = on;
        self
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
    /// Sub-agent nesting depth for this loop (0 = root; Task children
    /// pass parent depth + 1). Feeds `ToolUseContext::task_depth`.
    pub fn task_depth(mut self, depth: usize) -> Self {
        self.task_depth = depth;
        self
    }

    /// Attach a mid-turn steering inbox (see [`QueryLoop::steer`]).
    pub fn steer(
        mut self,
        rx: Arc<std::sync::Mutex<mpsc::UnboundedReceiver<Vec<ContentBlock>>>>,
    ) -> Self {
        self.steer = Some(rx);
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
    /// How many times to retry a transient API failure with exponential
    /// backoff. Defaults to 10.
    pub fn max_api_retries(mut self, n: u32) -> Self {
        self.max_api_retries = n;
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
    /// Override the declared context window. Defaults to
    /// [`DEFAULT_MODEL_MAX_TOKENS`].
    pub fn model_max_tokens(mut self, n: u32) -> Self {
        self.model_max_tokens = n;
        self
    }
    /// Enable or disable reactive auto-compaction. Defaults to `true`.
    pub fn auto_compact(mut self, on: bool) -> Self {
        self.auto_compact_enabled = on;
        self
    }
    /// Inject a shared [`AutoCompactState`] so latches and circuit
    /// breaker counters survive across successive [`QueryLoop::run`]
    /// calls. Without this, each run starts with a fresh state.
    pub fn compact_state(mut self, s: Arc<Mutex<AutoCompactState>>) -> Self {
        self.compact_state = Some(s);
        self
    }
    /// Enable the microcompact ladder step (clear old tool results before
    /// resorting to LLM compaction). Defaults to `false`.
    pub fn microcompact(mut self, on: bool) -> Self {
        self.microcompact_enabled = on;
        self
    }
    /// Extra instructions appended to the auto-compaction summarization
    /// prompt.
    pub fn compact_instructions(mut self, ci: impl Into<String>) -> Self {
        self.compact_instructions = Some(ci.into());
        self
    }
    /// Durable sink for compaction analysis bullets.
    pub fn session_memory(mut self, sm: Arc<dyn SessionMemoryStore>) -> Self {
        self.session_memory = Some(sm);
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
            task_depth: self.task_depth,
            steer: self.steer,
            max_iterations: self.max_iterations,
            max_api_retries: self.max_api_retries,
            cwd: self.cwd,
            file_cache,
            model_max_tokens: self.model_max_tokens,
            auto_compact_enabled: self.auto_compact_enabled,
            compact_state: self
                .compact_state
                .unwrap_or_else(|| Arc::new(Mutex::new(AutoCompactState::new()))),
            temperature: self.temperature,
            use_prompt_cache: self.use_prompt_cache,
            microcompact_enabled: self.microcompact_enabled,
            compact_instructions: self.compact_instructions,
            session_memory: self.session_memory,
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
    let compact_state = qloop.compact_state.clone();
    // The tool set is immutable for the loop's lifetime — serialize the
    // definitions once, not on every model round-trip (each `definitions()`
    // call rebuilds and re-sorts every tool's JSON schema).
    let tool_defs = qloop.tools.definitions();

    // Fixed per-request overhead the message-store token estimate omits: the
    // system prompt and the serialized tool schemas. Constant across iterations.
    let overhead_tokens = estimate_prompt_overhead(qloop.system.as_deref(), &tool_defs);
    // Delta calibration for the output clamp: `(actual_prompt, msg_estimate)`
    // from the previous request. The store only grows within a turn, so
    // `actual_prev + (msg_estimate_now - msg_estimate_prev)` predicts the
    // current prompt accurately — it folds in the true system/tool overhead the
    // provider counted rather than trusting our estimate of it.
    let mut prompt_calibration: Option<(u32, u32)> = None;

    loop {
        if abort.is_aborted() {
            let _ = tx.unbounded_send(Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            )));
            return;
        }

        if iter >= qloop.max_iterations {
            // Surface the partial run as a distinguishable Result first —
            // hosts can tell the runaway backstop from a generic failure
            // and show what the run accomplished before the cap.
            let mut data = final_result.clone();
            data.stop_reason = Some("max_iterations".into());
            let _ = tx.unbounded_send(Ok(Event::Result { data }));
            let _ = tx.unbounded_send(Err(AgentError::other(format!(
                "QueryLoop hit max_iterations ({})",
                qloop.max_iterations
            ))));
            return;
        }
        iter += 1;
        if let Ok(mut s) = compact_state.lock() {
            s.next_turn();
        }

        // ----- Mid-turn steering -----
        // Drain any user messages injected while this multi-step turn was in
        // flight and append them (as User turns) BEFORE the request snapshot,
        // so the model sees the new instruction on the very next round-trip.
        // The store ends with the prior user prompt / tool_results here, so a
        // fresh User message is coherent (providers coalesce adjacent
        // same-role turns). Announced as a Notice so the UI can show it.
        if let Some(steer) = &qloop.steer {
            if let Ok(mut rx) = steer.lock() {
                while let Ok(blocks) = rx.try_recv() {
                    let msg = Message::User {
                        header: child_header(&qloop.store),
                        content: blocks,
                    };
                    if push(&qloop.store, msg).is_ok() {
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "agent.steer".into(),
                            message: "injected a mid-turn user message".into(),
                        }));
                    }
                }
            }
        }

        // ----- Reactive auto-compaction (Q-δ) -----
        // Before each streaming turn, evaluate whether the cumulative
        // token estimate of the current MessageStore has crossed the
        // auto-compact threshold. On hit, run a full hook-instrumented
        // compaction in-place and continue with the rewritten store.
        if qloop.auto_compact_enabled {
            if let Err(e) = maybe_auto_compact(&qloop, &compact_state, &abort, &tx).await {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        }

        // ----- Streaming phase -----
        // The store is kept well-formed at the source (every dispatched
        // tool_use gets a tool_result before the turn can exit), so the raw
        // snapshot is safe to send. Cross-turn same-role adjacency from an
        // interrupt (user tool_result followed by the next user prompt) is
        // folded by each provider's renderer.
        let messages = match snapshot(&qloop.store) {
            Ok(m) => m,
            Err(e) => {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        };
        // Clamp `max_tokens` so `prompt + max_tokens` cannot overflow the
        // context window. `max_output_tokens` is a static per-model ceiling
        // (e.g. 384k on a ~1M window); without this, a turn whose tool results
        // grow the store past `window - max_output` mid-flight hard-400s on the
        // very next request. Estimate the current prompt (calibrated against the
        // last request's provider-reported count) and cap the completion to the
        // room left. Recomputed every iteration because the store grows within a
        // turn.
        let msg_estimate: u32 = messages
            .iter()
            .map(estimate_tokens)
            .fold(0, u32::saturating_add);
        let prompt_estimate = match prompt_calibration {
            Some((actual_prev, est_prev)) => {
                actual_prev.saturating_add(msg_estimate.saturating_sub(est_prev))
            }
            None => msg_estimate.saturating_add(overhead_tokens),
        };
        let effective_output = clamp_output_to_window(
            qloop.max_output_tokens,
            prompt_estimate,
            qloop.model_max_tokens,
        );
        let mut req = StreamRequest::new(qloop.model.clone(), messages)
            .with_max_output_tokens(effective_output);
        if let Some(t) = qloop.temperature {
            req = req.with_temperature(t);
        }
        req = req.with_prompt_cache(qloop.use_prompt_cache);
        if let Some(s) = &qloop.system {
            req = req.with_system(s.clone());
        }
        if !tool_defs.is_empty() {
            req = req.with_tools(tool_defs.clone());
        }

        // Open the provider stream, retrying transient API failures (rate
        // limits, 5xx, transport drops) with exponential backoff. Each attempt
        // waits longer than the last, and every retry is announced as a Notice
        // so the UI can show it. Permanent errors (auth / bad request) and an
        // abort break out immediately.
        let upstream = {
            let mut attempt: u32 = 0;
            loop {
                match qloop.provider.stream(req.clone(), abort.clone()).await {
                    Ok(s) => break s,
                    Err(e) => {
                        if abort.is_aborted() {
                            return;
                        }
                        if attempt >= qloop.max_api_retries || !is_retryable_api_error(&e) {
                            let _ = tx.unbounded_send(Err(e));
                            return;
                        }
                        attempt += 1;
                        let delay = retry_delay(attempt, &e);
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "api_retry".into(),
                            message: format!(
                                "API error (attempt {attempt}/{}), retrying in {}s — {e}",
                                qloop.max_api_retries,
                                delay.as_secs(),
                            ),
                        }));
                        // Back off, but wake immediately if the user aborts.
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {}
                            _ = abort.token().cancelled() => return,
                        }
                    }
                }
            }
        };

        let TurnSummary {
            assistant_blocks,
            pending_tool_uses,
            stop_reason,
            model,
            prompt_tokens,
        } = match consume_turn(upstream, &tx, &abort).await {
            Ok(s) => s,
            Err(()) => return, // already-emitted error or aborted
        };
        // Calibrate the next iteration's clamp on the provider's actual prompt
        // count paired with our estimate of the messages we just sent.
        if let Some(actual) = prompt_tokens {
            prompt_calibration = Some((actual, msg_estimate));
        }
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
        // Set on a consumer-gone / executor-error / abort path, so we repair
        // the store (push results) and then stop instead of re-looping. Lives
        // here (before the permission loop) so a denial whose result fails to
        // forward to a dropped consumer also trips it.
        let mut terminate = false;
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
                    if !forward_tool_result(&tx, synthetic.clone()) {
                        terminate = true;
                    }
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
                    if !forward_tool_result(&tx, synthetic.clone()) {
                        terminate = true;
                    }
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
                if !forward_tool_result(&tx, synthetic.clone()) {
                    terminate = true;
                }
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

        // Ids dispatched to the executor, captured before it consumes them.
        // The abort path emits a single Err(Aborted) and drops the buffered
        // stream, so any survivor that never yields a result is reconciled
        // below into a synthetic `[interrupted]` result.
        let dispatched_ids: Vec<String> = surviving.iter().map(|tu| tu.id.clone()).collect();

        // ----- ToolCollecting phase: dispatch survivors via executor -----
        // Skipped when the consumer already vanished mid-permission-phase
        // (`terminate`); the reconcile below then synthesizes results for every
        // survivor so the store is still repaired before we return.
        if !surviving.is_empty() && !terminate {
            let ctx = ToolUseContext {
                cwd: qloop.cwd.clone(),
                abort: abort.clone(),
                file_cache: qloop.file_cache.clone(),
                permissions: qloop.permissions.clone(),
                hooks: qloop.hooks.clone(),
                task_depth: qloop.task_depth,
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
                            // Consumer gone: stop streaming, but fall through to
                            // repair the store (push tool_results) so no
                            // dangling tool_use is left behind.
                            terminate = true;
                            break;
                        }
                    }
                    Ok(other) => {
                        if tx.unbounded_send(Ok(other)).is_err() {
                            terminate = true;
                            break;
                        }
                    }
                    Err(e) => {
                        // Executor error (incl. mid-dispatch abort, which emits
                        // one Aborted and drops remaining survivors). Forward
                        // best-effort, then break to repair the store.
                        let _ = tx.unbounded_send(Err(e));
                        terminate = true;
                        break;
                    }
                }
            }
        }

        // ----- Reconcile: every dispatched tool_use MUST get a result -----
        // The assistant message already pushed carries N tool_use blocks; the
        // permission loop + executor filled `tool_results`. Anything still
        // missing (mid-dispatch abort dropped survivors, executor error, or a
        // consumer-gone break above) gets a synthetic `[interrupted]` result,
        // so the store never holds a tool_use without a matching tool_result.
        let answered: std::collections::HashSet<&str> =
            tool_results.iter().map(|(id, _, _)| id.as_str()).collect();
        let missing: Vec<String> = dispatched_ids
            .iter()
            .filter(|id| !answered.contains(id.as_str()))
            .cloned()
            .collect();
        for id in missing {
            tool_results.push((id, false, serde_json::json!({ "error": "[interrupted]" })));
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

        // A consumer-gone / executor-error / abort break repaired the store
        // above; stop here instead of streaming another turn into the void.
        if terminate {
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

/// Reserve held back from the context window when clamping a request's
/// completion size, covering residual estimator bias. Scales with the window —
/// a 1M window's prompt estimate can be off by more in absolute terms than a
/// 200k one — with a floor so small windows stay safe.
const OUTPUT_CLAMP_RESERVE_DIVISOR: u32 = 32;
const OUTPUT_CLAMP_RESERVE_FLOOR: u32 = 8_192;

/// Never request fewer than this many output tokens: providers reject
/// `max_tokens=0`, and a handful is useless. If the prompt is so large that even
/// this floor overflows the window the request may still 400 — but reactive
/// auto-compaction runs first each iteration, so the store should already have
/// been trimmed before we get here.
const MIN_CLAMPED_OUTPUT: u32 = 512;

/// Estimate the fixed per-request overhead the message-store token estimate
/// omits: the system prompt plus the serialized tool schemas. Byte-rate (÷4)
/// matches [`estimate_text_tokens`]'s ASCII path — good enough for a reserve.
fn estimate_prompt_overhead(system: Option<&str>, tools: &[ToolDefinition]) -> u32 {
    let sys = system.map(estimate_text_tokens).unwrap_or(0);
    let tool_bytes = serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0);
    sys.saturating_add((tool_bytes as u32) / 4)
}

/// Shrink a request's completion budget so `prompt + max_tokens` fits inside the
/// context `window`. `configured` is the static per-model output ceiling;
/// `prompt` is the (calibrated) current occupancy. Returns the room left after a
/// scaled reserve, never above `configured` nor below [`MIN_CLAMPED_OUTPUT`]. A
/// degenerate window (0/1) is left alone. Over-clamping only shortens a
/// completion (the loop continues); under-clamping hard-400s.
fn clamp_output_to_window(configured: u32, prompt: u32, window: u32) -> u32 {
    if window <= 1 {
        return configured;
    }
    let reserve = (window / OUTPUT_CLAMP_RESERVE_DIVISOR).max(OUTPUT_CLAMP_RESERVE_FLOOR);
    let headroom = window.saturating_sub(prompt).saturating_sub(reserve);
    configured.min(headroom).max(MIN_CLAMPED_OUTPUT)
}

#[derive(Debug, Default)]
struct TurnSummary {
    assistant_blocks: Vec<ContentBlock>,
    pending_tool_uses: Vec<RequestedToolUse>,
    stop_reason: Option<String>,
    model: Option<String>,
    /// Full prompt occupancy the provider reported for this request (non-cached
    /// input + both cache tiers), if a Usage event arrived. Feeds the next
    /// iteration's output-clamp calibration.
    prompt_tokens: Option<u32>,
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
                    if tx.unbounded_send(Ok(Event::TextDelta { delta })).is_err() {
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
                Event::Usage {
                    input_tokens,
                    cache_read,
                    cache_create,
                    ..
                } => {
                    // Record the provider's own prompt count (fields are Copy, so
                    // `event` stays intact to forward). The next iteration uses it
                    // to size `max_tokens` so `prompt + max_tokens` stays under the
                    // context window.
                    summary.prompt_tokens = Some(
                        input_tokens
                            .saturating_add(cache_read)
                            .saturating_add(cache_create),
                    );
                    if tx.unbounded_send(Ok(event)).is_err() {
                        return Err(());
                    }
                }
                Event::Error { .. }
                | Event::Notice { .. }
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

    // Order blocks the way the model produced them: reasoning first, then the
    // answer text, then tool calls. Storing thinking BEFORE text keeps a resumed
    // session's transcript consistent with the live view (reasoning above the
    // answer) instead of surfacing the answer before its reasoning.
    if let Some(thinking) = accumulated_thinking {
        summary.assistant_blocks.push(ContentBlock::Thinking {
            thinking,
            signature: None,
        });
    }
    if !accumulated_text.is_empty() {
        summary.assistant_blocks.push(ContentBlock::Text {
            text: accumulated_text,
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

/// Whether an API failure is transient and worth retrying. Providers that report
/// `HTTP <code>: ...` (anthropic-compat) are classified precisely — rate limits,
/// overload, and 5xx retry; auth / bad-request do not. SDK-wrapped errors without
/// a clean status (openai/ollama) are treated as retryable transport failures
/// unless they name an obvious permanent condition.
fn is_retryable_api_error(e: &AgentError) -> bool {
    let AgentError::Provider { message, .. } = e else {
        return false;
    };
    if let Some(code) = parse_http_status(message) {
        // 429 (rate limit) + a few transient client statuses, and ANY 5xx server
        // error — 500/502/503/504, Anthropic 529 overload, and gateway/proxy 52x
        // (common from third-party endpoints) — except 501 Not Implemented, which
        // is permanent.
        return code == 408
            || code == 409
            || code == 425
            || code == 429
            || (code >= 500 && code != 501);
    }
    let m = message.to_ascii_lowercase();
    !(m.contains("invalid")
        || m.contains("unauthorized")
        || m.contains("401")
        || m.contains("403")
        || m.contains("400")
        || m.contains("404"))
}

/// Parse the leading numeric status from a `"HTTP <code> ...: ..."` message.
fn parse_http_status(message: &str) -> Option<u16> {
    let rest = message.strip_prefix("HTTP ")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Exponential backoff for a 1-based retry `attempt`: 1s, 2s, 4s, 8s, … capped at
/// 60s so a long outage doesn't stall each retry indefinitely.
fn retry_backoff(attempt: u32) -> std::time::Duration {
    let secs = 1u64
        .checked_shl(attempt.saturating_sub(1))
        .unwrap_or(u64::MAX)
        .min(60);
    std::time::Duration::from_secs(secs)
}

/// Actual delay before retry `attempt`. Server guidance wins: providers embed
/// the `Retry-After` header as a `retry-after=<secs>` token in the error text
/// (the response object doesn't survive the error path), honored up to 120s.
/// Otherwise [`retry_backoff`] with EQUAL JITTER — uniform in
/// `[base/2, base]` — so a fleet of agents retrying the same outage doesn't
/// stampede the recovering endpoint in lockstep.
fn retry_delay(attempt: u32, error: &AgentError) -> std::time::Duration {
    if let AgentError::Provider { message, .. } = error {
        if let Some(secs) = parse_retry_after(message) {
            return std::time::Duration::from_secs(secs.min(120));
        }
    }
    let base_ms = retry_backoff(attempt).as_millis() as u64;
    let half = base_ms / 2;
    // Cheap jitter without a rand dependency: subsecond clock noise.
    let noise = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()))
        .unwrap_or(0);
    std::time::Duration::from_millis(half + noise % (half + 1))
}

/// Parse the numeric `retry-after=<secs>` token a provider embedded in an
/// error message. HTTP-date forms are ignored (backoff covers those).
fn parse_retry_after(message: &str) -> Option<u64> {
    let idx = message.find("retry-after=")?;
    let digits: String = message[idx + "retry-after=".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn push(store: &Arc<Mutex<MessageStore>>, msg: Message) -> Result<(), AgentError> {
    let mut s = store
        .lock()
        .map_err(|_| AgentError::other("query store lock poisoned"))?;
    s.push(msg)
}

fn child_header(store: &Arc<Mutex<MessageStore>>) -> Header {
    let parent = store.lock().ok().and_then(|s| s.last().map(|m| m.uuid()));
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

/// Forward a synthetic tool result to the consumer. Returns `false` if the
/// consumer has gone away (receiver dropped), so the caller can stop the turn
/// instead of streaming another round into a dead channel.
fn forward_tool_result(
    tx: &mpsc::UnboundedSender<Result<Event, AgentError>>,
    synthetic: SyntheticToolResult,
) -> bool {
    tx.unbounded_send(Ok(Event::ToolResult {
        id: synthetic.0.id,
        ok: synthetic.0.ok,
        output: synthetic.0.output,
    }))
    .is_ok()
}

/// Reactive auto-compaction helper invoked once per turn.
///
/// Reads the live `MessageStore`, sums token estimates, evaluates
/// [`AutoCompactState`], and on `should_compact == true`:
///
/// 1. Calls [`compact_with_hooks`] (which fires PreCompact + PostCompact
///    hooks around [`super::super::compact::summarize::compact_conversation`]).
/// 2. Rewrites the store in-place via [`apply_compaction_to_store`].
/// 3. Forwards an `Event::Notice`-equivalent observability ping by
///    surfacing a synthetic `Event::Unknown`-like marker — for now we
///    rely on hooks for telemetry and only short-circuit on
///    [`CompactError::Aborted`].
///
/// **Failure handling**:
/// - `Aborted` due to outer abort → propagates as [`AgentError::Aborted`].
/// - `Aborted` from a `PreCompact` hook returning `Block` (outer abort
///   NOT fired) → recoverable: surfaces an `Event::Notice`, records a
///   failure for the breaker, and the turn continues.
/// - Any other error → records a failure (advances circuit breaker),
///   surfaces an `Event::Notice`, and the turn continues with the
///   un-compacted store. Three in a row opens the breaker.
///
/// **Direction**: uses [`PartialCompactDirection::EarliestHalf`] so the
/// most recent half of the transcript (including the user message that
/// just arrived via [`QueryLoop::run`]) is preserved verbatim. Without
/// this the current turn's user input would be tombstoned and the
/// provider would never see fresh input.
///
/// **No-progress detection**: after a successful compaction, the
/// post-compact token total is re-evaluated. If still ≥ threshold,
/// [`AutoCompactState::record_no_progress`] is latched so subsequent
/// turns skip auto-compaction entirely (preventing oscillation when
/// the summary itself is dense).
/// Minimum messages to compact in a partial direction. Below this we
/// skip — replacing 1 message with a boundary + summary pair is a net
/// loss in both message count and tokens.
const MIN_PARTIAL_COMPACT_TARGET: usize = 2;

async fn maybe_auto_compact(
    qloop: &QueryLoop,
    state: &Arc<Mutex<AutoCompactState>>,
    abort: &AbortController,
    tx: &mpsc::UnboundedSender<Result<Event, AgentError>>,
) -> Result<(), AgentError> {
    // Snapshot once — we want a stable view of the store for token
    // accounting AND for the compaction request itself.
    let snapshot = snapshot(&qloop.store)?;
    if snapshot.len() < 2 {
        return Ok(());
    }

    // Auto-compaction uses EarliestHalf, which compacts `len/2`
    // messages. Skip when that would be fewer than the minimum.
    let target_count = snapshot.len() / 2;
    if target_count < MIN_PARTIAL_COMPACT_TARGET {
        return Ok(());
    }

    let current_tokens: u32 = snapshot
        .iter()
        .map(estimate_tokens)
        .fold(0u32, u32::saturating_add);

    let decision = {
        let s = state
            .lock()
            .map_err(|_| AgentError::other("compact state lock poisoned"))?;
        s.evaluate(current_tokens, qloop.model_max_tokens)
    };

    // ----- Ladder step ①: microcompact (no LLM call) -----
    // Clearing old tool-result payloads is free, so it runs from the
    // WARNING threshold up — well before LLM summarization is needed —
    // instead of waiting for the auto-compact threshold. When it alone
    // keeps the total under the auto threshold, LLM compaction is skipped
    // this turn. Deliberately NOT recorded as success/failure: the
    // circuit breaker tracks LLM compaction attempts only.
    let mut snapshot = snapshot;
    if qloop.microcompact_enabled
        && current_tokens >= crate::compact::warning_threshold(qloop.model_max_tokens)
    {
        let threshold = crate::compact::auto_compact_threshold(qloop.model_max_tokens);
        let (mc, new_total, new_snapshot) = {
            let mut store = qloop
                .store
                .lock()
                .map_err(|_| AgentError::other("query store lock poisoned"))?;
            let mc = apply_microcompact_to_store(&mut store, &MicrocompactConfig::default())?;
            let new_total = store
                .iter()
                .map(estimate_tokens)
                .fold(0u32, u32::saturating_add);
            let new_snapshot: Vec<Message> = store.iter().cloned().collect();
            (mc, new_total, new_snapshot)
        };
        if mc.cleared_count > 0 {
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.compact.micro".into(),
                message: format!(
                    "microcompact cleared {} tool result(s), freed ~{} tokens",
                    mc.cleared_count, mc.tokens_freed
                ),
            }));
            snapshot = new_snapshot;
            if new_total < threshold {
                return Ok(());
            }
        }
    }

    if !decision.should_compact {
        return Ok(());
    }
    // Defense in depth: evaluate() should already have masked these,
    // but guard against future refactors changing the surface.
    if matches!(
        decision.reason,
        AutoCompactReason::CircuitBreakerOpen { .. } | AutoCompactReason::NoProgress
    ) {
        return Ok(());
    }

    // Run the compaction. `EarliestHalf` keeps the recent half — vital
    // because [`QueryLoop::run`] just pushed the user's prompt and we
    // need the model to see it on the next streaming turn.
    //
    // Use a child abort so an internal abort during summarization
    // doesn't poison the outer loop. Parent → child propagates, child
    // → parent does not.
    let mut request = CompactWithHooksRequest::new(&snapshot, qloop.model.clone())
        .with_trigger(CompactTrigger::Auto)
        .with_direction(PartialCompactDirection::EarliestHalf)
        .with_abort(abort.child());
    if let Some(ci) = qloop.compact_instructions.as_deref() {
        request = request.with_custom_instructions(ci);
    }
    let outcome = compact_with_hooks(&qloop.hooks, &*qloop.provider, request).await;

    match outcome {
        Ok(result) => {
            let auto_threshold = crate::compact::auto_compact_threshold(qloop.model_max_tokens);
            // Splice result into the live store under a short critical
            // section.
            let new_total = {
                let mut store = qloop
                    .store
                    .lock()
                    .map_err(|_| AgentError::other("query store lock poisoned"))?;
                apply_compaction_to_store(&mut store, &result)?;
                store
                    .iter()
                    .map(estimate_tokens)
                    .fold(0u32, u32::saturating_add)
            };
            {
                let mut s = state
                    .lock()
                    .map_err(|_| AgentError::other("compact state lock poisoned"))?;
                s.record_success();
                // No-progress detection: if the post-compact total is
                // still ≥ threshold, latch the flag so we don't
                // oscillate. The user can clear it via
                // [`AutoCompactState::reset_no_progress`].
                if new_total >= auto_threshold {
                    s.record_no_progress();
                }
            }
            if let Some(sm) = &qloop.session_memory {
                match promote_to_store(sm.as_ref(), &result).await {
                    Ok(n) if n > 0 => {
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "agent.compact.memory".into(),
                            message: format!("promoted {n} analysis bullet(s) to session memory"),
                        }));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "agent.compact.memory_failed".into(),
                            message: e.to_string(),
                        }));
                    }
                }
            }
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.compact.ok".into(),
                message: format!(
                    "auto-compact {} → {} tokens ({} messages replaced)",
                    result.pre_compact_tokens,
                    result.post_compact_tokens,
                    result.replaced_uuids.len()
                ),
            }));
            Ok(())
        }
        Err(CompactError::Aborted) => {
            // Distinguish outer abort vs. PreCompact hook block. The
            // hook path returns Aborted without setting the abort
            // token, so we can recover.
            if abort.is_aborted() {
                Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "compact aborted".into()),
                ))
            } else {
                if let Ok(mut s) = state.lock() {
                    s.record_failure();
                }
                let _ = tx.unbounded_send(Ok(Event::Notice {
                    code: "agent.compact.blocked".into(),
                    message: "PreCompact hook blocked auto-compaction".into(),
                }));
                Ok(())
            }
        }
        Err(other) => {
            if let Ok(mut s) = state.lock() {
                s.record_failure();
            }
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.compact.failed".into(),
                message: other.to_string(),
            }));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures::StreamExt;

    use super::*;
    use crate::permission::{PermissionMode, RuleSource};
    use crate::provider::ProviderCapabilities;
    use crate::testing::MockProvider;
    use crate::tool::Tool;

    #[test]
    fn clamp_output_keeps_prompt_plus_completion_within_window() {
        // The reported failure: ~1M window, 384k output ceiling, a mid-turn
        // history of 900_839 tokens. 900_839 + 384_000 = 1_284_839 > 1_048_565.
        let window = 1_048_565;
        let prompt = 900_839;
        let out = clamp_output_to_window(384_000, prompt, window);
        assert!(out < 384_000, "must clamp below the static ceiling");
        assert!(
            prompt + out < window,
            "prompt {prompt} + completion {out} must fit window {window}"
        );
        // Reserve is window/32 here (32_767 > 8_192 floor).
        assert_eq!(out, window - prompt - window / OUTPUT_CLAMP_RESERVE_DIVISOR);
    }

    #[test]
    fn clamp_output_is_a_noop_with_headroom_and_never_zero() {
        // Small prompt on a large window → the full configured ceiling.
        assert_eq!(clamp_output_to_window(384_000, 40_000, 1_048_565), 384_000);
        // Prompt fills the window → floored, never max_tokens=0.
        assert_eq!(
            clamp_output_to_window(16_384, 1_048_000, 1_048_565),
            MIN_CLAMPED_OUTPUT
        );
        // Degenerate window is left untouched (no real model has one).
        assert_eq!(clamp_output_to_window(16_384, 0, 1), 16_384);
    }

    #[test]
    fn delta_calibration_predicts_grown_prompt() {
        // Iteration 1: 100k message estimate, provider counted 130k (30k of
        // system + tools). Iteration 2's store grew to 880k of messages.
        let (actual_prev, est_prev) = (130_000u32, 100_000u32);
        let msg_now = 880_000u32;
        let predicted = actual_prev.saturating_add(msg_now.saturating_sub(est_prev));
        // Additive overhead carries forward (not multiplied): ~910k, not 1.14M.
        assert_eq!(predicted, 910_000);
        let out = clamp_output_to_window(384_000, predicted, 1_048_565);
        assert!(predicted + out < 1_048_565, "calibrated request must fit");
    }

    #[test]
    fn overhead_estimate_counts_system_and_tools() {
        let tools = vec![ToolDefinition::new(
            "grep",
            "search files",
            serde_json::json!({"type": "object"}),
        )];
        let with = estimate_prompt_overhead(Some("you are a helpful agent"), &tools);
        let without = estimate_prompt_overhead(None, &[]);
        assert!(with > without, "system + tools must add to the estimate");
    }

    #[test]
    fn retryable_classifies_http_statuses() {
        let prov = |m: &str| AgentError::provider("anthropic", m);
        // Rate limit + ANY 5xx server/gateway/overload error → retry.
        assert!(is_retryable_api_error(&prov(
            "HTTP 429 Too Many Requests: slow down"
        )));
        assert!(is_retryable_api_error(&prov(
            "HTTP 500 Internal Server Error: x"
        )));
        assert!(is_retryable_api_error(&prov(
            "HTTP 503 Service Unavailable: x"
        )));
        assert!(is_retryable_api_error(&prov("HTTP 504 Gateway Timeout: x")));
        assert!(is_retryable_api_error(&prov(
            "HTTP 522 Connection Timed Out: cf"
        )));
        assert!(is_retryable_api_error(&prov("HTTP 529 Overloaded: x")));
        // Auth / bad request / not-implemented → do NOT retry.
        assert!(!is_retryable_api_error(&prov(
            "HTTP 401 Unauthorized: bad key"
        )));
        assert!(!is_retryable_api_error(&prov(
            "HTTP 400 Bad Request: schema"
        )));
        assert!(!is_retryable_api_error(&prov(
            "HTTP 501 Not Implemented: x"
        )));
        // No status = transport failure → retry; aborts/others → no.
        assert!(is_retryable_api_error(&prov(
            "error sending request: connection reset"
        )));
        assert!(!is_retryable_api_error(&AgentError::Aborted("user".into())));
        assert!(!is_retryable_api_error(&prov("invalid x-api-key header")));
    }

    #[test]
    fn backoff_grows_then_caps() {
        let s = |a| retry_backoff(a).as_secs();
        assert_eq!((s(1), s(2), s(3), s(4)), (1, 2, 4, 8));
        assert!(s(2) > s(1) && s(3) > s(2)); // each longer than the last…
        assert_eq!(s(7), 60); // …until the 60s cap
        assert_eq!(s(10), 60);
    }

    #[test]
    fn retry_delay_honors_server_retry_after() {
        // A provider that embedded Retry-After overrides the backoff curve.
        let e = AgentError::provider("anthropic", "HTTP 429 retry-after=7: slow down");
        assert_eq!(retry_delay(1, &e).as_secs(), 7);
        // Clamped to 120s so a hostile header can't park the loop forever.
        let e = AgentError::provider("anthropic", "HTTP 503 retry-after=9999: x");
        assert_eq!(retry_delay(1, &e).as_secs(), 120);
    }

    #[test]
    fn retry_delay_jitters_within_equal_jitter_band() {
        // No server hint → equal jitter in [base/2, base] for the attempt.
        let e = AgentError::provider("x", "HTTP 500: boom");
        let base = retry_backoff(4).as_millis() as u64; // 8000ms
        for _ in 0..50 {
            let d = retry_delay(4, &e).as_millis() as u64;
            assert!(d >= base / 2 && d <= base, "delay {d} out of band");
        }
    }

    #[test]
    fn parse_retry_after_extracts_seconds() {
        assert_eq!(parse_retry_after("HTTP 429 retry-after=12: x"), Some(12));
        assert_eq!(parse_retry_after("HTTP 500: no header"), None);
        // HTTP-date form is ignored (non-numeric) — backoff covers it.
        assert_eq!(parse_retry_after("retry-after=Wed, 21 Oct"), None);
    }

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

    /// Tool that aborts the turn from inside its own call — simulates the user
    /// hitting Esc (or the consumer dropping the stream) while a tool runs.
    #[derive(Debug)]
    struct AbortingTool {
        abort: AbortController,
    }

    #[async_trait]
    impl Tool for AbortingTool {
        fn name(&self) -> &str {
            "slow"
        }
        fn description(&self) -> &str {
            "aborts the turn mid-call"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            ctx: &ToolUseContext,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            self.abort.abort_with_reason("user interrupted");
            ctx.abort.cancelled().await;
            Ok(input)
        }
    }

    /// Provider that captures every `StreamRequest` it receives.
    /// Returns an empty stream so the loop terminates after a single
    /// turn when paired with a no-tool registry.
    #[derive(Debug)]
    struct CapturingProvider {
        captured: Arc<std::sync::Mutex<Vec<StreamRequest>>>,
    }

    #[async_trait]
    impl Provider for CapturingProvider {
        fn id(&self) -> &str {
            "capturing"
        }
        fn capabilities(&self) -> crate::provider::ProviderCapabilities {
            crate::provider::ProviderCapabilities {
                supports_tool_use: true,
                ..Default::default()
            }
        }
        async fn stream(
            &self,
            req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            if let Ok(mut g) = self.captured.lock() {
                g.push(req);
            }
            Ok(Box::new(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn loop_forwards_registered_tools_to_provider() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            captured: captured.clone(),
        });
        let registry = echo_registry("calc", Arc::new(AtomicUsize::new(0)));
        let qloop = QueryLoop::builder(provider, "m").tools(registry).build();
        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        while stream.next().await.is_some() {}
        let captured = captured.lock().unwrap();
        assert!(
            !captured.is_empty(),
            "should have captured at least one request"
        );
        assert_eq!(captured[0].tools.len(), 1);
        assert_eq!(captured[0].tools[0].name, "calc");
    }

    #[tokio::test]
    async fn run_blocks_pushes_rich_user_content() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            captured: captured.clone(),
        });
        let qloop = QueryLoop::builder(provider, "m").build();
        let content = vec![
            ContentBlock::Text {
                text: "describe".into(),
            },
            ContentBlock::Image {
                source: crate::message::ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "abc123".into(),
                },
            },
        ];
        let mut stream = qloop
            .run_blocks(content.clone(), AbortController::new())
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        let captured = captured.lock().unwrap();
        let Message::User {
            content: observed, ..
        } = &captured[0].messages[0]
        else {
            panic!("expected user message");
        };
        assert_eq!(observed, &content);
    }

    #[tokio::test]
    async fn single_turn_text_only() {
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "hi ".into(),
            },
            Event::TextDelta {
                delta: "there".into(),
            },
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
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "echo"));

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .build();

        let store = qloop.store.clone();
        let mut stream = qloop
            .run("call echo", AbortController::new())
            .await
            .unwrap();
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
    async fn interrupted_tool_call_leaves_no_dangling_tool_use() {
        // The provider asks for a tool; the tool aborts the turn mid-call.
        // The store must still end with a tool_result for that tool_use, or
        // the next request would 400 ("tool_use without tool_result").
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::ToolUse {
                id: "tu_1".into(),
                name: "slow".into(),
                input: serde_json::json!({}),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("tool_use".into()),
                    ..Default::default()
                },
            },
        ]]));
        let abort = AbortController::new();
        let mut r = ToolRegistry::new();
        r.register(Arc::new(AbortingTool {
            abort: abort.clone(),
        }));
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "slow"));
        let qloop = QueryLoop::builder(provider, "mock")
            .tools(Arc::new(r))
            .permissions(perms)
            .build();

        let store = qloop.store.clone();
        let mut stream = qloop.run("go", abort.clone()).await.unwrap();
        while stream.next().await.is_some() {}

        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        let tool_use_ids: Vec<String> = snap
            .iter()
            .filter_map(|m| match m {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        let result_ids: std::collections::HashSet<String> = snap
            .iter()
            .filter_map(|m| match m {
                Message::User { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            })
            .collect();
        assert!(
            !tool_use_ids.is_empty(),
            "a tool_use was recorded in the store"
        );
        for id in &tool_use_ids {
            assert!(
                result_ids.contains(id),
                "tool_use {id} must have a matching tool_result (no dangling)"
            );
        }
    }

    #[tokio::test]
    async fn partial_abort_fills_only_unanswered_tool_uses() {
        // Two tool_uses in one assistant turn; one tool aborts the turn. Both
        // ids must end with a result (the finished one real or synthetic, the
        // aborted one synthetic) — never a dangling tool_use.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::ToolUse {
                id: "tu_1".into(),
                name: "echo".into(),
                input: serde_json::json!({ "v": 1 }),
            },
            Event::ToolUse {
                id: "tu_2".into(),
                name: "slow".into(),
                input: serde_json::json!({}),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("tool_use".into()),
                    ..Default::default()
                },
            },
        ]]));
        let abort = AbortController::new();
        let mut r = ToolRegistry::new();
        r.register(Arc::new(EchoTool {
            name: "echo".into(),
            calls: Arc::new(AtomicUsize::new(0)),
        }));
        r.register(Arc::new(AbortingTool {
            abort: abort.clone(),
        }));
        let perms = Arc::new(
            PermissionManager::new()
                .allow(RuleSource::User, "echo")
                .allow(RuleSource::User, "slow"),
        );
        let qloop = QueryLoop::builder(provider, "mock")
            .tools(Arc::new(r))
            .permissions(perms)
            .build();

        let store = qloop.store.clone();
        let mut stream = qloop.run("go", abort.clone()).await.unwrap();
        while stream.next().await.is_some() {}

        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        let result_ids: std::collections::HashSet<String> = snap
            .iter()
            .filter_map(|m| match m {
                Message::User { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            })
            .collect();
        assert!(
            result_ids.contains("tu_1"),
            "tu_1 has a result: {result_ids:?}"
        );
        assert!(
            result_ids.contains("tu_2"),
            "tu_2 has a result: {result_ids:?}"
        );
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
        let perms = Arc::new(PermissionManager::new().deny(RuleSource::Policy, "echo"));

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
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolResult { ok: false, .. })));
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
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "echo"));

        let blocking_hook =
            Arc::new(crate::hook::RustHookHandler::new(
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

    fn compact_response() -> &'static str {
        "<analysis>- prior turns happened</analysis><summary>Prior context summarized.</summary>"
    }

    /// Build a builder pre-populated with `extra` messages plus aggressive
    /// compact thresholds so any 2+ messages trigger immediately.
    ///
    /// Takes `Arc<dyn Provider>` (not `Arc<MockProvider>`) so tests that
    /// need to inspect the outgoing `StreamRequest` (e.g. asserting on
    /// `compact_instructions`) can plug in a custom `Provider` impl.
    fn compact_loop_builder(
        provider: Arc<dyn Provider>,
        store: Arc<Mutex<MessageStore>>,
    ) -> QueryLoopBuilder {
        QueryLoop::builder(provider, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store)
            // Saturates every threshold to 0: any non-empty snapshot fires.
            .model_max_tokens(20_001)
    }

    fn preload_store(messages: Vec<Message>) -> Arc<Mutex<MessageStore>> {
        let mut s = MessageStore::new();
        for m in messages {
            s.push(m).unwrap();
        }
        Arc::new(Mutex::new(s))
    }

    fn user_text(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn steer_injects_a_user_message_before_the_next_round_trip() {
        // A message sent into the steer inbox is drained at the top of the
        // loop and appended as a User turn before the request snapshot, so
        // the model sees it. Sending it before run() means iteration 1 picks
        // it up deterministically.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta { delta: "ok".into() },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let store = preload_store(vec![]);
        let (tx_steer, rx_steer) = mpsc::unbounded::<Vec<ContentBlock>>();
        tx_steer
            .unbounded_send(vec![ContentBlock::Text {
                text: "actually, also do X".into(),
            }])
            .unwrap();
        let qloop = QueryLoop::builder(provider, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .steer(Arc::new(std::sync::Mutex::new(rx_steer)))
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut saw_steer_notice = false;
        while let Some(item) = stream.next().await {
            if let Ok(Event::Notice { code, .. }) = &item {
                if code == "agent.steer" {
                    saw_steer_notice = true;
                }
            }
        }
        assert!(saw_steer_notice, "a steer notice should be emitted");
        // The injected message is in the store as a User turn.
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        let injected = snap.iter().any(|m| match m {
            Message::User { content, .. } => content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { text } if text.contains("also do X"))),
            _ => false,
        });
        assert!(injected, "steered message must land in the store");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_fires_on_threshold_hit_and_rewrites_store() {
        // Provider scripts: turn 0 = compaction summary (consumed by
        // compact_with_hooks), turn 1 = the regular assistant streaming
        // reply that follows, ending with end_turn.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::TextDelta {
                    delta: compact_response().into(),
                },
                Event::Result {
                    data: Default::default(),
                },
            ],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        // Preload 4 messages whose token mass sits in the FIRST two, so
        // EarliestHalf (token-midpoint split) compacts exactly those and
        // preserves the small trailing 2 verbatim. After run() pushes
        // the user prompt, snapshot length is 5, replaced = 2.
        let store = preload_store(vec![
            user_text(&"u1 ".repeat(200)),
            assistant_text(&"a1 ".repeat(200)),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        // Notice event (code = "agent.compact.ok") proves auto-compact fired.
        let notice = events.iter().find_map(|e| match e {
            Event::Notice { code, message } if code == "agent.compact.ok" => Some(message.clone()),
            _ => None,
        });
        assert!(
            notice.is_some(),
            "expected agent.compact.ok notice, got events: {events:?}"
        );

        // Both scripted turns were consumed (compact + assistant).
        assert_eq!(provider.remaining_turns(), 0);

        // EarliestHalf snapshot of 5 (4 preload + run-pushed user)
        // tombstones the first 2 in place AND inserts boundary+summary
        // AT the boundary (after the tombstones, before preserved).
        // Final layout (8 messages):
        //   0: Tombstone(u1)
        //   1: Tombstone(a1)
        //   2: System boundary
        //   3: User summary "[Context summary]…"
        //   4: User u2 (preserved)
        //   5: Assistant a2 (preserved)
        //   6: User "trigger" (preserved fresh prompt)
        //   7: Assistant fallback reply
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert_eq!(snap.len(), 8, "got {snap:?}");
        assert!(matches!(snap[0], Message::Tombstone { .. }));
        assert!(matches!(snap[1], Message::Tombstone { .. }));
        assert!(matches!(
            snap[2],
            Message::System { ref text, .. } if text == crate::compact::COMPACT_BOUNDARY_TEXT
        ));
        match &snap[3] {
            Message::User { content, .. } => {
                let text = content
                    .iter()
                    .find_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .expect("summary user msg has text block");
                assert!(text.starts_with("[Context summary]"));
            }
            other => panic!("expected User summary at index 3, got {other:?}"),
        }
        // Critical: the fresh "trigger" prompt sits AFTER the summary
        // (at idx 6), so Anthropic's view (skipping Tombstone+System) is
        // [summary-as-user, u2, a2, trigger, assistant_reply] —
        // chronologically coherent.
        let trigger_idx = snap.iter().position(|m| match m {
            Message::User { content, .. } => content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { text } if text == "trigger")),
            _ => false,
        });
        assert!(trigger_idx.is_some(), "fresh user turn was tombstoned away");
        assert!(
            trigger_idx.unwrap() > 3,
            "fresh user prompt should sit after summary at idx 3, got {:?}",
            trigger_idx
        );
        // Final assistant reply.
        assert!(matches!(snap[7], Message::Assistant { .. }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_disabled_does_not_consume_provider_turn() {
        // Only script the assistant turn. If auto-compact tried to run
        // it would fail because the second turn's events don't parse as
        // <analysis>/<summary>.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "hello".into(),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let store = preload_store(vec![user_text("u1"), assistant_text("a1")]);
        let qloop = compact_loop_builder(provider.clone(), store.clone())
            .auto_compact(false)
            .build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // No compact notice, and the single scripted turn was consumed.
        assert_eq!(provider.remaining_turns(), 0);
        assert!(!events.iter().any(
            |e| matches!(e, Event::Notice { code, .. } if code.starts_with("agent.compact."))
        ));
        // Store: u1, a1, trigger, assistant — 4, no tombstones.
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert_eq!(snap.len(), 4);
        assert!(snap.iter().all(|m| !matches!(m, Message::Tombstone { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_failure_surfaces_notice_and_loop_continues() {
        // Turn 0 = empty stream (compaction returns EmptyResponse).
        // Turn 1 = regular assistant reply ending the loop.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![Event::Result {
                data: Default::default(),
            }],
            vec![
                Event::TextDelta {
                    delta: "fallback".into(),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        // Preload 4 messages so EarliestHalf compacts 2 (≥ MIN_PARTIAL_COMPACT_TARGET).
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // agent.compact.failed Notice event surfaced.
        assert!(events.iter().any(|e| matches!(
            e,
            Event::Notice { code, .. } if code == "agent.compact.failed"
        )));
        // Loop still completed with a final Result.
        assert!(matches!(events.last(), Some(Event::Result { .. })));
        // Both turns consumed (failed compact + assistant).
        assert_eq!(provider.remaining_turns(), 0);
        // Store unchanged by failed compaction (no tombstones).
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert!(snap.iter().all(|m| !matches!(m, Message::Tombstone { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pre_compact_hook_block_is_recoverable_not_fatal() {
        // The blocking hook returns Block on PreCompact. compact_with_hooks
        // converts that to CompactError::Aborted, but our integration
        // distinguishes it from a real outer abort and continues.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "still alive".into(),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let blocker = Arc::new(crate::hook::RustHookHandler::new(
            "pre-compact-blocker",
            |event| match event {
                HookEvent::PreCompact { .. } => HookOutcome::Block,
                _ => HookOutcome::Ok,
            },
        ));
        let mut hooks = HookRunner::new();
        hooks.register(blocker);

        // Preload 4 so EarliestHalf target_count ≥ MIN_PARTIAL_COMPACT_TARGET.
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider.clone(), store.clone())
            .hooks(Arc::new(hooks))
            .build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // agent.compact.blocked notice (not a fatal Aborted error).
        assert!(events.iter().any(|e| matches!(
            e,
            Event::Notice { code, .. } if code == "agent.compact.blocked"
        )));
        // Loop continued and completed normally.
        assert!(matches!(events.last(), Some(Event::Result { .. })));
        assert_eq!(provider.remaining_turns(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn shared_compact_state_persists_failures_across_runs() {
        // Each run() spawns a fresh drive(), but a shared
        // Arc<Mutex<AutoCompactState>> survives across calls so that
        // circuit-breaker counts and the no_progress latch persist.
        // Verify by failing two compactions in run #1 (provider returns
        // empty), then making another call: the breaker advances to 2
        // failures rather than resetting to 0 each run.
        let provider = Arc::new(MockProvider::with_turns(vec![
            // run #1 turn 0 — failed compact (empty)
            vec![Event::Result {
                data: Default::default(),
            }],
            // run #1 turn 1 — assistant reply
            vec![
                Event::TextDelta { delta: "a".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
            // run #2 turn 0 — failed compact (empty)
            vec![Event::Result {
                data: Default::default(),
            }],
            // run #2 turn 1 — assistant reply
            vec![
                Event::TextDelta { delta: "b".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let shared_state = Arc::new(Mutex::new(AutoCompactState::new()));

        let qloop = compact_loop_builder(provider.clone(), store.clone())
            .compact_state(shared_state.clone())
            .build();
        let mut stream = qloop.run("first", AbortController::new()).await.unwrap();
        while stream.next().await.is_some() {}

        // After run #1: 1 compact failure recorded.
        assert_eq!(shared_state.lock().unwrap().consecutive_failures, 1);

        // Round 2: rebuild the loop sharing the same state + store.
        let qloop2 = compact_loop_builder(provider.clone(), store.clone())
            .compact_state(shared_state.clone())
            .build();
        let mut stream2 = qloop2.run("second", AbortController::new()).await.unwrap();
        while stream2.next().await.is_some() {}

        // After run #2: counter should be 2 (latched across runs).
        assert_eq!(shared_state.lock().unwrap().consecutive_failures, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_skipped_when_target_below_minimum() {
        // 3 messages total → EarliestHalf target = 1 < MIN_PARTIAL_COMPACT_TARGET.
        // Provider scripts only the assistant turn; auto-compaction
        // must NOT consume an extra turn.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "small".into(),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let store = preload_store(vec![user_text("u1"), assistant_text("a1")]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Only the assistant turn was consumed (not a compaction turn).
        assert_eq!(provider.remaining_turns(), 0);
        // No compact notice fired.
        assert!(!events.iter().any(
            |e| matches!(e, Event::Notice { code, .. } if code.starts_with("agent.compact."))
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_fires_at_minimum_target_boundary() {
        // 4 messages total → EarliestHalf target = 2 == MIN_PARTIAL_COMPACT_TARGET
        // (the equality boundary). Should fire, not skip.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::TextDelta {
                    delta: compact_response().into(),
                },
                Event::Result {
                    data: Default::default(),
                },
            ],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        // Preload 3 → run() pushes 1 → snapshot len = 4, mid = 2 == MIN.
        let store = preload_store(vec![user_text("u1"), assistant_text("a1"), user_text("u2")]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Both turns consumed (compact + assistant).
        assert_eq!(provider.remaining_turns(), 0);
        // agent.compact.ok notice fired.
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.ok")));
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
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "echo"));

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

    fn user_tool_result(tu_id: &str, payload: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tu_id.into(),
                content: agent_tool_result_text(payload),
                is_error: false,
            }],
        }
    }

    fn agent_tool_result_text(payload: &str) -> crate::message::ToolResultContent {
        crate::message::ToolResultContent::Text(payload.into())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn microcompact_ladder_skips_llm_compaction_when_enough() {
        // One huge OLD tool result (~100K tokens) + 5 small recent ones.
        // model_max_tokens = 50_000 → auto threshold = 17_000.
        // Pre-micro total ≈ 100K ≥ 17K → fires; clearing tu_0 alone drops
        // the total below 17K → the LLM compaction turn must NOT run.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta { delta: "ok".into() },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let mut preload = vec![user_tool_result("tu_0", &"x".repeat(400_000))];
        for i in 1..6 {
            preload.push(user_tool_result(&format!("tu_{i}"), &"y".repeat(600)));
        }
        let store = preload_store(preload);
        let qloop = QueryLoop::builder(provider.clone(), "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .model_max_tokens(50_000)
            .microcompact(true)
            .build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Only the assistant turn was consumed — no LLM compaction.
        assert_eq!(provider.remaining_turns(), 0);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.micro")));
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.ok")));
        // tu_0's payload was cleared in the store; no tombstones exist.
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert!(!snap.iter().any(|m| matches!(m, Message::Tombstone { .. })));
        match &snap[0] {
            Message::User { content, .. } => match &content[0] {
                ContentBlock::ToolResult { content, .. } => {
                    assert_eq!(
                        content,
                        &agent_tool_result_text(crate::compact::CLEARED_PLACEHOLDER)
                    );
                }
                other => panic!("expected tool result, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_promotes_analysis_to_session_memory() {
        let tagged = "<analysis>\n\
            - DECISION: rebuild MessageStore on microcompact.\n\
            - REQUIREMENT: never include co-author lines in commits.\n\
            - OPEN QUESTION: split loop_.rs into submodules?\n\
            </analysis>\n\
            <summary>Working on the compact ladder; decisions recorded.</summary>";
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::TextDelta {
                    delta: tagged.into(),
                },
                Event::Result {
                    data: Default::default(),
                },
            ],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let mem = Arc::new(crate::compact::InMemoryStore::new());
        let qloop = compact_loop_builder(provider.clone(), store)
            .session_memory(mem.clone())
            .build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.memory")));
        let entries = mem.list().await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].kind, crate::compact::SessionMemoryKind::Decision);
        assert_eq!(
            entries[1].kind,
            crate::compact::SessionMemoryKind::Requirement
        );
        assert_eq!(
            entries[2].kind,
            crate::compact::SessionMemoryKind::OpenQuestion
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_instructions_reach_the_summarization_prompt() {
        // Captures every StreamRequest's system prompt.
        #[derive(Debug)]
        struct CapturingProvider {
            systems: Arc<std::sync::Mutex<Vec<Option<String>>>>,
            turns: std::sync::Mutex<Vec<Vec<Event>>>,
        }
        #[async_trait]
        impl Provider for CapturingProvider {
            fn id(&self) -> &str {
                "capturing"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            async fn stream(
                &self,
                req: StreamRequest,
                _abort: AbortController,
            ) -> Result<Box<dyn EventStream>, AgentError> {
                self.systems.lock().unwrap().push(req.system.clone());
                let turn = self.turns.lock().unwrap().remove(0);
                Ok(Box::new(futures::stream::iter(turn.into_iter().map(Ok))))
            }
        }
        let systems = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            systems: systems.clone(),
            turns: std::sync::Mutex::new(vec![
                vec![
                    Event::TextDelta {
                        delta: "<analysis>- did X</analysis><summary>Did X.</summary>".into(),
                    },
                    Event::Result {
                        data: Default::default(),
                    },
                ],
                vec![
                    Event::TextDelta { delta: "ok".into() },
                    Event::Result {
                        data: ResultData {
                            stop_reason: Some("end_turn".into()),
                            ..Default::default()
                        },
                    },
                ],
            ]),
        });
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider, store)
            .compact_instructions("ALWAYS TAG REQUIREMENT BULLETS")
            .build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
        // First request is the compaction call; its system prompt must
        // carry the custom instructions appended to the vendor prompt.
        let captured = systems.lock().unwrap();
        let first = captured[0].as_deref().expect("compact call has a system");
        assert!(first.contains("ALWAYS TAG REQUIREMENT BULLETS"));
    }
}
