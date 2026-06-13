//! `Task` — let the model spawn a child `QueryLoop` to delegate
//! a sub-task with isolated context.
//!
//! Modeled on Claude Code's `Task` tool. The model picks an
//! `agent_type` and supplies a `description` + `prompt`; the host's
//! `TaskAgentFactory` decides which `Provider` / model / tool
//! registry / system prompt the child runs with. The child loop
//! runs to completion and the aggregated assistant text is returned
//! as the tool result, so the parent loop sees a single tool reply
//! with the child's final answer — exactly like a tool call to any
//! other tool.
//!
//! Why a factory instead of holding a single `Provider`? In real
//! deployments hosts wire up multiple "agent shapes" (researcher /
//! reviewer / planner) — different system prompts, different tool
//! allowlists, sometimes different models. The factory layer is
//! how the host expresses that vocabulary; the tool stays generic.
//!
//! Recursion safety: the host configures `max_depth` and the tool
//! reads / increments a depth counter on `ctx`. We piggy-back on
//! `ToolUseContext::abort` for cancellation — when the parent
//! loop's abort fires, the child loop's stream surfaces an
//! `AgentError::Aborted` and we forward it.
//!
//! Output shape: `{output, agent_type, usage_input_tokens,
//! usage_output_tokens, stop_reason}`. Optional usage fields stay
//! present even when zero so the parent model has a stable
//! schema to reason about.

use std::path::PathBuf;
use std::sync::Arc;

use agent::error::AgentError;
use agent::file_cache::FileStateCache;
use agent::hook::HookRunner;
use agent::permission::PermissionManager;
use agent::provider::Provider;
use agent::query::QueryLoop;
use agent::stream::{Event, ResultData};
use agent::tool::{SafetyClass, Tool, ToolRegistry, ToolUseContext};
use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

/// Configuration for a single agent shape (researcher, reviewer,
/// planner, …). Returned from [`TaskAgentFactory::build`].
#[derive(Debug, Clone)]
pub struct TaskAgentConfig {
    pub provider: Arc<dyn Provider>,
    pub model: String,
    pub tools: Arc<ToolRegistry>,
    /// Optional system prompt for the child.
    pub system: Option<String>,
    /// Optional cap on the child's tool-loop iterations. Defaults
    /// to QueryLoop's default when `None`.
    pub max_iterations: Option<usize>,
    /// Optional shared permission manager. When `None`, the child
    /// gets a fresh one — useful when the host wants the sub-agent
    /// to fail-open on every permission decision (delegated trust)
    /// or fail-closed (ask-the-user). Either is a host policy.
    pub permissions: Option<Arc<PermissionManager>>,
    /// Optional working directory for the child loop. When `None`,
    /// QueryLoop's default applies. Hosts pass the parent cwd so the
    /// child's path resolution and tool sandboxing match the parent.
    pub cwd: Option<PathBuf>,
    /// Optional shared file-state cache. When `None`, the child gets
    /// its own. Sharing the parent's keeps read-before-write tracking
    /// consistent across parent and child.
    pub file_cache: Option<Arc<FileStateCache>>,
    /// Optional shared hook runner. When `None`, the child runs without
    /// hooks. Hosts pass the parent's runner so edit history, background-
    /// shell tracking, and external hook blockers also apply to the child.
    pub hooks: Option<Arc<HookRunner>>,
}

/// Factory for resolving an `agent_type` string to a concrete
/// child-loop config. Hosts implement this to enumerate which
/// sub-agents the model is allowed to summon.
#[async_trait]
pub trait TaskAgentFactory: Send + Sync + std::fmt::Debug {
    /// Build the child config. Implementations should return an
    /// error if `agent_type` isn't recognized — the tool surfaces
    /// it back to the model so it can pick a different one.
    async fn build(&self, agent_type: &str) -> Result<TaskAgentConfig, AgentError>;
}

/// Default cap on Task-tool recursion depth. Hosts can override via
/// [`TaskTool::with_max_depth`].
pub const DEFAULT_MAX_DEPTH: usize = 3;

#[derive(Debug)]
pub struct TaskTool {
    factory: Arc<dyn TaskAgentFactory>,
    max_depth: usize,
}

impl TaskTool {
    pub fn new(factory: Arc<dyn TaskAgentFactory>) -> Self {
        Self {
            factory,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    pub fn with_max_depth(mut self, n: usize) -> Self {
        self.max_depth = n.max(1);
        self
    }
}

#[derive(Debug, Deserialize)]
struct TaskInput {
    /// Short label for the sub-task (1 line). Surfaces in
    /// progress / hook output, not used to drive routing.
    #[serde(default)]
    description: Option<String>,
    /// User-facing instructions for the child loop. This becomes
    /// the child's first user message.
    prompt: String,
    /// Which agent shape to summon. Resolved against the
    /// `TaskAgentFactory`.
    agent_type: String,
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &str {
        "Task"
    }
    fn description(&self) -> &str {
        "Delegate a sub-task to a fresh child agent. Pick an `agent_type` registered with the host. The child runs to completion and returns its final assistant text."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Short label for the sub-task (1 line)."},
                "prompt": {"type": "string", "description": "Instructions for the child agent."},
                "agent_type": {"type": "string", "description": "Agent shape to summon (host-defined)."}
            },
            "required": ["prompt", "agent_type"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        // Child loops can call mutating tools, but Task itself only
        // forwards a prompt. The child's tools carry their own
        // SafetyClass; the parent's permission gate evaluates Task
        // independently of what the child ends up calling. Mark
        // Mutating so the parent permission rules apply by default;
        // hosts who disagree can wrap with PermissionMatcher.
        SafetyClass::Mutating
    }
    async fn call(&self, ctx: &ToolUseContext, input: Value) -> Result<Value, AgentError> {
        let parsed: TaskInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("Task invalid input: {e}")))?;
        if parsed.prompt.trim().is_empty() {
            return Err(AgentError::other("Task prompt must be non-empty"));
        }
        if parsed.agent_type.trim().is_empty() {
            return Err(AgentError::other("Task agent_type must be non-empty"));
        }

        // Recursion guard. Depth lives in a process-global thread-
        // local since `ToolUseContext` doesn't carry an extension
        // map. Single-threaded per-call execution is the norm for
        // the tool dispatch path, so this is fine.
        let depth = depth_state::current();
        if depth >= self.max_depth {
            return Err(AgentError::other(format!(
                "Task: max recursion depth {} reached (current depth {depth})",
                self.max_depth
            )));
        }

        let cfg = self
            .factory
            .build(&parsed.agent_type)
            .await
            .map_err(|e| AgentError::other(format!("Task factory: {e}")))?;

        let mut builder =
            QueryLoop::builder(cfg.provider.clone(), cfg.model.clone()).tools(cfg.tools.clone());
        if let Some(s) = cfg.system.as_deref() {
            builder = builder.system(s);
        }
        if let Some(n) = cfg.max_iterations {
            builder = builder.max_iterations(n);
        }
        if let Some(p) = cfg.permissions.clone() {
            builder = builder.permissions(p);
        }
        if let Some(c) = cfg.cwd.clone() {
            builder = builder.cwd(c);
        }
        if let Some(fc) = cfg.file_cache.clone() {
            builder = builder.file_cache(fc);
        }
        if let Some(h) = cfg.hooks.clone() {
            builder = builder.hooks(h);
        }

        let child = builder.build();
        // Forward the parent's abort to the child so cancelling the
        // outer loop also short-circuits the child.
        let abort = ctx.abort.clone();

        let _depth_guard = depth_state::Increment::new();
        let mut stream = child.run(parsed.prompt.clone(), abort).await?;

        let mut output = String::new();
        let mut usage_input = 0u32;
        let mut usage_output = 0u32;
        let mut stop_reason: Option<String> = None;
        let mut error: Option<AgentError> = None;
        while let Some(event) = stream.next().await {
            match event {
                Ok(Event::TextDelta { delta }) => output.push_str(&delta),
                Ok(Event::Usage {
                    input_tokens,
                    output_tokens,
                    ..
                }) => {
                    usage_input = usage_input.saturating_add(input_tokens);
                    usage_output = usage_output.saturating_add(output_tokens);
                }
                Ok(Event::Result {
                    data:
                        ResultData {
                            stop_reason: sr, ..
                        },
                }) => {
                    stop_reason = sr;
                }
                Ok(Event::Error { message, .. }) => {
                    error = Some(AgentError::other(format!("Task child error: {message}")));
                }
                Ok(_) => {}
                Err(e) => {
                    error = Some(e);
                }
            }
        }
        if let Some(e) = error {
            return Err(e);
        }
        Ok(json!({
            "output": output,
            "agent_type": parsed.agent_type,
            "description": parsed.description,
            "usage_input_tokens": usage_input,
            "usage_output_tokens": usage_output,
            "stop_reason": stop_reason,
        }))
    }
}

mod depth_state {
    //! Thread-local recursion depth for `Task`. The dispatch path
    //! runs each tool call on a single tokio task, so a thread-
    //! local works as long as we increment / decrement around the
    //! await point without yielding to a different task.

    use std::cell::Cell;

    thread_local! {
        static DEPTH: Cell<usize> = const { Cell::new(0) };
    }

    pub fn current() -> usize {
        DEPTH.with(|d| d.get())
    }

    /// RAII guard. Increments on construction, decrements on drop.
    pub struct Increment;

    impl Increment {
        pub fn new() -> Self {
            DEPTH.with(|d| d.set(d.get().saturating_add(1)));
            Self
        }
    }

    impl Drop for Increment {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::abort::AbortController;
    use agent::file_cache::FileStateCache;
    use agent::hook::HookRunner;
    use agent::permission::PermissionManager;
    use agent::provider::{ProviderCapabilities, StreamRequest};
    use agent::stream::{Event as AgentEvent, EventStream};
    use futures::stream;
    use std::num::NonZeroUsize;

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
        }
    }

    /// Provider that emits a single TextDelta + Usage + Result and
    /// terminates. Lets us drive the child QueryLoop to completion
    /// without a real LLM.
    #[derive(Debug)]
    struct StubProvider {
        reply: String,
    }

    #[async_trait]
    impl Provider for StubProvider {
        fn id(&self) -> &str {
            "stub"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        async fn stream(
            &self,
            _req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            let reply = self.reply.clone();
            let events: Vec<Result<AgentEvent, AgentError>> = vec![
                Ok(AgentEvent::TextDelta { delta: reply }),
                Ok(AgentEvent::Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read: 0,
                    cache_create: 0,
                }),
                Ok(AgentEvent::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                }),
            ];
            Ok(Box::new(stream::iter(events)))
        }
    }

    #[derive(Debug)]
    struct StubFactory {
        provider: Arc<dyn Provider>,
    }

    #[async_trait]
    impl TaskAgentFactory for StubFactory {
        async fn build(&self, agent_type: &str) -> Result<TaskAgentConfig, AgentError> {
            if agent_type == "researcher" {
                Ok(TaskAgentConfig {
                    provider: self.provider.clone(),
                    model: "stub-model".into(),
                    tools: Arc::new(ToolRegistry::new()),
                    system: Some("you are a researcher".into()),
                    max_iterations: Some(2),
                    permissions: None,
                    cwd: None,
                    file_cache: None,
                    hooks: None,
                })
            } else {
                Err(AgentError::other(format!(
                    "unknown agent_type: {agent_type}"
                )))
            }
        }
    }

    fn factory(reply: &str) -> Arc<StubFactory> {
        Arc::new(StubFactory {
            provider: Arc::new(StubProvider {
                reply: reply.into(),
            }),
        })
    }

    #[tokio::test]
    async fn task_returns_child_assistant_text() {
        let tool = TaskTool::new(factory("the result is 42"));
        let out = tool
            .call(
                &ctx(),
                json!({"prompt": "compute", "agent_type": "researcher"}),
            )
            .await
            .unwrap();
        assert_eq!(out["output"], "the result is 42");
        assert_eq!(out["agent_type"], "researcher");
        assert_eq!(out["usage_input_tokens"], 10);
        assert_eq!(out["usage_output_tokens"], 5);
        assert_eq!(out["stop_reason"], "end_turn");
    }

    #[tokio::test]
    async fn task_unknown_agent_type_errors() {
        let tool = TaskTool::new(factory("x"));
        let err = tool
            .call(&ctx(), json!({"prompt": "x", "agent_type": "bogus"}))
            .await
            .expect_err("unknown");
        assert!(err.to_string().contains("unknown agent_type"));
    }

    #[tokio::test]
    async fn task_rejects_empty_prompt_or_agent_type() {
        let tool = TaskTool::new(factory("x"));
        let err = tool
            .call(&ctx(), json!({"prompt": "  ", "agent_type": "researcher"}))
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("non-empty"));
        let err = tool
            .call(&ctx(), json!({"prompt": "x", "agent_type": "  "}))
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("non-empty"));
    }

    #[tokio::test]
    async fn task_classified_mutating() {
        let tool = TaskTool::new(factory("x"));
        assert_eq!(tool.safety_class(), SafetyClass::Mutating);
    }

    #[tokio::test]
    async fn task_max_depth_caps_recursion() {
        let tool = TaskTool::new(factory("x")).with_max_depth(2);
        // Synthesize the depth state to simulate already being 2
        // levels in.
        let _g1 = depth_state::Increment::new();
        let _g2 = depth_state::Increment::new();
        let err = tool
            .call(
                &ctx(),
                json!({"prompt": "go deeper", "agent_type": "researcher"}),
            )
            .await
            .expect_err("depth exceeded");
        assert!(err.to_string().contains("max recursion"));
    }
}
