//! Tool trait + dependency-injection context (Phase 2 / Task 2.2).
//!
//! Tools live **outside** this crate. Downstream applications supply their
//! own implementations (canvas / file / shell / grep / git / web / etc.)
//! and register them with [`ToolRegistry`]. This module only defines the
//! contract — name, description, input schema, async call — and the
//! per-call DI context.

mod registry;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::file_cache::FileStateCache;
use crate::hook::HookRunner;
use crate::permission::PermissionManager;

pub use registry::ToolRegistry;

/// Per-call dependency-injection context. The QueryEngine builds one of
/// these once per turn and passes a reference into every tool invocation.
#[derive(Debug, Clone)]
pub struct ToolUseContext {
    /// Working directory the tool should resolve relative paths against.
    pub cwd: PathBuf,
    /// Abort controller scoped to this turn (or finer). Tools should
    /// `tokio::select!` on `abort.cancelled()` for prompt cancellation.
    pub abort: AbortController,
    /// Shared read cache for file-based tools.
    pub file_cache: Arc<FileStateCache>,
    /// 7-step permission chain (Phase 3 stub today; real impl in 3.1).
    pub permissions: Arc<PermissionManager>,
    /// Typed hook registry (Phase 3 stub today; real impl in 3.x).
    pub hooks: Arc<HookRunner>,
}

impl ToolUseContext {
    /// Convenience constructor for tests + simple integrations. Real
    /// usage typically goes through `QueryEngine` (Phase 2 batch E) which
    /// builds the context per turn from session-scoped state.
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            abort: AbortController::new(),
            file_cache: Arc::new(FileStateCache::new(
                std::num::NonZeroUsize::new(64).unwrap(),
                8 * 1024 * 1024, // 8 MiB
            )),
            permissions: Arc::new(PermissionManager::new()),
            hooks: Arc::new(HookRunner::new()),
        }
    }
}

/// A tool the LLM can invoke during a turn.
///
/// Implementations are typically struct values stored as `Arc<dyn Tool>`
/// inside [`ToolRegistry`]. Each tool declares a stable name, a free-text
/// description (used in the LLM's tool selection prompt), and a JSON
/// Schema for inputs (use [`crate::json::schema`] to generate from a
/// struct that derives [`schemars::JsonSchema`]).
///
/// Errors returned from `call` should already be human-readable — the
/// QueryEngine surfaces them as `Event::ToolResult { ok: false, ... }`
/// without further unwrapping.
#[async_trait]
pub trait Tool: Send + Sync + std::fmt::Debug {
    /// Stable identifier. Must match the `name` field that providers
    /// echo back in `tool_use` events.
    fn name(&self) -> &str;

    /// Free-text description shown to the LLM. Keep concise (<200
    /// chars) and oriented around when to invoke, not how the impl
    /// works.
    fn description(&self) -> &str;

    /// JSON Schema (draft 2020-12) for the input payload. The
    /// QueryEngine forwards this to the provider so the LLM knows what
    /// fields to fill.
    fn input_schema(&self) -> serde_json::Value;

    /// Invoke the tool. The return value is forwarded as
    /// `Event::ToolResult { output, ok: true }`; an `Err` is surfaced as
    /// `Event::ToolResult { ok: false, output: { "error": "..." } }`.
    async fn call(
        &self,
        ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Returns its input unchanged."
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn echo_tool_through_trait_object() {
        let t: Arc<dyn Tool> = Arc::new(EchoTool);
        let ctx = ToolUseContext::new("/tmp");
        let out = t
            .call(&ctx, serde_json::json!({"hello": "world"}))
            .await
            .unwrap();
        assert_eq!(out, serde_json::json!({"hello": "world"}));
        assert_eq!(t.name(), "echo");
    }

    #[tokio::test]
    async fn context_abort_propagates_clones() {
        let ctx = ToolUseContext::new("/tmp");
        let cloned = ctx.clone();
        ctx.abort.abort_with_reason("cancel");
        assert!(cloned.abort.is_aborted());
        assert_eq!(cloned.abort.reason().as_deref(), Some("cancel"));
    }
}
