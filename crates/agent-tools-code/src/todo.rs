//! `TodoWrite` — in-memory shared todo list.
//!
//! Mirrors Claude Code's TodoWrite tool. The model uses it to plan
//! a multi-step task: each call replaces the entire list, so the
//! model can re-emit the latest state with statuses progressed.
//!
//! State lives in an `Arc<RwLock<Vec<TodoItem>>>` shared between
//! the tool instance and the host. Hosts read the current list to
//! render progress in their UI; the tool writes when the model
//! invokes it.
//!
//! `Mutating` — the list IS state, even though it lives in memory.
//! Hosts that don't want the model to plan can simply omit
//! `TodoWriteTool` from their registry.

use std::sync::Arc;

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::RwLock;

/// One todo entry. `subject` is the user-visible title; the optional
/// `description` carries detail. `status` advances through pending →
/// in_progress → completed (or `cancelled`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    pub subject: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub status: TodoStatus,
    /// Optional stable id assigned by the model; if missing, the
    /// tool synthesizes a sequential one per call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TodoStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

/// Shared todo state. Cheap to clone (Arc) — host keeps one copy,
/// the tool owns another.
#[derive(Debug, Clone, Default)]
pub struct TodoState {
    inner: Arc<RwLock<Vec<TodoItem>>>,
}

impl TodoState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the current list. Cheap clone of the Vec.
    pub async fn snapshot(&self) -> Vec<TodoItem> {
        self.inner.read().await.clone()
    }

    /// Replace the list wholesale. The TodoWrite tool calls this
    /// on every invocation; hosts can also call it directly to
    /// inject planning state programmatically.
    pub async fn set(&self, items: Vec<TodoItem>) {
        *self.inner.write().await = items;
    }

    /// Number of items currently held.
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }
}

#[derive(Debug)]
pub struct TodoWriteTool {
    state: TodoState,
}

impl TodoWriteTool {
    pub fn new(state: TodoState) -> Self {
        Self { state }
    }

    /// Convenience: construct a tool + return its (cloneable) state
    /// so the host can read snapshots without re-plumbing.
    pub fn with_fresh_state() -> (Self, TodoState) {
        let s = TodoState::new();
        (Self::new(s.clone()), s)
    }
}

#[derive(Debug, Deserialize)]
struct TodoWriteInput {
    todos: Vec<TodoItem>,
}

const MAX_ITEMS: usize = 100;

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "TodoWrite"
    }
    fn description(&self) -> &str {
        "Replace the planning todo list with `todos`. Each call should re-emit the full list with updated statuses."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "description": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"]
                            },
                            "id": {"type": "string"}
                        },
                        "required": ["subject"]
                    }
                }
            },
            "required": ["todos"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: TodoWriteInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("TodoWrite invalid input: {e}")))?;
        if parsed.todos.len() > MAX_ITEMS {
            return Err(AgentError::other(format!(
                "TodoWrite: list capped at {MAX_ITEMS} items, got {}",
                parsed.todos.len()
            )));
        }
        // Synthesize ids for items that don't carry one — keeps the
        // host's UI keying stable across edits.
        let with_ids: Vec<TodoItem> = parsed
            .todos
            .into_iter()
            .enumerate()
            .map(|(i, mut item)| {
                if item.subject.trim().is_empty() {
                    return Err(AgentError::other(format!(
                        "TodoWrite: item {i} has empty subject"
                    )));
                }
                if item.id.is_none() {
                    item.id = Some(format!("todo_{i}"));
                }
                Ok(item)
            })
            .collect::<Result<Vec<_>, AgentError>>()?;

        self.state.set(with_ids.clone()).await;

        // Status-bucket counts so the host (or the model) can quickly
        // see "5 done / 2 in_progress / 3 pending" without re-counting.
        let mut counts = StatusCounts::default();
        for item in &with_ids {
            counts.bump(item.status);
        }
        Ok(json!({
            "todos": with_ids,
            "counts": {
                "pending": counts.pending,
                "in_progress": counts.in_progress,
                "completed": counts.completed,
                "cancelled": counts.cancelled,
                "total": with_ids.len(),
            }
        }))
    }
}

#[derive(Debug, Default)]
struct StatusCounts {
    pending: u32,
    in_progress: u32,
    completed: u32,
    cancelled: u32,
}

impl StatusCounts {
    fn bump(&mut self, s: TodoStatus) {
        match s {
            TodoStatus::Pending => self.pending += 1,
            TodoStatus::InProgress => self.in_progress += 1,
            TodoStatus::Completed => self.completed += 1,
            TodoStatus::Cancelled => self.cancelled += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::abort::AbortController;
    use std::num::NonZeroUsize;

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
        }
    }

    #[tokio::test]
    async fn todo_write_replaces_list_wholesale() {
        let (tool, state) = TodoWriteTool::with_fresh_state();
        // First write
        tool.call(
            &ctx(),
            json!({"todos": [
                {"subject": "step 1", "status": "pending"},
                {"subject": "step 2", "status": "pending"}
            ]}),
        )
        .await
        .unwrap();
        assert_eq!(state.len().await, 2);
        // Second write replaces (not appends)
        tool.call(
            &ctx(),
            json!({"todos": [{"subject": "only", "status": "in_progress"}]}),
        )
        .await
        .unwrap();
        assert_eq!(state.len().await, 1);
        let snap = state.snapshot().await;
        assert_eq!(snap[0].subject, "only");
        assert_eq!(snap[0].status, TodoStatus::InProgress);
    }

    #[tokio::test]
    async fn todo_write_status_counts_in_response() {
        let (tool, _state) = TodoWriteTool::with_fresh_state();
        let out = tool
            .call(
                &ctx(),
                json!({"todos": [
                    {"subject": "a", "status": "pending"},
                    {"subject": "b", "status": "in_progress"},
                    {"subject": "c", "status": "completed"},
                    {"subject": "d", "status": "completed"}
                ]}),
            )
            .await
            .unwrap();
        assert_eq!(out["counts"]["pending"], 1);
        assert_eq!(out["counts"]["in_progress"], 1);
        assert_eq!(out["counts"]["completed"], 2);
        assert_eq!(out["counts"]["cancelled"], 0);
        assert_eq!(out["counts"]["total"], 4);
    }

    #[tokio::test]
    async fn todo_write_synthesizes_ids_for_missing() {
        let (tool, state) = TodoWriteTool::with_fresh_state();
        tool.call(
            &ctx(),
            json!({"todos": [
                {"subject": "no-id"},
                {"subject": "with-id", "id": "explicit"}
            ]}),
        )
        .await
        .unwrap();
        let snap = state.snapshot().await;
        assert_eq!(snap[0].id.as_deref(), Some("todo_0"));
        assert_eq!(snap[1].id.as_deref(), Some("explicit"));
    }

    #[tokio::test]
    async fn todo_write_default_status_pending() {
        let (tool, state) = TodoWriteTool::with_fresh_state();
        tool.call(&ctx(), json!({"todos": [{"subject": "x"}]}))
            .await
            .unwrap();
        let snap = state.snapshot().await;
        assert_eq!(snap[0].status, TodoStatus::Pending);
    }

    #[tokio::test]
    async fn todo_write_rejects_empty_subject() {
        let (tool, _state) = TodoWriteTool::with_fresh_state();
        let err = tool
            .call(&ctx(), json!({"todos": [{"subject": "  "}]}))
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("empty subject"));
    }

    #[tokio::test]
    async fn todo_write_rejects_too_many_items() {
        let (tool, _state) = TodoWriteTool::with_fresh_state();
        let mut many = Vec::with_capacity(MAX_ITEMS + 1);
        for i in 0..=MAX_ITEMS {
            many.push(json!({"subject": format!("step {i}")}));
        }
        let err = tool
            .call(&ctx(), json!({"todos": many}))
            .await
            .expect_err("over cap");
        assert!(err.to_string().contains("capped"));
    }

    #[tokio::test]
    async fn todo_state_independent_clones_share_storage() {
        let (tool, state) = TodoWriteTool::with_fresh_state();
        let state2 = state.clone();
        tool.call(
            &ctx(),
            json!({"todos": [{"subject": "shared", "status": "in_progress"}]}),
        )
        .await
        .unwrap();
        // Reading via the second handle sees the same state.
        let snap = state2.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].subject, "shared");
    }

    #[tokio::test]
    async fn todo_write_classified_mutating() {
        let (tool, _state) = TodoWriteTool::with_fresh_state();
        assert_eq!(tool.safety_class(), SafetyClass::Mutating);
    }

    #[tokio::test]
    async fn todo_status_serde_round_trip() {
        for s in [
            TodoStatus::Pending,
            TodoStatus::InProgress,
            TodoStatus::Completed,
            TodoStatus::Cancelled,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: TodoStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, s);
        }
    }

    #[tokio::test]
    async fn todo_state_set_directly_works() {
        let state = TodoState::new();
        state
            .set(vec![TodoItem {
                subject: "host-injected".into(),
                description: Some("not via tool".into()),
                status: TodoStatus::InProgress,
                id: Some("id-1".into()),
            }])
            .await;
        let snap = state.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].subject, "host-injected");
    }
}
