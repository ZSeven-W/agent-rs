//! Task data model.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable per-task identifier. UUID v4 by default; tests can hand in
/// a fixed string for deterministic graph assertions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
    Canceled,
}

impl TaskStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Canceled)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedTask {
    pub id: TaskId,
    /// Short imperative title — what the task does.
    pub subject: String,
    /// Longer explanation surfaced when the task is selected.
    #[serde(default)]
    pub description: String,
    /// Present-continuous form for the spinner / status line ("Running tests").
    #[serde(default)]
    pub active_form: Option<String>,
    pub status: TaskStatus,
    /// Tasks that must reach `Completed` before this one can move
    /// out of `Pending`. Direct deps only — transitive resolution is
    /// the graph's job.
    #[serde(default)]
    pub blocked_by: BTreeSet<TaskId>,
    /// Tasks that this one blocks. The graph keeps both edge sets
    /// consistent on insert/remove.
    #[serde(default)]
    pub blocks: BTreeSet<TaskId>,
    /// Free-form metadata — owner, link, ETA, etc.
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub created_at_unix_ms: u64,
    pub updated_at_unix_ms: u64,
}

impl PlannedTask {
    pub fn new(subject: impl Into<String>) -> Self {
        let now = now_ms();
        Self {
            id: TaskId::new(),
            subject: subject.into(),
            description: String::new(),
            active_form: None,
            status: TaskStatus::Pending,
            blocked_by: BTreeSet::new(),
            blocks: BTreeSet::new(),
            metadata: Default::default(),
            created_at_unix_ms: now,
            updated_at_unix_ms: now,
        }
    }
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_id_round_trip_serde() {
        let id = TaskId::from_string("abc");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"abc\"");
        assert_eq!(serde_json::from_str::<TaskId>(&json).unwrap(), id);
    }

    #[test]
    fn task_status_terminal_set() {
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Canceled.is_terminal());
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::InProgress.is_terminal());
        assert!(!TaskStatus::Blocked.is_terminal());
    }

    #[test]
    fn new_task_starts_pending_with_timestamps() {
        let t = PlannedTask::new("hi");
        assert_eq!(t.status, TaskStatus::Pending);
        assert_eq!(t.created_at_unix_ms, t.updated_at_unix_ms);
    }

    #[test]
    fn task_serde_roundtrip() {
        let t = PlannedTask {
            id: TaskId::from_string("t1"),
            subject: "do x".into(),
            description: "details".into(),
            active_form: Some("doing x".into()),
            status: TaskStatus::InProgress,
            blocked_by: [TaskId::from_string("t0")].iter().cloned().collect(),
            blocks: BTreeSet::new(),
            metadata: serde_json::Map::new(),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
        };
        let json = serde_json::to_string(&t).unwrap();
        let parsed: PlannedTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }
}
