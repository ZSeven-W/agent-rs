//! Swarm task descriptors (Phase 6 / Task 6.3).
//!
//! A `SwarmTask` is what a leader hands to a worker via the mailbox.
//! Distinct from [`crate::tool::Tool`] (a tool is something the LLM
//! calls; a swarm task is a unit of work assigned to another agent).

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SwarmTaskPriority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SwarmTaskStatus {
    #[default]
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SwarmTask {
    pub id: Uuid,
    pub parent_agent: String,
    pub assignee: Option<String>,
    pub assignment: String,
    pub priority: SwarmTaskPriority,
    pub status: SwarmTaskStatus,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub result: serde_json::Value,
}

impl SwarmTask {
    pub fn new(parent_agent: impl Into<String>, assignment: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            parent_agent: parent_agent.into(),
            assignee: None,
            assignment: assignment.into(),
            priority: SwarmTaskPriority::Normal,
            status: SwarmTaskStatus::Pending,
            created_at_ms: now_ms(),
            completed_at_ms: None,
            result: serde_json::Value::Null,
        }
    }

    pub fn with_priority(mut self, priority: SwarmTaskPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn assign_to(mut self, agent: impl Into<String>) -> Self {
        self.assignee = Some(agent.into());
        self
    }

    pub fn mark_in_progress(&mut self) {
        self.status = SwarmTaskStatus::InProgress;
    }

    pub fn mark_completed(&mut self, result: serde_json::Value) {
        self.status = SwarmTaskStatus::Completed;
        self.completed_at_ms = Some(now_ms());
        self.result = result;
    }

    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = SwarmTaskStatus::Failed;
        self.completed_at_ms = Some(now_ms());
        self.result = serde_json::json!({ "error": error.into() });
    }

    pub fn mark_cancelled(&mut self) {
        self.status = SwarmTaskStatus::Cancelled;
        self.completed_at_ms = Some(now_ms());
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            SwarmTaskStatus::Completed | SwarmTaskStatus::Failed | SwarmTaskStatus::Cancelled
        )
    }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_task_has_pending_status() {
        let t = SwarmTask::new("leader", "fix the build");
        assert_eq!(t.status, SwarmTaskStatus::Pending);
        assert_eq!(t.priority, SwarmTaskPriority::Normal);
        assert!(t.assignee.is_none());
        assert!(!t.is_terminal());
    }

    #[test]
    fn priority_ordering_critical_higher_than_low() {
        assert!(SwarmTaskPriority::Critical > SwarmTaskPriority::Low);
        assert!(SwarmTaskPriority::High > SwarmTaskPriority::Normal);
    }

    #[test]
    fn lifecycle_progressions() {
        let mut t = SwarmTask::new("leader", "x").assign_to("worker-1");
        assert_eq!(t.assignee.as_deref(), Some("worker-1"));
        t.mark_in_progress();
        assert_eq!(t.status, SwarmTaskStatus::InProgress);
        t.mark_completed(serde_json::json!({"ok": true}));
        assert_eq!(t.status, SwarmTaskStatus::Completed);
        assert!(t.is_terminal());
        assert!(t.completed_at_ms.is_some());
    }

    #[test]
    fn failed_carries_error() {
        let mut t = SwarmTask::new("leader", "x");
        t.mark_failed("disk full");
        assert_eq!(t.status, SwarmTaskStatus::Failed);
        assert_eq!(t.result["error"], "disk full");
    }

    #[test]
    fn serde_roundtrip() {
        let t = SwarmTask::new("leader", "ship it")
            .with_priority(SwarmTaskPriority::High)
            .assign_to("alice");
        let j = serde_json::to_string(&t).unwrap();
        let back: SwarmTask = serde_json::from_str(&j).unwrap();
        assert_eq!(t, back);
    }
}
