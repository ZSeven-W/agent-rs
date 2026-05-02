//! Task graph — registry + dependency tracking + status transitions.
//!
//! Cycle detection: `add_dep` rejects edges that would create a
//! cycle. The check uses depth-first traversal from the new
//! dependency target back to the source — O(V+E) but acceptable
//! given a typical graph has O(20) nodes.
//!
//! Status auto-transitions: when a task completes, every task that
//! had it as a blocker re-evaluates. If all blockers are now
//! `Completed`, the task moves from `Blocked → Pending` (so the
//! host can pick it up).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::task::{now_ms, PlannedTask, TaskId, TaskStatus};

#[derive(Debug, thiserror::Error)]
pub enum TaskGraphError {
    #[error("task `{0}` not found")]
    NotFound(TaskId),
    #[error("task `{0}` already exists")]
    Duplicate(TaskId),
    #[error("dependency would create cycle through `{cycle_through}`")]
    Cycle { cycle_through: TaskId },
    #[error("invalid status transition `{from:?}` → `{to:?}`")]
    BadTransition { from: TaskStatus, to: TaskStatus },
}

/// In-memory task graph. Cheap to clone — Arc-shared inner store.
#[derive(Debug, Clone, Default)]
pub struct TaskGraph {
    inner: Arc<Mutex<BTreeMap<TaskId, PlannedTask>>>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a fresh task. Returns `Duplicate` if the id is
    /// already in the graph.
    pub fn insert(&self, task: PlannedTask) -> Result<(), TaskGraphError> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.contains_key(&task.id) {
            return Err(TaskGraphError::Duplicate(task.id));
        }
        // If the task arrived with `blocked_by` already set, mirror
        // those edges into each blocker's `blocks` set + transition
        // to Blocked when appropriate.
        let id = task.id.clone();
        let blockers: Vec<TaskId> = task.blocked_by.iter().cloned().collect();
        let mut t = task;
        if !blockers.is_empty() && t.status == TaskStatus::Pending {
            t.status = TaskStatus::Blocked;
        }
        g.insert(id.clone(), t);
        for b in blockers {
            if let Some(blocker) = g.get_mut(&b) {
                blocker.blocks.insert(id.clone());
            }
        }
        Ok(())
    }

    pub fn get(&self, id: &TaskId) -> Option<PlannedTask> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.get(id).cloned()
    }

    pub fn list(&self) -> Vec<PlannedTask> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.values().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a `blocked_by` edge from `task` to `blocker`. Both must
    /// exist. Detects cycles: if `task` is reachable from `blocker`
    /// via the existing `blocks` graph, the new edge would create
    /// a cycle and is rejected.
    pub fn add_dep(&self, task: &TaskId, blocker: &TaskId) -> Result<(), TaskGraphError> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.contains_key(task) {
            return Err(TaskGraphError::NotFound(task.clone()));
        }
        if !g.contains_key(blocker) {
            return Err(TaskGraphError::NotFound(blocker.clone()));
        }
        // Cycle check: adding `task.blocked_by += blocker` mirrors
        // into `blocker.blocks += task`. If `task` already reaches
        // `blocker` via existing `blocks` edges, the new back-edge
        // closes a cycle. Search forward from `task` for `blocker`.
        if reachable(&g, task, blocker) {
            return Err(TaskGraphError::Cycle {
                cycle_through: blocker.clone(),
            });
        }
        let blocker_done = g
            .get(blocker)
            .map(|b| b.status == TaskStatus::Completed)
            .unwrap_or(false);
        let now = now_ms();
        let t = g.get_mut(task).unwrap();
        t.blocked_by.insert(blocker.clone());
        if !blocker_done && t.status == TaskStatus::Pending {
            t.status = TaskStatus::Blocked;
        }
        t.updated_at_unix_ms = now;
        let b = g.get_mut(blocker).unwrap();
        b.blocks.insert(task.clone());
        b.updated_at_unix_ms = now;
        Ok(())
    }

    /// Set status. Validates the transition and (when transitioning
    /// to Completed) auto-unblocks dependents whose remaining
    /// `blocked_by` set is now empty.
    pub fn set_status(&self, id: &TaskId, new_status: TaskStatus) -> Result<(), TaskGraphError> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let cur = g
            .get(id)
            .ok_or_else(|| TaskGraphError::NotFound(id.clone()))?
            .status;
        if !is_valid_transition(cur, new_status) {
            return Err(TaskGraphError::BadTransition {
                from: cur,
                to: new_status,
            });
        }
        let now = now_ms();
        let t = g.get_mut(id).unwrap();
        t.status = new_status;
        t.updated_at_unix_ms = now;
        if new_status == TaskStatus::Completed {
            // Unblock dependents.
            let dependents: Vec<TaskId> = t.blocks.iter().cloned().collect();
            for d in dependents {
                let still_blocked = g
                    .get(&d)
                    .map(|dep| {
                        dep.blocked_by.iter().any(|b| {
                            g.get(b)
                                .map(|x| x.status != TaskStatus::Completed)
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if !still_blocked {
                    if let Some(dep) = g.get_mut(&d) {
                        if dep.status == TaskStatus::Blocked {
                            dep.status = TaskStatus::Pending;
                            dep.updated_at_unix_ms = now;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn remove(&self, id: &TaskId) -> bool {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(removed) = g.remove(id) else {
            return false;
        };
        // Clean up dangling edges.
        for b in &removed.blocked_by {
            if let Some(t) = g.get_mut(b) {
                t.blocks.remove(id);
            }
        }
        for d in &removed.blocks {
            if let Some(t) = g.get_mut(d) {
                t.blocked_by.remove(id);
            }
        }
        true
    }

    /// Tasks whose status is `Pending` and whose `blocked_by` is
    /// empty — i.e., immediately runnable. Sorted by id for stable
    /// iteration.
    pub fn ready(&self) -> Vec<PlannedTask> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.values()
            .filter(|t| t.status == TaskStatus::Pending && t.blocked_by.is_empty())
            .cloned()
            .collect()
    }
}

fn reachable(g: &BTreeMap<TaskId, PlannedTask>, start: &TaskId, target: &TaskId) -> bool {
    let mut stack = vec![start.clone()];
    let mut visited: std::collections::BTreeSet<TaskId> = std::collections::BTreeSet::new();
    while let Some(cur) = stack.pop() {
        if &cur == target {
            return true;
        }
        if !visited.insert(cur.clone()) {
            continue;
        }
        if let Some(t) = g.get(&cur) {
            for n in &t.blocks {
                stack.push(n.clone());
            }
        }
    }
    false
}

fn is_valid_transition(from: TaskStatus, to: TaskStatus) -> bool {
    use TaskStatus::*;
    if from == to {
        return true;
    }
    matches!(
        (from, to),
        (Pending, InProgress)
            | (Pending, Blocked)
            | (Pending, Canceled)
            | (Blocked, Pending)
            | (Blocked, Canceled)
            | (InProgress, Completed)
            | (InProgress, Blocked)
            | (InProgress, Canceled)
    )
}

#[cfg(test)]
mod tests {
    use super::super::task::PlannedTask;
    use super::*;

    fn task(id: &str) -> PlannedTask {
        let mut t = PlannedTask::new(format!("subject-{id}"));
        t.id = TaskId::from_string(id);
        t
    }

    #[test]
    fn insert_round_trip() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        assert_eq!(g.len(), 1);
        assert!(g.get(&TaskId::from_string("a")).is_some());
    }

    #[test]
    fn duplicate_insert_errors() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        assert!(matches!(
            g.insert(task("a")).unwrap_err(),
            TaskGraphError::Duplicate(_)
        ));
    }

    #[test]
    fn add_dep_blocks_dependent_until_blocker_completes() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        g.insert(task("b")).unwrap();
        g.add_dep(&TaskId::from_string("b"), &TaskId::from_string("a"))
            .unwrap();
        // b is now blocked.
        assert_eq!(
            g.get(&TaskId::from_string("b")).unwrap().status,
            TaskStatus::Blocked
        );
        // Drive a → InProgress → Completed.
        g.set_status(&TaskId::from_string("a"), TaskStatus::InProgress)
            .unwrap();
        g.set_status(&TaskId::from_string("a"), TaskStatus::Completed)
            .unwrap();
        // b auto-unblocks to Pending.
        assert_eq!(
            g.get(&TaskId::from_string("b")).unwrap().status,
            TaskStatus::Pending
        );
    }

    #[test]
    fn cycle_detection() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        g.insert(task("b")).unwrap();
        g.insert(task("c")).unwrap();
        // a blocks b blocks c — adding c blocks a should cycle.
        g.add_dep(&TaskId::from_string("b"), &TaskId::from_string("a"))
            .unwrap();
        g.add_dep(&TaskId::from_string("c"), &TaskId::from_string("b"))
            .unwrap();
        assert!(matches!(
            g.add_dep(&TaskId::from_string("a"), &TaskId::from_string("c"))
                .unwrap_err(),
            TaskGraphError::Cycle { .. }
        ));
    }

    #[test]
    fn invalid_transition_errors() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        // Pending → Completed is not allowed (must go through InProgress).
        assert!(matches!(
            g.set_status(&TaskId::from_string("a"), TaskStatus::Completed)
                .unwrap_err(),
            TaskGraphError::BadTransition { .. }
        ));
    }

    #[test]
    fn remove_cleans_up_edges() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        g.insert(task("b")).unwrap();
        g.add_dep(&TaskId::from_string("b"), &TaskId::from_string("a"))
            .unwrap();
        assert!(g.remove(&TaskId::from_string("a")));
        // b's blocked_by must no longer reference a.
        let b = g.get(&TaskId::from_string("b")).unwrap();
        assert!(b.blocked_by.is_empty());
    }

    #[test]
    fn ready_returns_only_unblocked_pending() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        g.insert(task("b")).unwrap();
        g.add_dep(&TaskId::from_string("b"), &TaskId::from_string("a"))
            .unwrap();
        let ready = g.ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, TaskId::from_string("a"));
    }

    #[test]
    fn cancel_from_pending_or_blocked_or_in_progress() {
        let g = TaskGraph::new();
        g.insert(task("p")).unwrap();
        g.insert(task("b")).unwrap();
        g.insert(task("ip")).unwrap();
        g.add_dep(&TaskId::from_string("b"), &TaskId::from_string("p"))
            .unwrap();
        g.set_status(&TaskId::from_string("ip"), TaskStatus::InProgress)
            .unwrap();
        for id in ["p", "b", "ip"] {
            g.set_status(&TaskId::from_string(id), TaskStatus::Canceled)
                .unwrap();
            assert_eq!(
                g.get(&TaskId::from_string(id)).unwrap().status,
                TaskStatus::Canceled
            );
        }
    }

    #[test]
    fn add_dep_to_already_completed_blocker_keeps_pending() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        g.insert(task("b")).unwrap();
        g.set_status(&TaskId::from_string("a"), TaskStatus::InProgress)
            .unwrap();
        g.set_status(&TaskId::from_string("a"), TaskStatus::Completed)
            .unwrap();
        // Adding a as a blocker AFTER it completes should keep b
        // Pending (no real wait).
        g.add_dep(&TaskId::from_string("b"), &TaskId::from_string("a"))
            .unwrap();
        assert_eq!(
            g.get(&TaskId::from_string("b")).unwrap().status,
            TaskStatus::Pending
        );
    }

    #[test]
    fn add_dep_unknown_task_errors() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        assert!(matches!(
            g.add_dep(&TaskId::from_string("ghost"), &TaskId::from_string("a"))
                .unwrap_err(),
            TaskGraphError::NotFound(_)
        ));
    }

    #[test]
    fn list_returns_all_tasks() {
        let g = TaskGraph::new();
        g.insert(task("a")).unwrap();
        g.insert(task("b")).unwrap();
        assert_eq!(g.list().len(), 2);
    }
}
