//! Abort control — wraps [`tokio_util::sync::CancellationToken`] with an
//! optional reason string captured at abort time.
//!
//! Phase 1 / Task 1.4. Used by [`Provider::stream`](crate::provider::Provider)
//! and every long-running operation in later phases. Cheap to clone; child
//! controllers cascade-cancel from their parent but track their own reason.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
pub struct AbortController {
    token: CancellationToken,
    reason: Arc<Mutex<Option<String>>>,
    activity: TurnActivity,
}

/// Shared liveness and retry-safety signal for one root agent turn.
///
/// Child abort controllers deliberately share this state: provider events and
/// tool calls from nested `Task`/team loops are still activity of the root
/// turn. `side_effect_risk` is monotonic and conservative; once a mutating or
/// unclassified tool starts, replaying the root prompt is no longer safe.
#[derive(Clone, Debug)]
pub struct TurnActivity(Arc<TurnActivityInner>);

#[derive(Debug)]
struct TurnActivityInner {
    state: Mutex<TurnActivityState>,
    worker_count: watch::Sender<usize>,
}

#[derive(Debug, Clone, Copy)]
struct TurnActivityState {
    last_activity_at: Instant,
    side_effect_risk: bool,
    unresolved_external_work: bool,
    active_workers: usize,
}

/// RAII registration for a detached runtime worker that belongs to a turn.
/// Hosts can await [`TurnActivity::wait_for_quiescence`] after hard cancel so
/// acknowledging the stop cannot race nested task destructors.
#[derive(Debug)]
pub struct TurnWorkGuard {
    activity: TurnActivity,
    active: bool,
}

impl TurnActivity {
    pub fn new() -> Self {
        let (worker_count, _) = watch::channel(0);
        Self(Arc::new(TurnActivityInner {
            state: Mutex::new(TurnActivityState {
                last_activity_at: Instant::now(),
                side_effect_risk: false,
                unresolved_external_work: false,
                active_workers: 0,
            }),
            worker_count,
        }))
    }

    /// Record source-side progress before it enters any host/UI queue.
    pub fn pulse(&self) {
        if let Ok(mut state) = self.0.state.lock() {
            state.last_activity_at = Instant::now();
        }
    }

    pub fn last_activity_at(&self) -> Instant {
        self.0
            .state
            .lock()
            .map(|state| state.last_activity_at)
            .unwrap_or_else(|_| Instant::now())
    }

    /// Mark that this turn may already have changed external state.
    pub fn mark_side_effect_risk(&self) {
        if let Ok(mut state) = self.0.state.lock() {
            state.last_activity_at = Instant::now();
            state.side_effect_risk = true;
        }
    }

    pub fn side_effect_risk(&self) -> bool {
        self.0
            .state
            .lock()
            .map(|state| state.side_effect_risk)
            .unwrap_or(true)
    }

    /// Mark side effects that intentionally outlive the local tool future or
    /// whose remote cancellation cannot be proven. Scheduler hosts use this
    /// stronger latch to stop recurrence even when the turn itself succeeds.
    pub fn mark_unresolved_external_work(&self) {
        if let Ok(mut state) = self.0.state.lock() {
            state.last_activity_at = Instant::now();
            state.side_effect_risk = true;
            state.unresolved_external_work = true;
        }
    }

    pub fn unresolved_external_work(&self) -> bool {
        self.0
            .state
            .lock()
            .map(|state| state.unresolved_external_work)
            .unwrap_or(true)
    }

    /// Register one nested runtime worker. The returned guard must live inside
    /// the spawned future so Tokio cancellation drops it only after that
    /// future's process/tool cleanup guards have run.
    pub fn track_worker(&self) -> TurnWorkGuard {
        let active = if let Ok(mut state) = self.0.state.lock() {
            state.active_workers = state.active_workers.saturating_add(1);
            self.0.worker_count.send_replace(state.active_workers);
            true
        } else {
            false
        };
        TurnWorkGuard {
            activity: self.clone(),
            active,
        }
    }

    pub fn active_workers(&self) -> usize {
        self.0
            .state
            .lock()
            .map(|state| state.active_workers)
            .unwrap_or(usize::MAX)
    }

    /// Wait until every tracked driver, supervisor, and tool task has exited
    /// and dropped its cleanup guards.
    pub async fn wait_for_quiescence(&self) {
        let mut counts = self.0.worker_count.subscribe();
        loop {
            if *counts.borrow_and_update() == 0 {
                return;
            }
            if counts.changed().await.is_err() {
                return;
            }
        }
    }
}

impl Drop for TurnWorkGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Ok(mut state) = self.activity.0.state.lock() {
            state.active_workers = state.active_workers.saturating_sub(1);
            self.activity
                .0
                .worker_count
                .send_replace(state.active_workers);
        }
        self.active = false;
    }
}

impl Default for TurnActivity {
    fn default() -> Self {
        Self::new()
    }
}

impl AbortController {
    /// Create a new root controller. Not aborted, no reason.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            reason: Arc::new(Mutex::new(None)),
            activity: TurnActivity::new(),
        }
    }

    /// Create a child controller. Cancelling the parent cancels the child;
    /// cancelling the child does **not** cancel the parent. Each controller
    /// owns its own reason — a child cancelled-by-cascade has no reason
    /// unless something explicitly calls `abort_with_reason` on the child.
    pub fn child(&self) -> Self {
        Self {
            token: self.token.child_token(),
            reason: Arc::new(Mutex::new(None)),
            activity: self.activity.clone(),
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Cancel without a reason. Idempotent.
    pub fn abort(&self) {
        self.token.cancel();
    }

    /// Cancel with a human-readable reason. The reason is captured on the
    /// first call; subsequent calls are ignored (preserves the first cause).
    pub fn abort_with_reason(&self, reason: impl Into<String>) {
        if let Ok(mut guard) = self.reason.lock() {
            if guard.is_none() {
                *guard = Some(reason.into());
            }
        }
        self.token.cancel();
    }

    /// The reason set by [`Self::abort_with_reason`], if any.
    pub fn reason(&self) -> Option<String> {
        self.reason.lock().ok().and_then(|g| g.clone())
    }

    /// Handle used by hosts to supervise this turn without depending on a UI.
    pub fn activity(&self) -> TurnActivity {
        self.activity.clone()
    }

    pub fn pulse(&self) {
        self.activity.pulse();
    }

    pub fn mark_side_effect_risk(&self) {
        self.activity.mark_side_effect_risk();
    }

    pub fn mark_unresolved_external_work(&self) {
        self.activity.mark_unresolved_external_work();
    }

    /// Awaitable that resolves once the controller has been aborted.
    pub async fn cancelled(&self) {
        self.token.cancelled().await
    }

    /// Borrow the underlying [`CancellationToken`] — useful when interacting
    /// with libraries that take one directly (e.g., reqwest cancel via
    /// `tokio::select!`).
    pub fn token(&self) -> &CancellationToken {
        &self.token
    }
}

impl Default for AbortController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_not_aborted() {
        let a = AbortController::new();
        assert!(!a.is_aborted());
        assert!(a.reason().is_none());
    }

    #[test]
    fn abort_sets_aborted() {
        let a = AbortController::new();
        a.abort();
        assert!(a.is_aborted());
        assert!(a.reason().is_none());
    }

    #[test]
    fn abort_with_reason_captures() {
        let a = AbortController::new();
        a.abort_with_reason("user clicked cancel");
        assert!(a.is_aborted());
        assert_eq!(a.reason().as_deref(), Some("user clicked cancel"));
    }

    #[test]
    fn abort_with_reason_first_wins() {
        let a = AbortController::new();
        a.abort_with_reason("first");
        a.abort_with_reason("second"); // no-op for reason
        assert_eq!(a.reason().as_deref(), Some("first"));
    }

    #[test]
    fn child_cascades_from_parent() {
        let parent = AbortController::new();
        let child = parent.child();
        assert!(!child.is_aborted());
        parent.abort_with_reason("parent abort");
        assert!(child.is_aborted());
        // Child does not inherit parent's reason — own slot stays empty.
        assert!(child.reason().is_none());
        assert_eq!(parent.reason().as_deref(), Some("parent abort"));
    }

    #[test]
    fn child_abort_does_not_cascade_up() {
        let parent = AbortController::new();
        let child = parent.child();
        child.abort_with_reason("child abort");
        assert!(child.is_aborted());
        assert!(!parent.is_aborted());
        assert!(parent.reason().is_none());
        assert_eq!(child.reason().as_deref(), Some("child abort"));
    }

    #[test]
    fn clone_shares_state() {
        let a = AbortController::new();
        let b = a.clone();
        a.abort_with_reason("from a");
        assert!(b.is_aborted());
        // Same reason slot is shared because clone keeps the same Arc.
        assert_eq!(b.reason().as_deref(), Some("from a"));
    }

    #[test]
    fn child_shares_turn_activity_and_side_effect_latch() {
        let parent = AbortController::new();
        let child = parent.child();
        let before = parent.activity().last_activity_at();
        child.pulse();
        assert!(parent.activity().last_activity_at() >= before);
        child.mark_side_effect_risk();
        assert!(parent.activity().side_effect_risk());
    }

    #[tokio::test]
    async fn cancelled_future_resolves_after_abort() {
        let a = AbortController::new();
        let a2 = a.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            a2.abort();
        });
        // Should resolve quickly via the spawned task.
        a.cancelled().await;
        assert!(a.is_aborted());
    }

    #[tokio::test]
    async fn quiescence_waits_for_shared_child_workers() {
        let parent = AbortController::new();
        let child = parent.child();
        let work = child.activity().track_worker();
        assert_eq!(parent.activity().active_workers(), 1);
        assert!(tokio::time::timeout(
            std::time::Duration::from_millis(20),
            parent.activity().wait_for_quiescence(),
        )
        .await
        .is_err());

        drop(work);
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            parent.activity().wait_for_quiescence(),
        )
        .await
        .expect("worker drop should acknowledge quiescence");
        assert_eq!(parent.activity().active_workers(), 0);
    }
}
