//! Abort control — wraps [`tokio_util::sync::CancellationToken`] with an
//! optional reason string captured at abort time.
//!
//! Phase 1 / Task 1.4. Used by [`Provider::stream`](crate::provider::Provider)
//! and every long-running operation in later phases. Cheap to clone; child
//! controllers cascade-cancel from their parent but track their own reason.

use std::sync::{Arc, Mutex};

use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
pub struct AbortController {
    token: CancellationToken,
    reason: Arc<Mutex<Option<String>>>,
}

impl AbortController {
    /// Create a new root controller. Not aborted, no reason.
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            reason: Arc::new(Mutex::new(None)),
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
}
