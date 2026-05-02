//! `AppStateStore` — typed transient runtime state, snapshot + broadcast.
//!
//! Read pattern:
//!
//! ```no_run
//! # use agent::state::{AppStateStore, AgentMode};
//! # let store = AppStateStore::new();
//! let snap = store.snapshot();
//! println!("session = {:?}", snap.session_id);
//! ```
//!
//! Write pattern (replace the entire snapshot, broadcast):
//!
//! ```no_run
//! # use agent::state::{AppStateStore, AgentMode};
//! # let store = AppStateStore::new();
//! store.update(|s| s.with_mode(AgentMode::Bypass));
//! ```
//!
//! Subscribe pattern (full snapshot stream):
//!
//! ```no_run
//! # use agent::state::AppStateStore;
//! # async fn run(store: AppStateStore) {
//! let mut rx = store.subscribe();
//! while let Ok(snap) = rx.recv().await {
//!     // re-render with snap
//! }
//! # }
//! ```

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Permission/agent mode — mirrors the four-mode taxonomy used by the
/// permission manager. Re-declared here so the state surface doesn't
/// pull the permission module into its public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentMode {
    /// Default — every tool that doesn't have an explicit allow rule
    /// triggers a permission prompt.
    #[default]
    Default,
    /// Edits are auto-accepted; other tools follow default rules.
    AcceptEdits,
    /// All tools auto-allowed (typically for trusted CI/test runs).
    Bypass,
    /// Plan mode — read-only tools allowed, write tools blocked.
    Plan,
}

/// Snapshot of the model + effort + output configuration the host
/// is currently running with.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    pub model: String,
    /// Effort level as a stable string (mirrors `api::EffortLevel`).
    #[serde(default)]
    pub effort: Option<String>,
    /// Output config kind as a stable string (mirrors
    /// `api::OutputMode`'s tag).
    #[serde(default)]
    pub output_mode: Option<String>,
}

/// One user message awaiting send (e.g., user typed multiple lines
/// faster than the agent could process). Surfaced in the UI as a
/// pending-input indicator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedMessage {
    pub id: String,
    pub text: String,
    pub queued_at_unix_ms: u64,
}

/// The full transient state snapshot. Cheap to clone (most fields
/// are short strings or small enums). Host keeps an `Arc<AppState>`
/// in render-side data; updaters produce a new `AppState` and the
/// store broadcasts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppState {
    pub session_id: Option<String>,
    pub working_dir: Option<PathBuf>,
    pub mode: AgentMode,
    pub config: ConfigSnapshot,
    /// Name of the tool currently executing (if any). Cleared once
    /// the tool returns or fails.
    pub running_tool: Option<String>,
    /// Identifier of the in-flight assistant turn, when applicable.
    pub running_turn_id: Option<String>,
    /// Pending user messages — head-of-queue is the next to send.
    #[serde(default)]
    pub queued_messages: Vec<QueuedMessage>,
    /// Last error message surfaced to the user (banner). Cleared by
    /// host when the user dismisses.
    pub last_error: Option<String>,
}

impl AppState {
    /// Builder-pattern helpers — return a copy with the field set,
    /// keeping the read/write split simple. Hosts typically chain
    /// these inside [`AppStateStore::update`].
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }
    pub fn clear_session(mut self) -> Self {
        self.session_id = None;
        self
    }
    pub fn with_working_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(path.into());
        self
    }
    pub fn with_mode(mut self, mode: AgentMode) -> Self {
        self.mode = mode;
        self
    }
    pub fn with_config(mut self, c: ConfigSnapshot) -> Self {
        self.config = c;
        self
    }
    pub fn with_running_tool(mut self, name: impl Into<String>) -> Self {
        self.running_tool = Some(name.into());
        self
    }
    pub fn clear_running_tool(mut self) -> Self {
        self.running_tool = None;
        self
    }
    pub fn with_running_turn(mut self, id: impl Into<String>) -> Self {
        self.running_turn_id = Some(id.into());
        self
    }
    pub fn clear_running_turn(mut self) -> Self {
        self.running_turn_id = None;
        self
    }
    pub fn enqueue_message(mut self, msg: QueuedMessage) -> Self {
        self.queued_messages.push(msg);
        self
    }
    /// Pop the head of the queue. Returns the popped message + the
    /// new state.
    pub fn pop_queued(mut self) -> (Option<QueuedMessage>, Self) {
        let popped = if self.queued_messages.is_empty() {
            None
        } else {
            Some(self.queued_messages.remove(0))
        };
        (popped, self)
    }
    pub fn with_last_error(mut self, msg: impl Into<String>) -> Self {
        self.last_error = Some(msg.into());
        self
    }
    pub fn clear_last_error(mut self) -> Self {
        self.last_error = None;
        self
    }
}

/// Thread-safe state container with broadcast-based subscriptions.
#[derive(Debug, Clone)]
pub struct AppStateStore {
    inner: Arc<RwLock<AppState>>,
    tx: broadcast::Sender<AppState>,
}

impl AppStateStore {
    /// Default-initialised store (capacity 64 for the broadcast
    /// channel — enough for typical UI fan-outs of 1–8 listeners
    /// with some headroom for slow consumers).
    pub fn new() -> Self {
        Self::with_initial(AppState::default())
    }

    pub fn with_initial(initial: AppState) -> Self {
        let (tx, _rx) = broadcast::channel(64);
        Self {
            inner: Arc::new(RwLock::new(initial)),
            tx,
        }
    }

    /// Read the current snapshot. Cheap clone of the inner state.
    pub fn snapshot(&self) -> AppState {
        self.inner
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|p| p.into_inner().clone())
    }

    /// Apply an updater closure. The closure receives a mutable
    /// owned `AppState` and returns the new one. The store stores
    /// the new value AND broadcasts it. Returns the new snapshot.
    pub fn update<F>(&self, f: F) -> AppState
    where
        F: FnOnce(AppState) -> AppState,
    {
        // Inline the read+write+broadcast under a single critical
        // section so subscribers see updates in monotonic order.
        let new = {
            let mut guard = self.inner.write().unwrap_or_else(|p| p.into_inner());
            let prev = guard.clone();
            let next = f(prev);
            *guard = next.clone();
            next
        };
        // SendErr just means there are zero current subscribers;
        // ignore — the snapshot is still authoritative.
        let _ = self.tx.send(new.clone());
        new
    }

    /// Subscribe to full-snapshot updates. Each `Ok(state)` yielded
    /// is the post-update state from a single `update()` call.
    /// `RecvError::Lagged` is returned if a slow subscriber falls
    /// behind the channel's 64-deep buffer; callers should resync
    /// from [`Self::snapshot`] in that case.
    pub fn subscribe(&self) -> broadcast::Receiver<AppState> {
        self.tx.subscribe()
    }

    /// Number of currently-active subscribers. Useful for telemetry.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Replace the entire snapshot. Same broadcast semantics as
    /// [`Self::update`]. Use when the new state is independent of
    /// the prior (e.g., loading a saved session).
    pub fn replace(&self, next: AppState) -> AppState {
        self.update(|_| next)
    }
}

impl Default for AppStateStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_returns_default_initially() {
        let store = AppStateStore::new();
        let snap = store.snapshot();
        assert_eq!(snap.mode, AgentMode::Default);
        assert!(snap.session_id.is_none());
        assert!(snap.queued_messages.is_empty());
    }

    #[test]
    fn update_applies_closure_and_returns_new() {
        let store = AppStateStore::new();
        let new = store.update(|s| s.with_session_id("S1").with_mode(AgentMode::Bypass));
        assert_eq!(new.session_id.as_deref(), Some("S1"));
        assert_eq!(new.mode, AgentMode::Bypass);
        // Snapshot reflects the update.
        assert_eq!(store.snapshot(), new);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn subscribe_receives_updates() {
        let store = AppStateStore::new();
        let mut rx = store.subscribe();
        store.update(|s| s.with_mode(AgentMode::Plan));
        let received = rx.recv().await.unwrap();
        assert_eq!(received.mode, AgentMode::Plan);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn multiple_subscribers_each_see_update() {
        let store = AppStateStore::new();
        let mut a = store.subscribe();
        let mut b = store.subscribe();
        store.update(|s| s.with_mode(AgentMode::AcceptEdits));
        assert_eq!(a.recv().await.unwrap().mode, AgentMode::AcceptEdits);
        assert_eq!(b.recv().await.unwrap().mode, AgentMode::AcceptEdits);
    }

    #[test]
    fn enqueue_and_pop_queue() {
        let store = AppStateStore::new();
        let msg = QueuedMessage {
            id: "m1".into(),
            text: "hello".into(),
            queued_at_unix_ms: 0,
        };
        let new = store.update(|s| s.enqueue_message(msg.clone()));
        assert_eq!(new.queued_messages.len(), 1);
        let popped_state = store.update(|s| {
            let (popped, s2) = s.pop_queued();
            assert_eq!(popped.unwrap().id, "m1");
            s2
        });
        assert!(popped_state.queued_messages.is_empty());
    }

    #[test]
    fn replace_overrides_state_entirely() {
        let store = AppStateStore::new();
        store.update(|s| s.with_session_id("old"));
        let fresh = AppState::default().with_session_id("new");
        let after = store.replace(fresh);
        assert_eq!(after.session_id.as_deref(), Some("new"));
    }

    #[test]
    fn subscriber_count_tracks_handles() {
        let store = AppStateStore::new();
        assert_eq!(store.subscriber_count(), 0);
        let _a = store.subscribe();
        assert_eq!(store.subscriber_count(), 1);
        let _b = store.subscribe();
        assert_eq!(store.subscriber_count(), 2);
        drop(_a);
        assert_eq!(store.subscriber_count(), 1);
    }

    #[test]
    fn appstate_serialization_roundtrip() {
        let s = AppState::default()
            .with_session_id("S1")
            .with_mode(AgentMode::Plan)
            .with_running_tool("read_file");
        let json = serde_json::to_string(&s).unwrap();
        let parsed: AppState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn updates_are_ordered() {
        let store = AppStateStore::new();
        let mut rx = store.subscribe();
        for i in 0..10 {
            store.update(|s| s.with_session_id(format!("S{i}")));
        }
        let mut seen: Vec<String> = Vec::new();
        for _ in 0..10 {
            let snap = rx.recv().await.unwrap();
            seen.push(snap.session_id.unwrap());
        }
        let expected: Vec<String> = (0..10).map(|i| format!("S{i}")).collect();
        assert_eq!(seen, expected);
    }

    #[test]
    fn appstate_chained_builders_are_independent() {
        let s = AppState::default();
        let a = s.clone().with_session_id("A");
        let b = s.with_session_id("B");
        assert_eq!(a.session_id.as_deref(), Some("A"));
        assert_eq!(b.session_id.as_deref(), Some("B"));
    }

    #[test]
    fn clear_helpers_remove_fields() {
        let store = AppStateStore::new();
        store.update(|s| {
            s.with_session_id("S")
                .with_running_tool("rt")
                .with_last_error("oops")
        });
        let cleared = store.update(|s| s.clear_session().clear_running_tool().clear_last_error());
        assert!(cleared.session_id.is_none());
        assert!(cleared.running_tool.is_none());
        assert!(cleared.last_error.is_none());
    }
}
