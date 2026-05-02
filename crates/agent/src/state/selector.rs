//! Selector — subscribe to a slice of [`super::AppState`].
//!
//! Wraps [`super::AppStateStore::subscribe`] with a projection
//! function so a UI surface that only renders on (e.g.) mode changes
//! doesn't re-render every time the running-tool field flips.
//!
//! ```no_run
//! # use agent::state::{AppStateStore, AgentMode, Selector};
//! # async fn run(store: AppStateStore) {
//! let mut sel = Selector::new(&store, |s| s.mode);
//! while let Some(change) = sel.next().await {
//!     // change.prev / change.next are the AgentMode values.
//! }
//! # }
//! ```

use std::fmt;

use tokio::sync::broadcast::error::RecvError;

use super::store::{AppState, AppStateStore};

/// One observed change in a projected slice. `prev == next` is
/// possible only at first observation when the initial value is
/// taken from the snapshot — subsequent yields always have prev != next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorChange<T> {
    pub prev: T,
    pub next: T,
}

/// A live projection of one slice of [`AppState`]. Holds:
///
/// - A broadcast receiver from the store.
/// - The projection function (`AppState` → `T`).
/// - The most-recently-yielded value, used to dedup unchanged frames.
pub struct Selector<T, F>
where
    F: Fn(&AppState) -> T,
    T: PartialEq + Clone,
{
    rx: tokio::sync::broadcast::Receiver<AppState>,
    project: F,
    last: T,
}

impl<T, F> fmt::Debug for Selector<T, F>
where
    F: Fn(&AppState) -> T,
    T: PartialEq + Clone + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Selector")
            .field("last", &self.last)
            .finish()
    }
}

impl<T, F> Selector<T, F>
where
    F: Fn(&AppState) -> T,
    T: PartialEq + Clone,
{
    /// Subscribe to `store` and seed the last-value cache from the
    /// current snapshot.
    pub fn new(store: &AppStateStore, project: F) -> Self {
        let snap = store.snapshot();
        let initial = project(&snap);
        Self {
            rx: store.subscribe(),
            project,
            last: initial,
        }
    }

    /// The current cached value (read-only).
    pub fn current(&self) -> &T {
        &self.last
    }

    /// Await the next *change* in the projected value. Returns
    /// `None` when the underlying channel is closed (store dropped)
    /// or the receiver has lagged so far behind that recovery isn't
    /// meaningful — the caller should re-subscribe in that case.
    ///
    /// Yields a [`SelectorChange`] only when `project(state)` differs
    /// from the cached value; identical projections are filtered out
    /// silently.
    pub async fn next(&mut self) -> Option<SelectorChange<T>> {
        loop {
            match self.rx.recv().await {
                Ok(state) => {
                    let projected = (self.project)(&state);
                    if projected != self.last {
                        let prev = std::mem::replace(&mut self.last, projected.clone());
                        return Some(SelectorChange {
                            prev,
                            next: projected,
                        });
                    }
                    // No change — keep awaiting.
                }
                Err(RecvError::Lagged(_)) => {
                    // Slow consumer: caller should re-subscribe via
                    // a fresh Selector. Surface as None to terminate.
                    return None;
                }
                Err(RecvError::Closed) => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::store::{AgentMode, AppStateStore};
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn selector_yields_only_relevant_changes() {
        let store = AppStateStore::new();
        let mut sel = Selector::new(&store, |s| s.mode);

        // Updating a different field — must NOT yield.
        store.update(|s| s.with_session_id("S1"));
        // Schedule a real mode change in the background.
        let s = store.clone();
        let h = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            s.update(|s| s.with_mode(AgentMode::Bypass));
        });
        let change = sel.next().await.unwrap();
        assert_eq!(change.prev, AgentMode::Default);
        assert_eq!(change.next, AgentMode::Bypass);
        h.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn selector_current_seeds_from_snapshot() {
        let store = AppStateStore::new();
        store.update(|s| s.with_mode(AgentMode::Plan));
        let sel = Selector::new(&store, |s| s.mode);
        assert_eq!(*sel.current(), AgentMode::Plan);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn selector_dedups_repeated_writes() {
        let store = AppStateStore::new();
        let mut sel = Selector::new(&store, |s| s.mode);
        // Three writes that all keep mode the same → no yields.
        store.update(|s| s.with_session_id("a"));
        store.update(|s| s.with_session_id("b"));
        store.update(|s| s.with_session_id("c"));
        // Then a real change.
        let s = store.clone();
        let h = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            s.update(|s| s.with_mode(AgentMode::AcceptEdits));
        });
        let change = sel.next().await.unwrap();
        assert_eq!(change.next, AgentMode::AcceptEdits);
        h.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn selector_returns_none_when_store_is_dropped() {
        let store = AppStateStore::new();
        let mut sel = Selector::new(&store, |s| s.mode);
        drop(store);
        assert!(sel.next().await.is_none());
    }
}
