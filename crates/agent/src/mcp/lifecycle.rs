//! MCP server lifecycle (Tier 1 / claude-code parity).
//!
//! Mirrors `services/mcp/useManageMCPConnections.ts` (the React hook
//! version) — without the React layer. Drives the connect →
//! discover → maintain → disconnect lifecycle for every server in
//! the registry. Pluggable [`Connector`] trait so tests don't need
//! a real subprocess + the production code can wire `rmcp` into it.
//!
//! ## Flow
//!
//! 1. [`Lifecycle::connect`] expands env in the server config, marks
//!    the registry entry `Connecting`, then awaits
//!    `connector.connect()`.
//! 2. On success it asks the [`Connection`] for `tool_names()` and
//!    `resource_uris()`, writes them into the registry, marks
//!    `Connected`.
//! 3. On failure it transitions the registry to
//!    [`super::registry::ServerState::Failed`] (preserving the
//!    accumulated `attempt_count` across retries via direct
//!    [`super::registry::McpRegistry::set_state`] rather than
//!    `mark_failed`, which would reset the counter on each call).
//!    Wired to the auto-retry policy of [`crate::api::retry`] when
//!    the host chooses to reconnect.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex as AsyncMutex;

use super::config::McpServerConfig;
use super::registry::{McpRegistry, RegisteredServer, ServerState};

/// Errors surfaced by the lifecycle manager.
#[derive(Debug, thiserror::Error)]
pub enum LifecycleError {
    #[error("server `{0}` not registered")]
    UnknownServer(String),
    #[error("server `{0}` is disabled")]
    Disabled(String),
    #[error("connector error: {0}")]
    Connector(String),
}

/// Per-server connection handle. The lifecycle manager uses these
/// for discovery + tool dispatch; the underlying transport is the
/// connector's responsibility.
#[async_trait]
pub trait Connection: std::fmt::Debug + Send + Sync {
    /// Tool names advertised by the server (post-handshake).
    fn tool_names(&self) -> Vec<String>;
    /// Resource URIs advertised by the server (post-handshake).
    fn resource_uris(&self) -> Vec<String>;
    /// Invoke a tool. Wire-shape mirrors the agent's [`crate::tool`]
    /// trait so the dispatcher can treat MCP-tools uniformly.
    async fn call_tool(
        &self,
        name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, LifecycleError>;
    /// Cleanly close the connection. Called on graceful shutdown.
    async fn close(&self) -> Result<(), LifecycleError>;
}

/// Pluggable factory: given a server name + config, produce a
/// [`Connection`] (or fail). Implementations wrap rmcp transports
/// (stdio child process, SSE, WebSocket) — mocked in tests.
#[async_trait]
pub trait Connector: std::fmt::Debug + Send + Sync {
    async fn connect(
        &self,
        name: &str,
        config: &McpServerConfig,
    ) -> Result<Arc<dyn Connection>, LifecycleError>;
}

/// In-memory mapping from server name → live [`Connection`].
type ConnectionMap = std::sync::Mutex<BTreeMap<String, Arc<dyn Connection>>>;

/// Per-server async lock guarding the connect path. Concurrent
/// `connect("srv")` calls serialize on the same lock so we don't
/// fire two transports for one server.
type ConnectLockMap = std::sync::Mutex<BTreeMap<String, Arc<AsyncMutex<()>>>>;

/// Top-level lifecycle manager. Owns the registry, connections, and
/// the connector factory.
#[derive(Debug, Clone)]
pub struct Lifecycle {
    pub registry: McpRegistry,
    pub connector: Arc<dyn Connector>,
    connections: Arc<ConnectionMap>,
    connect_locks: Arc<ConnectLockMap>,
    /// Process env captured at construction. Used to expand
    /// `$VAR` references in configs before connecting.
    pub env: BTreeMap<String, String>,
}

impl Lifecycle {
    pub fn new(registry: McpRegistry, connector: Arc<dyn Connector>) -> Self {
        Self {
            registry,
            connector,
            connections: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            connect_locks: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            env: std::env::vars().collect(),
        }
    }

    /// Get (or create) the per-server connect-lock. Concurrent
    /// callers race exactly once to insert the lock; subsequent
    /// callers acquire the same `Arc<AsyncMutex>`.
    fn lock_for(&self, name: &str) -> Arc<AsyncMutex<()>> {
        let mut guard = self.connect_locks.lock().unwrap_or_else(|p| p.into_inner());
        guard
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    pub fn with_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Add or replace a server config. If the named server already
    /// has a live connection, the connection is closed first so the
    /// registry and connection-map stay coherent.
    ///
    /// Hosts that mutate `self.registry` directly via
    /// [`McpRegistry::upsert`] bypass this coherence check and can
    /// leave stale connections behind. Prefer this method.
    ///
    /// Returns `Ok(())` on success. Returns `Err(LifecycleError::Connector(...))`
    /// if the stale connection's `close()` failed — the registry is
    /// still updated to the new config, but the caller can see the
    /// cleanup failure (transports needing explicit close may leak
    /// otherwise).
    pub async fn upsert_server(
        &self,
        name: impl Into<String>,
        config: McpServerConfig,
    ) -> Result<(), LifecycleError> {
        let name = name.into();
        let lock = self.lock_for(&name);
        let _guard = lock.lock().await;
        let stale = match self.connections.lock() {
            Ok(mut m) => m.remove(&name),
            Err(e) => {
                return Err(LifecycleError::Connector(format!(
                    "connection map poisoned: {e}"
                )));
            }
        };
        let close_result = match stale {
            Some(c) => c.close().await,
            None => Ok(()),
        };
        self.registry.upsert(name, config);
        close_result
    }

    /// Remove a server. Closes any live connection and drops the
    /// registry entry outright (no intermediate Disconnected state —
    /// callers that want a state-transition trail should call
    /// [`Self::disconnect`] before remove). Returns `true` if the
    /// server existed.
    ///
    /// Hosts should prefer this over [`McpRegistry::remove`], which
    /// drops the registry entry without closing the live transport
    /// and leaves a "ghost" connection that
    /// [`Self::call_tool`] could keep using.
    ///
    /// **Note**: the per-server async lock entry in
    /// [`Self::connect_locks`] is intentionally NOT dropped on
    /// remove. Tasks queued on the old lock would otherwise resume
    /// after `remove_server` returns and run in parallel with new
    /// operations that allocated a fresh lock for the re-added
    /// server name — defeating the single-critical-section
    /// invariant. The trade-off is bounded growth: lock entries
    /// stay around forever for any name that's ever been registered.
    /// In practice, MCP server lists are tiny (typically <20
    /// entries) so this is non-issue.
    pub async fn remove_server(&self, name: &str) -> Result<bool, LifecycleError> {
        if self.registry.get(name).is_none() {
            return Ok(false);
        }
        let lock = self.lock_for(name);
        let _guard = lock.lock().await;
        let stale = match self.connections.lock() {
            Ok(mut m) => m.remove(name),
            Err(e) => {
                return Err(LifecycleError::Connector(format!(
                    "connection map poisoned: {e}"
                )));
            }
        };
        let close_result = match stale {
            Some(c) => c.close().await,
            None => Ok(()),
        };
        let existed = self.registry.remove(name);
        close_result.map(|()| existed)
    }

    /// Connect a single server. Idempotent — calling on an already-
    /// connected server with a live handle returns Ok without
    /// reconnecting. If the registry believes a server is connected
    /// but the live handle has been dropped, this reconnects.
    ///
    /// Concurrent calls with the same `name` serialize on a per-
    /// server async lock, so two callers can't race two transports.
    pub async fn connect(&self, name: &str) -> Result<(), LifecycleError> {
        // Pre-flight: validate the server exists BEFORE allocating a
        // per-name lock. A spam of `connect("ghost")` calls would
        // otherwise grow `connect_locks` unboundedly.
        if self.registry.get(name).is_none() {
            return Err(LifecycleError::UnknownServer(name.into()));
        }

        let lock = self.lock_for(name);
        let _guard = lock.lock().await;

        // Re-read AFTER acquiring the lock — state may have changed
        // between pre-flight and lock acquisition (e.g., another
        // task removed/disabled the server while we were queued).
        let entry: RegisteredServer = self
            .registry
            .get(name)
            .ok_or_else(|| LifecycleError::UnknownServer(name.into()))?;
        if !entry.config.enabled() {
            return Err(LifecycleError::Disabled(name.into()));
        }
        if entry.state.is_connected() && self.has_live_connection(name) {
            return Ok(());
        }

        // Preserve the prior failure count so retry accounting
        // survives across the Connecting transition. Without this,
        // mark_failed reading `Connecting` state would reset the
        // counter to 1.
        let prior_attempts = match &entry.state {
            ServerState::Failed { attempt_count, .. } => *attempt_count,
            _ => 0,
        };

        let mut config = entry.config.clone();
        config.expand_env(&self.env);

        self.registry.mark_connecting(name);
        match self.connector.connect(name, &config).await {
            Ok(conn) => {
                let tools = conn.tool_names();
                let resources = conn.resource_uris();
                // Replace any stale handle (and close it) before
                // installing the fresh connection. If the lock is
                // poisoned, transition to Failed via set_state
                // (preserving the accumulated attempt_count) so the
                // registry never reports Connected without a live
                // handle.
                let stale = match self.connections.lock() {
                    Ok(mut map) => map.insert(name.to_string(), conn),
                    Err(e) => {
                        let next_attempts = prior_attempts.saturating_add(1);
                        self.registry.set_state(
                            name,
                            ServerState::Failed {
                                message: format!("connection map poisoned: {e}"),
                                attempt_count: next_attempts,
                                last_attempt_unix_ms: now_unix_ms(),
                            },
                        );
                        return Err(LifecycleError::Connector(format!(
                            "connection map poisoned: {e}"
                        )));
                    }
                };
                let close_result = match stale {
                    Some(old) => old.close().await,
                    None => Ok(()),
                };
                self.registry.mark_connected(name, tools, resources);
                // Surface stale-handle close failures the same way
                // upsert_server / remove_server do — caller can see
                // the cleanup failure even though the new connection
                // is live.
                close_result
            }
            Err(e) => {
                // Connector failed. If a stale handle was sitting in
                // the connections map (registry-says-not-Connected but
                // map-still-has-entry), drop it now — otherwise
                // call_tool would happily dispatch through the dead
                // handle and the registry-Failed state would be a lie.
                let stale = self
                    .connections
                    .lock()
                    .ok()
                    .and_then(|mut m| m.remove(name));
                if let Some(c) = stale {
                    let _ = c.close().await;
                }
                let next_attempts = prior_attempts.saturating_add(1);
                self.registry.set_state(
                    name,
                    ServerState::Failed {
                        message: e.to_string(),
                        attempt_count: next_attempts,
                        last_attempt_unix_ms: now_unix_ms(),
                    },
                );
                Err(e)
            }
        }
    }

    fn has_live_connection(&self, name: &str) -> bool {
        self.connections
            .lock()
            .map(|m| m.contains_key(name))
            .unwrap_or(false)
    }

    /// Attempt to connect every enabled server. Returns the per-
    /// server result so the caller can decide what to do about
    /// partial failures.
    pub async fn connect_all(&self) -> Vec<(String, Result<(), LifecycleError>)> {
        let names: Vec<String> = self
            .registry
            .snapshot()
            .into_iter()
            .filter(|e| e.config.enabled())
            .map(|e| e.name)
            .collect();
        let mut out = Vec::with_capacity(names.len());
        for n in names {
            let r = self.connect(&n).await;
            out.push((n, r));
        }
        out
    }

    /// Disconnect a single server. Returns
    /// [`LifecycleError::UnknownServer`] if the registry has no entry
    /// for `name` (mirrors `connect` and `call_tool`).
    /// Idempotent for registered non-disabled servers: a server with
    /// no live handle returns Ok and transitions the registry to
    /// [`ServerState::Disconnected`]. Disabled servers return
    /// [`LifecycleError::Disabled`] and the disabled state is
    /// preserved.
    ///
    /// Acquires the same per-server async lock as [`Self::connect`],
    /// so a `disconnect` racing an in-flight connect will wait for
    /// connect to settle before tearing the connection down. Without
    /// this, connect could land its newly-built connection AFTER
    /// disconnect cleared the map, leaving the registry permanently
    /// `Connected` with no live handle.
    pub async fn disconnect(&self, name: &str, reason: &str) -> Result<(), LifecycleError> {
        if self.registry.get(name).is_none() {
            return Err(LifecycleError::UnknownServer(name.into()));
        }
        let lock = self.lock_for(name);
        let _guard = lock.lock().await;
        // Re-read AFTER acquiring the lock — pre-flight state is
        // stale once we've waited for the lock. If the server was
        // disabled while we waited, leave its `Disabled` state alone
        // rather than overwriting with `Disconnected`.
        match self.registry.get(name).map(|e| e.state) {
            None => return Err(LifecycleError::UnknownServer(name.into())),
            Some(ServerState::Disabled) => return Err(LifecycleError::Disabled(name.into())),
            _ => {}
        }
        let conn = {
            let mut map = self
                .connections
                .lock()
                .map_err(|e| LifecycleError::Connector(format!("connection map poisoned: {e}")))?;
            map.remove(name)
        };
        // ALWAYS mark the registry disconnected, even if the inner
        // close call fails — the live handle has been removed from
        // the map already, so leaving the registry as Connected
        // would create a state where call_tool returns
        // UnknownServer while the registry claims success.
        let close_result = match conn {
            Some(c) => c.close().await,
            None => Ok(()),
        };
        self.registry.mark_disconnected(name, reason);
        close_result
    }

    /// Disconnect every connected server. Best-effort — errors are
    /// returned alongside but do not stop the loop. A poisoned
    /// connection-map mutex is recovered (we read the inner state
    /// rather than panicking), keeping disconnect_all consistent
    /// with `disconnect` / `call_tool` poison handling.
    pub async fn disconnect_all(&self, reason: &str) -> Vec<(String, Result<(), LifecycleError>)> {
        let names: Vec<String> = {
            let map = self.connections.lock().unwrap_or_else(|e| e.into_inner());
            map.keys().cloned().collect()
        };
        let mut out = Vec::with_capacity(names.len());
        for n in names {
            let r = self.disconnect(&n, reason).await;
            out.push((n, r));
        }
        out
    }

    /// Direct call into a connected server's tool. Returns
    /// `UnknownServer` if not connected, otherwise the connection's
    /// own error.
    pub async fn call_tool(
        &self,
        server: &str,
        tool: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, LifecycleError> {
        let conn = {
            let map = self
                .connections
                .lock()
                .map_err(|e| LifecycleError::Connector(format!("connection map poisoned: {e}")))?;
            map.get(server).cloned()
        };
        match conn {
            Some(c) => c.call_tool(tool, input).await,
            None => Err(LifecycleError::UnknownServer(server.into())),
        }
    }

    /// Number of currently-connected servers.
    pub fn live_count(&self) -> usize {
        self.connections.lock().map(|m| m.len()).unwrap_or(0)
    }
}

/// UNIX-epoch milliseconds. Returns 0 if the system clock is before
/// epoch (test environments occasionally do this).
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::registry::McpRegistry;
    use super::*;

    /// Test connector — succeeds always; advertises a fixed tool set.
    #[derive(Debug)]
    struct FakeConnector {
        tools: Vec<String>,
        connect_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Connector for FakeConnector {
        async fn connect(
            &self,
            _name: &str,
            _config: &McpServerConfig,
        ) -> Result<Arc<dyn Connection>, LifecycleError> {
            self.connect_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakeConnection {
                tools: self.tools.clone(),
            }))
        }
    }

    #[derive(Debug)]
    struct FakeConnection {
        tools: Vec<String>,
    }

    #[async_trait]
    impl Connection for FakeConnection {
        fn tool_names(&self) -> Vec<String> {
            self.tools.clone()
        }
        fn resource_uris(&self) -> Vec<String> {
            vec![]
        }
        async fn call_tool(
            &self,
            name: &str,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, LifecycleError> {
            Ok(serde_json::json!({ "echoed": name, "input": input }))
        }
        async fn close(&self) -> Result<(), LifecycleError> {
            Ok(())
        }
    }

    /// Always-failing connector — for failure-path tests.
    #[derive(Debug)]
    struct FailingConnector;
    #[async_trait]
    impl Connector for FailingConnector {
        async fn connect(
            &self,
            _name: &str,
            _config: &McpServerConfig,
        ) -> Result<Arc<dyn Connection>, LifecycleError> {
            Err(LifecycleError::Connector("nope".into()))
        }
    }

    /// Connection whose `close()` always errors. Lets tests verify
    /// stale-handle close failures propagate to the caller.
    #[derive(Debug)]
    struct FailingCloseConnection;
    #[async_trait]
    impl Connection for FailingCloseConnection {
        fn tool_names(&self) -> Vec<String> {
            vec![]
        }
        fn resource_uris(&self) -> Vec<String> {
            vec![]
        }
        async fn call_tool(
            &self,
            _name: &str,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, LifecycleError> {
            Ok(serde_json::Value::Null)
        }
        async fn close(&self) -> Result<(), LifecycleError> {
            Err(LifecycleError::Connector("close failed".into()))
        }
    }

    /// Connector that returns a [`FailingCloseConnection`] handle
    /// only on the first connect; subsequent calls return a
    /// well-behaved fake. Lets us test the connect-reconnect flow.
    #[derive(Debug)]
    struct FailingCloseThenFakeConnector {
        first_done: Arc<std::sync::Mutex<bool>>,
    }
    #[async_trait]
    impl Connector for FailingCloseThenFakeConnector {
        async fn connect(
            &self,
            _name: &str,
            _config: &McpServerConfig,
        ) -> Result<Arc<dyn Connection>, LifecycleError> {
            let mut first = self.first_done.lock().unwrap();
            if !*first {
                *first = true;
                Ok(Arc::new(FailingCloseConnection))
            } else {
                Ok(Arc::new(FakeConnection { tools: vec![] }))
            }
        }
    }

    fn stdio(cmd: &str, enabled: bool) -> McpServerConfig {
        McpServerConfig::Stdio {
            command: cmd.into(),
            args: vec![],
            env: Default::default(),
            cwd: None,
            enabled,
        }
    }

    #[tokio::test]
    async fn connect_marks_registry_connected_and_records_tools() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let calls = Arc::new(AtomicUsize::new(0));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FakeConnector {
                tools: vec!["fetch".into(), "ping".into()],
                connect_calls: calls.clone(),
            }),
        );
        lc.connect("srv").await.unwrap();
        let entry = reg.get("srv").unwrap();
        assert!(entry.state.is_connected());
        assert_eq!(entry.state.tool_names(), &["fetch", "ping"]);
        assert_eq!(lc.live_count(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn connect_idempotent_on_already_connected() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let calls = Arc::new(AtomicUsize::new(0));
        let lc = Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: calls.clone(),
            }),
        );
        lc.connect("srv").await.unwrap();
        lc.connect("srv").await.unwrap();
        // Second connect must NOT re-fire the connector.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn connect_disabled_returns_disabled_error() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", false));
        let lc = Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        assert!(matches!(
            lc.connect("srv").await.unwrap_err(),
            LifecycleError::Disabled(_)
        ));
    }

    #[tokio::test]
    async fn connect_unknown_server_errors() {
        let lc = Lifecycle::new(
            McpRegistry::new(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        assert!(matches!(
            lc.connect("ghost").await.unwrap_err(),
            LifecycleError::UnknownServer(_)
        ));
    }

    #[tokio::test]
    async fn connector_failure_marks_registry_failed() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(reg.clone(), Arc::new(FailingConnector));
        let err = lc.connect("srv").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(_)));
        let entry = reg.get("srv").unwrap();
        assert!(matches!(
            entry.state,
            super::super::registry::ServerState::Failed { .. }
        ));
    }

    #[tokio::test]
    async fn connect_all_skips_disabled() {
        let reg = McpRegistry::new();
        reg.upsert("a", stdio("a", true));
        reg.upsert("b", stdio("b", false));
        let calls = Arc::new(AtomicUsize::new(0));
        let lc = Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: calls.clone(),
            }),
        );
        let results = lc.connect_all().await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn disconnect_clears_connection_map() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        lc.connect("srv").await.unwrap();
        assert_eq!(lc.live_count(), 1);
        lc.disconnect("srv", "test").await.unwrap();
        assert_eq!(lc.live_count(), 0);
        let state = reg.get("srv").unwrap().state;
        assert!(matches!(
            state,
            super::super::registry::ServerState::Disconnected { .. }
        ));
    }

    #[tokio::test]
    async fn call_tool_routes_to_connection() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec!["fetch".into()],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        lc.connect("srv").await.unwrap();
        let r = lc
            .call_tool("srv", "fetch", serde_json::json!({"url": "x"}))
            .await
            .unwrap();
        assert_eq!(r["echoed"], "fetch");
        assert_eq!(r["input"]["url"], "x");
    }

    #[tokio::test]
    async fn call_tool_unknown_server_errors() {
        let lc = Lifecycle::new(
            McpRegistry::new(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        assert!(matches!(
            lc.call_tool("ghost", "x", serde_json::json!({}))
                .await
                .unwrap_err(),
            LifecycleError::UnknownServer(_)
        ));
    }

    #[tokio::test]
    async fn disconnect_unknown_server_errors() {
        let lc = Lifecycle::new(
            McpRegistry::new(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        assert!(matches!(
            lc.disconnect("ghost", "test").await.unwrap_err(),
            LifecycleError::UnknownServer(_)
        ));
    }

    #[tokio::test]
    async fn idempotent_connect_with_dropped_handle_reconnects() {
        // Simulate the registry-says-connected-but-no-live-handle case
        // by manually mutating state after a fresh connect.
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let calls = Arc::new(AtomicUsize::new(0));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: calls.clone(),
            }),
        );
        lc.connect("srv").await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Drop the live connection out from under the lifecycle to
        // simulate a transport hard-close.
        if let Ok(mut map) = lc.connections.lock() {
            map.remove("srv");
        }
        // Registry still says Connected. Calling connect should
        // detect the missing handle and reconnect.
        lc.connect("srv").await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn failed_attempt_count_preserves_across_retries() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(reg.clone(), Arc::new(FailingConnector));
        let _ = lc.connect("srv").await;
        let _ = lc.connect("srv").await;
        let _ = lc.connect("srv").await;
        match reg.get("srv").unwrap().state {
            super::super::registry::ServerState::Failed { attempt_count, .. } => {
                assert_eq!(attempt_count, 3, "attempts must accumulate, not reset");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn concurrent_connect_does_not_double_fire() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let calls = Arc::new(AtomicUsize::new(0));
        let lc = Arc::new(Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: calls.clone(),
            }),
        ));
        let lc1 = lc.clone();
        let lc2 = lc.clone();
        let h1 = tokio::spawn(async move { lc1.connect("srv").await });
        let h2 = tokio::spawn(async move { lc2.connect("srv").await });
        h1.await.unwrap().unwrap();
        h2.await.unwrap().unwrap();
        // Per-server lock should have made the second caller see the
        // first call's connected state and short-circuit.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn upsert_server_surfaces_close_failure() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FailingCloseThenFakeConnector {
                first_done: Arc::new(std::sync::Mutex::new(false)),
            }),
        );
        lc.connect("srv").await.unwrap();
        assert_eq!(lc.live_count(), 1);
        // Upsert with a *different* config so we can verify post-error
        // state. close() fails but the registry must still be updated
        // and the stale handle removed (post-error contract).
        let err = lc
            .upsert_server("srv", stdio("echo", false))
            .await
            .unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("close failed")));
        assert_eq!(lc.live_count(), 0);
        assert!(matches!(
            reg.get("srv").unwrap().state,
            super::super::registry::ServerState::Disabled
        ));
    }

    #[tokio::test]
    async fn remove_server_surfaces_close_failure() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FailingCloseThenFakeConnector {
                first_done: Arc::new(std::sync::Mutex::new(false)),
            }),
        );
        lc.connect("srv").await.unwrap();
        let err = lc.remove_server("srv").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("close failed")));
        // Post-error contract: stale handle removed AND registry
        // entry dropped, even though close() failed.
        assert_eq!(lc.live_count(), 0);
        assert!(reg.get("srv").is_none());
    }

    #[tokio::test]
    async fn reconnect_surfaces_stale_close_failure() {
        // Setup: first connect installs FailingCloseConnection. We
        // then transition the registry state to Idle WITHOUT touching
        // the connections map — the live (failing-close) handle is
        // still present while the registry believes the server isn't
        // connected. That registry/map mismatch is the inconsistency
        // connect's reconnect path is meant to repair.
        //
        // The next connect must:
        //   (a) detect registry != Connected and proceed with a
        //       fresh connect attempt;
        //   (b) install the new (FakeConnection) handle, replacing
        //       the stale FailingCloseConnection;
        //   (c) call close() on the stale handle and surface its
        //       error to the caller — the new handle stays live.
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg,
            Arc::new(FailingCloseThenFakeConnector {
                first_done: Arc::new(std::sync::Mutex::new(false)),
            }),
        );
        lc.connect("srv").await.unwrap();
        lc.registry
            .set_state("srv", super::super::registry::ServerState::Idle);
        let err = lc.connect("srv").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("close failed")));
        assert_eq!(lc.live_count(), 1);
    }

    #[tokio::test]
    async fn upsert_server_returns_error_on_poisoned_mutex() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Arc::new(Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        ));
        lc.connect("srv").await.unwrap();
        // Poison the connections mutex by panicking inside a guard.
        let lc_clone = lc.clone();
        let _ = std::thread::spawn(move || {
            let _guard = lc_clone.connections.lock().unwrap();
            panic!("poison");
        })
        .join();
        // Now the mutex is poisoned; upsert_server must surface
        // LifecycleError::Connector instead of silently succeeding.
        let err = lc
            .upsert_server("srv", stdio("echo", true))
            .await
            .unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("poisoned")));
    }

    /// Poisons the connections mutex by panicking inside a guard.
    fn poison_connections(lc: &Arc<Lifecycle>) {
        let lc_clone = lc.clone();
        let _ = std::thread::spawn(move || {
            let _guard = lc_clone.connections.lock().unwrap();
            panic!("poison");
        })
        .join();
    }

    #[tokio::test]
    async fn connect_returns_error_on_poisoned_mutex() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Arc::new(Lifecycle::new(
            reg.clone(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        ));
        poison_connections(&lc);
        // After connector returns the new handle, connect tries to
        // insert it into the connections map — that's the poisoned
        // mutex branch. Must surface as Connector error AND mark the
        // registry Failed.
        let err = lc.connect("srv").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("poisoned")));
        assert!(matches!(
            reg.get("srv").unwrap().state,
            super::super::registry::ServerState::Failed { .. }
        ));
    }

    #[tokio::test]
    async fn disconnect_returns_error_on_poisoned_mutex() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Arc::new(Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        ));
        lc.connect("srv").await.unwrap();
        poison_connections(&lc);
        let err = lc.disconnect("srv", "test").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("poisoned")));
    }

    #[tokio::test]
    async fn call_tool_returns_error_on_poisoned_mutex() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Arc::new(Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec!["fetch".into()],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        ));
        lc.connect("srv").await.unwrap();
        poison_connections(&lc);
        let err = lc
            .call_tool("srv", "fetch", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("poisoned")));
    }

    #[tokio::test]
    async fn remove_server_returns_error_on_poisoned_mutex() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Arc::new(Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        ));
        lc.connect("srv").await.unwrap();
        let lc_clone = lc.clone();
        let _ = std::thread::spawn(move || {
            let _guard = lc_clone.connections.lock().unwrap();
            panic!("poison");
        })
        .join();
        let err = lc.remove_server("srv").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("poisoned")));
    }

    #[tokio::test]
    async fn disconnect_on_disabled_server_returns_disabled() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", false));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        // Disabled state must NOT be overwritten with Disconnected.
        assert!(matches!(
            lc.disconnect("srv", "test").await.unwrap_err(),
            LifecycleError::Disabled(_)
        ));
        assert!(matches!(
            reg.get("srv").unwrap().state,
            super::super::registry::ServerState::Disabled
        ));
    }

    /// Records the McpServerConfig it was handed, so a test can
    /// assert env-expansion happened before the connector ran.
    #[derive(Debug)]
    struct ConfigCapturingConnector {
        captured: Arc<std::sync::Mutex<Option<McpServerConfig>>>,
    }

    #[async_trait]
    impl Connector for ConfigCapturingConnector {
        async fn connect(
            &self,
            _name: &str,
            config: &McpServerConfig,
        ) -> Result<Arc<dyn Connection>, LifecycleError> {
            *self.captured.lock().unwrap() = Some(config.clone());
            Ok(Arc::new(FakeConnection { tools: vec![] }))
        }
    }

    #[tokio::test]
    async fn connect_passes_env_expanded_config_to_connector() {
        let reg = McpRegistry::new();
        reg.upsert(
            "srv",
            McpServerConfig::Stdio {
                command: "$BIN".into(),
                args: vec!["--token=$TOKEN".into()],
                env: [("AUTH".into(), "Bearer $TOKEN".into())].into(),
                cwd: None,
                enabled: true,
            },
        );
        let captured = Arc::new(std::sync::Mutex::new(None));
        let lc = Lifecycle::new(
            reg,
            Arc::new(ConfigCapturingConnector {
                captured: captured.clone(),
            }),
        )
        .with_env(
            [
                ("BIN".into(), "/usr/bin/srv".into()),
                ("TOKEN".into(), "ghp_xxx".into()),
            ]
            .into(),
        );
        lc.connect("srv").await.unwrap();
        let cfg = captured.lock().unwrap().clone().unwrap();
        match cfg {
            McpServerConfig::Stdio {
                command, args, env, ..
            } => {
                assert_eq!(command, "/usr/bin/srv");
                assert_eq!(args[0], "--token=ghp_xxx");
                assert_eq!(env.get("AUTH").unwrap(), "Bearer ghp_xxx");
            }
            _ => panic!("wrong variant"),
        }
    }

    /// First connect succeeds; second connect calls a connector that
    /// always fails. Lets us simulate a "registry believes Failed but
    /// stale handle still in map" race.
    #[derive(Debug)]
    struct OkThenFailingConnector {
        first_done: Arc<std::sync::Mutex<bool>>,
    }
    #[async_trait]
    impl Connector for OkThenFailingConnector {
        async fn connect(
            &self,
            _name: &str,
            _config: &McpServerConfig,
        ) -> Result<Arc<dyn Connection>, LifecycleError> {
            let mut first = self.first_done.lock().unwrap();
            if !*first {
                *first = true;
                Ok(Arc::new(FakeConnection {
                    tools: vec!["x".into()],
                }))
            } else {
                Err(LifecycleError::Connector("reconnect failed".into()))
            }
        }
    }

    #[tokio::test]
    async fn failed_reconnect_drops_stale_handle() {
        // Setup: first connect succeeds; we transition the registry
        // to Idle WITHOUT touching the connections map (the live
        // handle stays). The second connect tries to reconnect via
        // a failing connector — the registry must end up Failed AND
        // the stale handle must be dropped from the connection map,
        // so a subsequent call_tool returns UnknownServer rather
        // than dispatching through a dead handle.
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(OkThenFailingConnector {
                first_done: Arc::new(std::sync::Mutex::new(false)),
            }),
        );
        lc.connect("srv").await.unwrap();
        assert_eq!(lc.live_count(), 1);
        // Force the reconnect path: registry says Idle, map still has
        // the live handle.
        reg.set_state("srv", super::super::registry::ServerState::Idle);
        let err = lc.connect("srv").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("reconnect failed")));
        // Stale handle MUST be gone.
        assert_eq!(lc.live_count(), 0);
        // call_tool now correctly returns UnknownServer.
        let err2 = lc
            .call_tool("srv", "x", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err2, LifecycleError::UnknownServer(_)));
        // Registry is Failed with attempt_count=1.
        match reg.get("srv").unwrap().state {
            super::super::registry::ServerState::Failed { attempt_count, .. } => {
                assert_eq!(attempt_count, 1);
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn disconnect_propagates_live_handle_close_error() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FailingCloseThenFakeConnector {
                first_done: Arc::new(std::sync::Mutex::new(false)),
            }),
        );
        lc.connect("srv").await.unwrap();
        let err = lc.disconnect("srv", "test").await.unwrap_err();
        assert!(matches!(err, LifecycleError::Connector(ref m) if m.contains("close failed")));
        assert!(matches!(
            reg.get("srv").unwrap().state,
            super::super::registry::ServerState::Disconnected { .. }
        ));
    }

    #[tokio::test]
    async fn upsert_server_closes_stale_connection() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg,
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        lc.connect("srv").await.unwrap();
        assert_eq!(lc.live_count(), 1);
        // Re-upsert the same name with a different config — the
        // existing live handle must be dropped.
        lc.upsert_server("srv", stdio("echo", false)).await.unwrap();
        assert_eq!(lc.live_count(), 0);
    }

    #[tokio::test]
    async fn remove_server_closes_live_connection_and_drops_registry_entry() {
        let reg = McpRegistry::new();
        reg.upsert("srv", stdio("echo", true));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        lc.connect("srv").await.unwrap();
        assert!(lc.remove_server("srv").await.unwrap());
        assert_eq!(lc.live_count(), 0);
        assert!(reg.get("srv").is_none());
        // remove_server on an unknown name returns Ok(false).
        assert!(!lc.remove_server("ghost").await.unwrap());
    }

    #[tokio::test]
    async fn connect_unknown_does_not_grow_lock_map() {
        let lc = Lifecycle::new(
            McpRegistry::new(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        for _ in 0..10 {
            let _ = lc.connect("ghost").await;
        }
        // Lock map must not have allocated for the unknown name.
        let n = lc.connect_locks.lock().unwrap().len();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn disconnect_all_drops_every_connection() {
        let reg = McpRegistry::new();
        reg.upsert("a", stdio("a", true));
        reg.upsert("b", stdio("b", true));
        let lc = Lifecycle::new(
            reg.clone(),
            Arc::new(FakeConnector {
                tools: vec![],
                connect_calls: Arc::new(AtomicUsize::new(0)),
            }),
        );
        lc.connect_all().await;
        assert_eq!(lc.live_count(), 2);
        let results = lc.disconnect_all("shutdown").await;
        assert_eq!(results.len(), 2);
        assert_eq!(lc.live_count(), 0);
    }
}
