//! MCP server registry (Tier 1 / claude-code parity).
//!
//! Tracks all configured servers + their connection state. The host
//! adds servers (parsed from config), the lifecycle manager updates
//! state (Idle / Connecting / Connected / Disconnected / Failed /
//! Disabled), and the tool dispatcher reads from the registry to
//! route tool calls to the right server.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::config::McpServerConfig;

/// Connection state for a single server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ServerState {
    /// Configured but not yet attempted.
    Idle,
    /// Connect in progress.
    Connecting { started_at_unix_ms: u64 },
    /// Successfully connected; tools/resources discovered.
    Connected {
        connected_at_unix_ms: u64,
        /// Tool names discovered on this connection — used by the
        /// tool dispatcher to route.
        tool_names: Vec<String>,
        /// Resource URIs discovered.
        resource_uris: Vec<String>,
    },
    /// Cleanly disconnected (host shut it down).
    Disconnected { reason: String },
    /// Connect / runtime failed.
    Failed {
        message: String,
        attempt_count: u32,
        last_attempt_unix_ms: u64,
    },
    /// Server is disabled in config — manager skips connect.
    Disabled,
}

impl ServerState {
    pub fn is_connected(&self) -> bool {
        matches!(self, Self::Connected { .. })
    }

    pub fn tool_names(&self) -> &[String] {
        match self {
            Self::Connected { tool_names, .. } => tool_names,
            _ => &[],
        }
    }

    pub fn resource_uris(&self) -> &[String] {
        match self {
            Self::Connected { resource_uris, .. } => resource_uris,
            _ => &[],
        }
    }
}

/// Registry entry — config + state.
#[derive(Debug, Clone)]
pub struct RegisteredServer {
    pub name: String,
    pub config: McpServerConfig,
    pub state: ServerState,
}

/// In-memory registry. Cheap to clone (Arc-shared).
#[derive(Debug, Clone, Default)]
pub struct McpRegistry {
    inner: Arc<Mutex<BTreeMap<String, RegisteredServer>>>,
}

impl McpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace an entry.
    ///
    /// **Low-level**: this method does NOT close any live connection
    /// associated with `name`. If a [`super::lifecycle::Lifecycle`]
    /// is in flight, prefer
    /// [`super::lifecycle::Lifecycle::upsert_server`], which acquires
    /// the per-server connect lock + closes the existing handle
    /// before replacing the config.
    pub fn upsert(&self, name: impl Into<String>, config: McpServerConfig) {
        let name = name.into();
        let initial_state = if config.enabled() {
            ServerState::Idle
        } else {
            ServerState::Disabled
        };
        let entry = RegisteredServer {
            name: name.clone(),
            config,
            state: initial_state,
        };
        self.with_lock(|m| {
            m.insert(name, entry);
        });
    }

    /// Remove an entry. Returns true if it existed.
    ///
    /// **Low-level**: this method does NOT close any live connection
    /// associated with `name`. If a [`super::lifecycle::Lifecycle`]
    /// holds a live handle for this server, that handle becomes a
    /// ghost — the registry no longer knows about it, but
    /// [`super::lifecycle::Lifecycle::call_tool`] could still
    /// dispatch through it. Prefer
    /// [`super::lifecycle::Lifecycle::remove_server`] for coherent
    /// removal.
    pub fn remove(&self, name: &str) -> bool {
        self.with_lock(|m| m.remove(name).is_some())
    }

    /// Snapshot of all entries — for UI listing.
    pub fn snapshot(&self) -> Vec<RegisteredServer> {
        self.with_lock(|m| m.values().cloned().collect())
    }

    /// Get a single server.
    pub fn get(&self, name: &str) -> Option<RegisteredServer> {
        self.with_lock(|m| m.get(name).cloned())
    }

    /// Update state for a server. No-op if the server is unknown.
    pub fn set_state(&self, name: &str, state: ServerState) {
        self.with_lock(|m| {
            if let Some(entry) = m.get_mut(name) {
                entry.state = state;
            }
        });
    }

    /// Convenience: mark connecting.
    pub fn mark_connecting(&self, name: &str) {
        self.set_state(
            name,
            ServerState::Connecting {
                started_at_unix_ms: now_ms(),
            },
        );
    }

    /// Convenience: mark connected with discovered tools/resources.
    pub fn mark_connected(&self, name: &str, tool_names: Vec<String>, resource_uris: Vec<String>) {
        self.set_state(
            name,
            ServerState::Connected {
                connected_at_unix_ms: now_ms(),
                tool_names,
                resource_uris,
            },
        );
    }

    /// Convenience: mark a failure, incrementing attempt_count if we
    /// were already in `Failed` state.
    pub fn mark_failed(&self, name: &str, message: impl Into<String>) {
        let message = message.into();
        self.with_lock(|m| {
            if let Some(entry) = m.get_mut(name) {
                let attempt_count = match &entry.state {
                    ServerState::Failed { attempt_count, .. } => attempt_count.saturating_add(1),
                    _ => 1,
                };
                entry.state = ServerState::Failed {
                    message,
                    attempt_count,
                    last_attempt_unix_ms: now_ms(),
                };
            }
        });
    }

    /// Convenience: mark disconnected.
    pub fn mark_disconnected(&self, name: &str, reason: impl Into<String>) {
        self.set_state(
            name,
            ServerState::Disconnected {
                reason: reason.into(),
            },
        );
    }

    /// Find which server (if any) advertises `tool_name`.
    ///
    /// **Ordering**: the underlying storage is a [`BTreeMap`] keyed by
    /// server name, so iteration is lexicographic. When two servers
    /// advertise the same tool name, the one with the
    /// alphabetically-earlier server name wins. Hosts that need
    /// deterministic priority should either:
    /// - namespace tools at the dispatch layer (`<server>:<tool>`), or
    /// - reject duplicate tool names at registration time, or
    /// - prefix server names with a numeric priority sigil
    ///   (`"00-priority-server"`).
    ///
    /// This contract is documented + tested rather than dynamically
    /// enforced — claude-code parity does the same.
    pub fn find_server_for_tool(&self, tool_name: &str) -> Option<String> {
        self.with_lock(|m| {
            m.iter()
                .find(|(_, e)| e.state.tool_names().iter().any(|t| t == tool_name))
                .map(|(name, _)| name.clone())
        })
    }

    /// Total registered servers.
    pub fn len(&self) -> usize {
        self.with_lock(|m| m.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn with_lock<R>(&self, f: impl FnOnce(&mut BTreeMap<String, RegisteredServer>) -> R) -> R {
        let mut guard = self.inner.lock().unwrap_or_else(|p| {
            // Poisoned mutex — recover the inner state. Registry
            // poisoning is non-fatal (no invariant beyond the map).
            p.into_inner()
        });
        f(&mut guard)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio(cmd: &str, enabled: bool) -> McpServerConfig {
        McpServerConfig::Stdio {
            command: cmd.into(),
            args: vec![],
            env: Default::default(),
            cwd: None,
            enabled,
        }
    }

    #[test]
    fn upsert_then_get_round_trip() {
        let r = McpRegistry::new();
        r.upsert("a", stdio("echo", true));
        let got = r.get("a").unwrap();
        assert_eq!(got.name, "a");
        assert!(matches!(got.state, ServerState::Idle));
    }

    #[test]
    fn disabled_config_starts_disabled() {
        let r = McpRegistry::new();
        r.upsert("a", stdio("echo", false));
        assert!(matches!(r.get("a").unwrap().state, ServerState::Disabled));
    }

    #[test]
    fn mark_connected_records_tools() {
        let r = McpRegistry::new();
        r.upsert("a", stdio("echo", true));
        r.mark_connecting("a");
        r.mark_connected("a", vec!["fetch".into(), "ping".into()], vec![]);
        let got = r.get("a").unwrap();
        match got.state {
            ServerState::Connected { tool_names, .. } => {
                assert_eq!(tool_names, vec!["fetch".to_string(), "ping".to_string()]);
            }
            _ => panic!("expected Connected"),
        }
    }

    #[test]
    fn mark_failed_increments_attempts() {
        let r = McpRegistry::new();
        r.upsert("a", stdio("echo", true));
        r.mark_failed("a", "no");
        r.mark_failed("a", "still no");
        match r.get("a").unwrap().state {
            ServerState::Failed { attempt_count, .. } => assert_eq!(attempt_count, 2),
            _ => panic!("expected Failed"),
        }
    }

    #[test]
    fn find_server_for_tool() {
        let r = McpRegistry::new();
        r.upsert("github", stdio("g", true));
        r.upsert("linear", stdio("l", true));
        r.mark_connected("github", vec!["create_issue".into()], vec![]);
        r.mark_connected("linear", vec!["search".into()], vec![]);
        assert_eq!(
            r.find_server_for_tool("create_issue").as_deref(),
            Some("github")
        );
        assert_eq!(r.find_server_for_tool("search").as_deref(), Some("linear"));
        assert!(r.find_server_for_tool("nope").is_none());
    }

    #[test]
    fn find_server_for_tool_returns_lex_earliest_on_dupe() {
        // Two servers both advertise "fetch". The contract: the
        // lex-earliest server name wins. Pin this so future code
        // changes don't silently flip the priority.
        let r = McpRegistry::new();
        r.upsert("z-server", stdio("z", true));
        r.upsert("a-server", stdio("a", true));
        r.mark_connected("z-server", vec!["fetch".into()], vec![]);
        r.mark_connected("a-server", vec!["fetch".into()], vec![]);
        assert_eq!(r.find_server_for_tool("fetch").as_deref(), Some("a-server"));
    }

    #[test]
    fn snapshot_returns_all() {
        let r = McpRegistry::new();
        r.upsert("a", stdio("a", true));
        r.upsert("b", stdio("b", false));
        assert_eq!(r.snapshot().len(), 2);
    }

    #[test]
    fn remove_drops_entry() {
        let r = McpRegistry::new();
        r.upsert("a", stdio("a", true));
        assert!(r.remove("a"));
        assert!(!r.remove("a"));
        assert!(r.is_empty());
    }

    #[test]
    fn unknown_set_state_is_no_op() {
        let r = McpRegistry::new();
        r.set_state("ghost", ServerState::Idle); // must not panic
        assert!(r.is_empty());
    }
}
