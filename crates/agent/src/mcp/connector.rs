//! Real MCP transport [`Connector`] implementation.
//!
//! Wraps the `rmcp` 1.5 client API behind the agent-rs [`Connector`] /
//! [`Connection`] traits. Three transports map to the
//! [`crate::mcp::McpServerConfig`] variants:
//!
//! | Variant   | Transport                                       | Status        |
//! |-----------|-------------------------------------------------|---------------|
//! | Stdio     | `rmcp::transport::TokioChildProcess`            | implemented   |
//! | Sse       | `StreamableHttpClientTransport<reqwest::Client>` (with `Accept: text/event-stream`) | implemented |
//! | WebSocket | `transport::ws` is gated behind a still-disabled feature (see rmcp 1.5 src/transport.rs:108-109) | returns `Connector("websocket transport not available in rmcp 1.5")` |
//!
//! The connector is feature-gated behind `mcp`. Hosts wire it up by
//! constructing [`RmcpConnector`] and passing it to
//! [`crate::mcp::Lifecycle::new`].
//!
//! # Tool result projection
//!
//! `rmcp::model::CallToolResult` carries:
//! - `content: Vec<Content>` — text/image blocks (the "rich" form).
//! - `structured_content: Option<Value>` — opaque host-defined JSON.
//! - `is_error: Option<bool>` — server-flagged failure.
//!
//! Our [`Connection::call_tool`] returns a single `serde_json::Value`.
//! Projection rules (most-specific wins):
//!
//! 1. If `is_error == Some(true)` → return `Err(LifecycleError::Connector(...))`
//!    so the host's permission/hook machinery sees a structured error.
//! 2. If `structured_content` is `Some` → return it verbatim.
//! 3. Else → return `serde_json::to_value(&content)` (an array of
//!    `{type, text|image|...}` blocks).

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientInfo, Implementation, Tool as McpTool,
};
use rmcp::service::{Peer, RoleClient, RunningService, ServiceExt};
use rmcp::transport::child_process::{ConfigureCommandExt, TokioChildProcess};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use tokio::process::Command;
use tokio::sync::Mutex as AsyncMutex;

use super::config::McpServerConfig;
use super::lifecycle::{Connection, Connector, LifecycleError};

fn build_implementation(name: &str, version: &str) -> Implementation {
    Implementation::new(name, version)
}

/// Production [`Connector`] backed by `rmcp` 1.5.
///
/// Cheap to clone — the client identity (`ClientInfo`) is shared via
/// `Arc`. One connector instance can produce many [`Connection`]s.
#[derive(Debug, Clone)]
pub struct RmcpConnector {
    client_info: Arc<ClientInfo>,
}

impl Default for RmcpConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl RmcpConnector {
    /// Construct with a sensible default `ClientInfo` advertising
    /// `agent-rs` + the crate version.
    pub fn new() -> Self {
        let mut info = ClientInfo::default();
        info.client_info = build_implementation("agent-rs", crate::VERSION);
        Self {
            client_info: Arc::new(info),
        }
    }

    /// Override the advertised client identity. Useful for hosts that
    /// want to surface their product name (OpenPencil, Zode) instead
    /// of the agent runtime's.
    pub fn with_client_info(mut self, info: ClientInfo) -> Self {
        self.client_info = Arc::new(info);
        self
    }
}

#[async_trait]
impl Connector for RmcpConnector {
    async fn connect(
        &self,
        name: &str,
        config: &McpServerConfig,
    ) -> Result<Arc<dyn Connection>, LifecycleError> {
        match config {
            McpServerConfig::Stdio {
                command,
                args,
                env,
                cwd,
                enabled: _,
            } => connect_stdio(self, name, command, args, env, cwd.as_deref()).await,
            McpServerConfig::Sse {
                url,
                headers,
                enabled: _,
            } => connect_streamable_http(self, name, url, headers).await,
            McpServerConfig::WebSocket { .. } => Err(LifecycleError::Connector(
                "websocket transport not available in rmcp 1.5; use sse or stdio".to_string(),
            )),
        }
    }
}

async fn connect_stdio(
    connector: &RmcpConnector,
    server_name: &str,
    command: &str,
    args: &[String],
    env: &BTreeMap<String, String>,
    cwd: Option<&str>,
) -> Result<Arc<dyn Connection>, LifecycleError> {
    // Capture the values into owned locals so the closure handed to
    // `configure` doesn't borrow connector / function args.
    let args_owned: Vec<String> = args.to_vec();
    let env_owned: Vec<(String, String)> =
        env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let cwd_owned: Option<String> = cwd.map(|s| s.to_string());
    let cmd = Command::new(command).configure(move |c| {
        c.args(&args_owned);
        for (k, v) in &env_owned {
            c.env(k, v);
        }
        if let Some(dir) = &cwd_owned {
            c.current_dir(dir);
        }
    });
    let transport = TokioChildProcess::new(cmd).map_err(|e| {
        LifecycleError::Connector(format!(
            "spawn '{command}' for MCP server '{server_name}' failed: {e}"
        ))
    })?;
    let client_info = (*connector.client_info).clone();
    let service = client_info.serve(transport).await.map_err(|e| {
        LifecycleError::Connector(format!(
            "MCP handshake with '{server_name}' over stdio failed: {e}"
        ))
    })?;
    let conn = RmcpConnection::new(server_name.to_string(), service).await?;
    Ok(Arc::new(conn))
}

async fn connect_streamable_http(
    connector: &RmcpConnector,
    server_name: &str,
    url: &str,
    headers: &BTreeMap<String, String>,
) -> Result<Arc<dyn Connection>, LifecycleError> {
    use http::{HeaderName, HeaderValue};
    // We deliberately do NOT use rmcp's `auth_header` shortcut: it
    // always sends `Authorization: Bearer <value>`, which corrupts
    // Basic / custom-scheme auth and is fragile under "Bearer "
    // capitalization / whitespace quirks. Instead, every host header
    // (including `Authorization`) is forwarded verbatim as a custom
    // header. rmcp's `RESERVED_HEADERS` set excludes `Authorization`,
    // so this is permitted.
    let mut custom: HashMap<HeaderName, HeaderValue> = HashMap::new();
    for (k, v) in headers {
        let name = HeaderName::try_from(k.as_bytes())
            .map_err(|e| LifecycleError::Connector(format!("invalid header name '{k}': {e}")))?;
        let value = HeaderValue::from_str(v).map_err(|e| {
            LifecycleError::Connector(format!("invalid header value for '{k}': {e}"))
        })?;
        custom.insert(name, value);
    }
    let config =
        StreamableHttpClientTransportConfig::with_uri(url.to_string()).custom_headers(custom);
    let transport = StreamableHttpClientTransport::with_client(reqwest::Client::new(), config);
    let client_info = (*connector.client_info).clone();
    let service = client_info.serve(transport).await.map_err(|e| {
        LifecycleError::Connector(format!(
            "MCP handshake with '{server_name}' over streamable HTTP ({url}) failed: {e}"
        ))
    })?;
    let conn = RmcpConnection::new(server_name.to_string(), service).await?;
    Ok(Arc::new(conn))
}

/// Live MCP connection. Holds:
///
/// - `peer`: a cheaply-cloneable `Peer<RoleClient>` used for every
///   tool call. Cloning it doesn't acquire the mutex, so concurrent
///   tool calls do not serialize and `close()` is not blocked while a
///   slow tool RPC is in flight.
/// - `service`: the owning `RunningService`, wrapped in
///   `AsyncMutex<Option<...>>` so `close()` can `take()` it for
///   graceful shutdown via `cancel().await`. Hosts MUST call and
///   await `close()` to completion for deterministic transport
///   teardown — dropping the `Arc<dyn Connection>` triggers rmcp's
///   `RunningService` drop guard, which only fires an async
///   cancellation; the underlying transport may still be in flight
///   when the drop returns.
/// - `tools` / `resources`: name snapshots populated at handshake.
///   Guarded by a sync mutex for cheap O(1) reads.
pub struct RmcpConnection {
    server_name: String,
    peer: Peer<RoleClient>,
    service: AsyncMutex<Option<RunningService<RoleClient, ClientInfo>>>,
    tools: Mutex<Vec<String>>,
    resources: Mutex<Vec<String>>,
}

impl std::fmt::Debug for RmcpConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RmcpConnection")
            .field("server_name", &self.server_name)
            .field("tools", &self.tools.lock().map(|g| g.clone()).ok())
            .field("resources", &self.resources.lock().map(|g| g.clone()).ok())
            .finish()
    }
}

impl RmcpConnection {
    async fn new(
        server_name: String,
        service: RunningService<RoleClient, ClientInfo>,
    ) -> Result<Self, LifecycleError> {
        let peer = service.peer().clone();
        let tools_vec = service.list_all_tools().await.map_err(|e| {
            LifecycleError::Connector(format!(
                "list_tools from MCP server '{server_name}' failed: {e}"
            ))
        })?;
        let tool_names: Vec<String> = tools_vec
            .iter()
            .map(|t: &McpTool| t.name.to_string())
            .collect();
        // Resource listing is optional — servers may not advertise the
        // `resources` capability. We treat `MethodNotFound` (-32601)
        // as "no resources" silently. Other failures (transport blip,
        // server-side internal error) are logged at WARN and the
        // resource list is left empty so an otherwise-usable tool
        // connection isn't blocked by a transient resource-discovery
        // failure. Hosts that need resource access can re-discover
        // later via the underlying rmcp peer.
        let resources_vec = match service.list_all_resources().await {
            Ok(v) => v.iter().map(|r| r.uri.clone()).collect(),
            Err(e) if is_method_not_found(&e) => {
                tracing::debug!(
                    target: "agent::mcp::connector",
                    %server_name,
                    "server doesn't advertise resources (MethodNotFound); treating as empty"
                );
                Vec::new()
            }
            Err(e) => {
                tracing::warn!(
                    target: "agent::mcp::connector",
                    %server_name,
                    error = %e,
                    "list_resources failed at handshake; continuing with empty resource list"
                );
                Vec::new()
            }
        };
        Ok(Self {
            server_name,
            peer,
            service: AsyncMutex::new(Some(service)),
            tools: Mutex::new(tool_names),
            resources: Mutex::new(resources_vec),
        })
    }
}

/// `true` for JSON-RPC `Method not found` (-32601) carried by
/// rmcp's `ServiceError::McpError`. Deliberately strict — we don't
/// match on stringified errors because a transport error body
/// containing the literal substring `-32601` would otherwise be
/// misclassified as "method not found", silently demoting a real
/// failure to "no resources".
fn is_method_not_found(err: &rmcp::ServiceError) -> bool {
    matches!(
        err,
        rmcp::ServiceError::McpError(data) if data.code.0 == -32601
    )
}

#[async_trait]
impl Connection for RmcpConnection {
    fn tool_names(&self) -> Vec<String> {
        self.tools.lock().map(|g| g.clone()).unwrap_or_default()
    }

    fn resource_uris(&self) -> Vec<String> {
        self.resources.lock().map(|g| g.clone()).unwrap_or_default()
    }

    async fn call_tool(
        &self,
        name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, LifecycleError> {
        let arguments = match input {
            serde_json::Value::Null => None,
            serde_json::Value::Object(map) => Some(map),
            other => {
                return Err(LifecycleError::Connector(format!(
                    "call_tool '{name}' on '{}' requires an object input or null, got {}",
                    self.server_name,
                    type_name(&other)
                )));
            }
        };
        let mut params = CallToolRequestParams::new(name.to_string());
        if let Some(map) = arguments {
            params = params.with_arguments(map);
        }
        // Hold the service mutex only long enough to verify the
        // connection is still open, then drop it before awaiting the
        // RPC. Otherwise a slow tool call would block `close()` and
        // serialize concurrent tool invocations on the same connection.
        {
            let guard = self.service.lock().await;
            if guard.is_none() {
                return Err(LifecycleError::Connector(format!(
                    "MCP connection to '{}' is closed; cannot call tool '{name}'",
                    self.server_name
                )));
            }
        }
        let result: CallToolResult = self.peer.call_tool(params).await.map_err(|e| {
            LifecycleError::Connector(format!(
                "call_tool '{name}' on '{}' failed: {e}",
                self.server_name
            ))
        })?;
        project_tool_result(name, &self.server_name, result)
    }

    async fn close(&self) -> Result<(), LifecycleError> {
        let mut guard = self.service.lock().await;
        if let Some(service) = guard.take() {
            service.cancel().await.map_err(|e| {
                LifecycleError::Connector(format!(
                    "graceful close of MCP server '{}' failed: {e}",
                    self.server_name
                ))
            })?;
        }
        Ok(())
    }
}

/// Project an `rmcp::CallToolResult` into our `serde_json::Value`
/// return shape. Pulled out into a free function so it's unit-testable
/// without standing up a full MCP server.
fn project_tool_result(
    tool_name: &str,
    server_name: &str,
    result: CallToolResult,
) -> Result<serde_json::Value, LifecycleError> {
    if matches!(result.is_error, Some(true)) {
        let detail = result
            .structured_content
            .as_ref()
            .map(|v| v.to_string())
            .or_else(|| {
                if result.content.is_empty() {
                    None
                } else {
                    serde_json::to_string(&result.content).ok()
                }
            })
            .unwrap_or_else(|| "<no detail>".to_string());
        return Err(LifecycleError::Connector(format!(
            "MCP server '{server_name}' tool '{tool_name}' returned isError=true: {detail}"
        )));
    }
    if let Some(structured) = result.structured_content {
        return Ok(structured);
    }
    serde_json::to_value(&result.content).map_err(|e| {
        LifecycleError::Connector(format!(
            "serializing tool '{tool_name}' content from '{server_name}' failed: {e}"
        ))
    })
}

fn type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmcp_connector_default_advertises_agent_rs() {
        let c = RmcpConnector::new();
        assert_eq!(c.client_info.client_info.name, "agent-rs");
        assert_eq!(c.client_info.client_info.version, crate::VERSION);
    }

    #[test]
    fn rmcp_connector_with_client_info_overrides() {
        let mut info = ClientInfo::default();
        info.client_info = build_implementation("openpencil", "1.0.0");
        let c = RmcpConnector::new().with_client_info(info);
        assert_eq!(c.client_info.client_info.name, "openpencil");
    }

    #[tokio::test]
    async fn websocket_transport_returns_clear_error() {
        let c = RmcpConnector::new();
        let cfg = McpServerConfig::WebSocket {
            url: "wss://example.com/mcp".into(),
            headers: Default::default(),
            enabled: true,
        };
        let err = c.connect("test", &cfg).await.expect_err("ws should fail");
        match err {
            LifecycleError::Connector(msg) => {
                assert!(msg.contains("websocket"), "got {msg}");
                assert!(msg.contains("not available"), "got {msg}");
            }
            other => panic!("expected Connector error, got {other:?}"),
        }
    }

    #[test]
    fn project_tool_result_prefers_structured_content() {
        let mut r = CallToolResult::default();
        r.structured_content = Some(serde_json::json!({"answer": 42}));
        let v = project_tool_result("calc", "demo", r).unwrap();
        assert_eq!(v, serde_json::json!({"answer": 42}));
    }

    #[test]
    fn project_tool_result_falls_back_to_content_array() {
        use rmcp::model::Content;
        let mut r = CallToolResult::default();
        r.content = vec![Content::text("hello")];
        let v = project_tool_result("echo", "demo", r).unwrap();
        // Content serializes to an array of typed blocks.
        assert!(v.is_array());
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "text");
    }

    #[test]
    fn project_tool_result_is_error_returns_connector_err() {
        let mut r = CallToolResult::default();
        r.is_error = Some(true);
        r.structured_content = Some(serde_json::json!({"reason": "boom"}));
        let err = project_tool_result("calc", "demo", r).expect_err("should err");
        let msg = match err {
            LifecycleError::Connector(m) => m,
            other => panic!("expected Connector, got {other:?}"),
        };
        assert!(msg.contains("isError=true"));
        assert!(msg.contains("boom"));
        assert!(msg.contains("calc"));
        assert!(msg.contains("demo"));
    }

    #[test]
    fn project_tool_result_is_error_with_empty_object_detail() {
        // structured_content = {} should surface as a literal "{}" detail,
        // not the no-detail sentinel. Lock the behavior so future
        // refactors don't silently change semantics.
        let mut r = CallToolResult::default();
        r.is_error = Some(true);
        r.structured_content = Some(serde_json::json!({}));
        let err = project_tool_result("calc", "demo", r).expect_err("should err");
        let msg = match err {
            LifecycleError::Connector(m) => m,
            other => panic!("expected Connector, got {other:?}"),
        };
        assert!(msg.contains("isError=true"));
        assert!(msg.contains("{}"), "got: {msg}");
        assert!(!msg.contains("<no detail>"), "got: {msg}");
    }

    #[test]
    fn project_tool_result_is_error_with_no_detail() {
        let mut r = CallToolResult::default();
        r.is_error = Some(true);
        // No content, no structured_content.
        let err = project_tool_result("ghost", "demo", r).expect_err("should err");
        let msg = match err {
            LifecycleError::Connector(m) => m,
            other => panic!("expected Connector, got {other:?}"),
        };
        assert!(msg.contains("<no detail>"), "got: {msg}");
    }

    #[test]
    fn is_method_not_found_recognizes_minus_32601() {
        // Construct a synthetic JSON-RPC error with code -32601.
        let data = rmcp::model::ErrorData {
            code: rmcp::model::ErrorCode(-32601),
            message: "Method not found".to_string().into(),
            data: None,
        };
        let err = rmcp::ServiceError::McpError(data);
        assert!(is_method_not_found(&err));
    }

    #[test]
    fn is_method_not_found_rejects_other_codes() {
        let data = rmcp::model::ErrorData {
            code: rmcp::model::ErrorCode(-32603), // internal error
            message: "Internal error".to_string().into(),
            data: None,
        };
        let err = rmcp::ServiceError::McpError(data);
        assert!(!is_method_not_found(&err));
    }

    #[tokio::test]
    async fn stdio_connect_with_nonexistent_binary_surfaces_helpful_error() {
        let c = RmcpConnector::new();
        let cfg = McpServerConfig::Stdio {
            command: "/this/binary/does/not/exist/no_such_mcp_server".into(),
            args: vec![],
            env: Default::default(),
            cwd: None,
            enabled: true,
        };
        let err = c
            .connect("ghost", &cfg)
            .await
            .expect_err("spawn should fail");
        match err {
            LifecycleError::Connector(msg) => {
                assert!(msg.contains("ghost"), "server name in msg: {msg}");
                assert!(
                    msg.contains("spawn") || msg.contains("No such") || msg.contains("not found"),
                    "spawn-style message expected: {msg}"
                );
            }
            other => panic!("expected Connector error, got {other:?}"),
        }
    }
}
