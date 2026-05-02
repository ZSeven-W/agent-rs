//! MCP (Model Context Protocol) client (Tier 1 / claude-code parity).
//!
//! Mirrors `services/mcp/` from Claude Code. Wraps the MCP wire
//! protocol (provided by [`rmcp`]) with the agent-rs-specific glue:
//!
//! - [`config`] — server configuration schema (stdio / SSE /
//!   WebSocket transports) + env-var expansion.
//! - [`registry`] — in-memory registry of configured servers + their
//!   connection state.
//! - [`lifecycle`] — connect / discover / disconnect / call_tool
//!   driver. Pluggable [`Connector`] trait so production wires
//!   `rmcp` while tests inject mocks.
//! - [`auth`] — generic OAuth 2.0 + PKCE helper for remote MCP
//!   servers that require authorization-code flow. Inline SHA-256
//!   + base64url-no-pad to keep the dep tree thin.
//! - [`elicitation`] — handler trait for server-initiated user
//!   prompts (OpenUrl / Confirm / PromptText).
//! - [`permissions`] — per-channel allow/deny lists layered on top
//!   of the agent-wide [`crate::permission::PermissionManager`].
//!
//! Feature-gated on `mcp` (which pulls `rmcp = "1.5"`).

pub mod auth;
pub mod config;
pub mod connector;
pub mod elicitation;
pub mod lifecycle;
pub mod permissions;
pub mod registry;

pub use auth::{AuthError, OauthClient, PendingAuthorization, Tokens};
pub use config::{parse_json, parse_json_str, ConfigError, McpConfig, McpServerConfig};
pub use connector::{RmcpConnection, RmcpConnector};
pub use elicitation::{
    AutoConfirmElicitationHandler, ElicitationHandler, ElicitationHandlerRef, ElicitationRequest,
    ElicitationResponse, NoopElicitationHandler,
};
pub use lifecycle::{Connection, Connector, Lifecycle, LifecycleError};
pub use permissions::{
    combine as combine_permissions, ChannelDecision, ChannelPermissions, McpPermissionRegistry,
    OuterDecision,
};
pub use registry::{McpRegistry, RegisteredServer, ServerState};
