//! Remote session wire types (JSON-RPC 2.0).
//!
//! Frame envelope:
//!
//! - Requests carry an `id`, a `method`, and `params`.
//! - Responses carry the same `id` plus exactly one of `result` or
//!   `error`.
//! - Notifications carry a `method` + `params` but NO `id` (one-way).
//!
//! Stable method names live in [`method`] for typo-resistance.

use serde::{Deserialize, Serialize};

/// Request id — string OR number per JSON-RPC 2.0. Notifications
/// have no id; we model that as a separate type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RpcId {
    Str(String),
    Num(i64),
}

impl RpcId {
    pub fn from_string(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }
    pub fn from_num(n: i64) -> Self {
        Self::Num(n)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RpcRequest {
    /// MUST be exactly "2.0".
    pub jsonrpc: String,
    pub id: RpcId,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl RpcRequest {
    pub fn new(id: RpcId, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RpcResponse {
    pub jsonrpc: String,
    pub id: RpcId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl RpcResponse {
    pub fn ok(id: RpcId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }
    pub fn err(id: RpcId, error: RpcError) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl RpcNotification {
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            method: method.into(),
            params,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl RpcError {
    pub fn new(code: RpcErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: code as i32,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// Pre-defined JSON-RPC error codes plus our extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(i32)]
pub enum RpcErrorCode {
    /// JSON parse error.
    ParseError = -32700,
    /// Invalid request envelope.
    InvalidRequest = -32600,
    /// Method does not exist.
    MethodNotFound = -32601,
    /// Params don't match the method's schema.
    InvalidParams = -32602,
    /// Generic internal error.
    InternalError = -32603,
    /// Agent layer returned a typed error.
    AgentError = -32000,
    /// Aborted by the host.
    Aborted = -32001,
}

/// Untagged frame: any of the three message types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RpcMessage {
    Request(RpcRequest),
    Response(RpcResponse),
    Notification(RpcNotification),
}

/// Stable method-name constants. New methods extend this list.
pub mod method {
    /// Open a new session. Params: `{ session_id: String, working_dir: String, model: String }`.
    pub const OPEN: &str = "session/open";
    /// Close the session. Params: `{ session_id: String, reason?: String }`.
    pub const CLOSE: &str = "session/close";
    /// Send a user message. Params: `{ session_id: String, text: String }`.
    /// Streams back via `session/event` notifications.
    pub const SEND: &str = "session/send";
    /// Cancel the in-flight turn (if any). Params: `{ session_id: String }`.
    pub const CANCEL: &str = "session/cancel";
    /// Server-side push: one streaming event for an in-flight turn.
    /// Notification (no response). Params: `{ session_id, event }`.
    pub const EVENT: &str = "session/event";
    /// Server-side push: the agent finished the turn. Params:
    /// `{ session_id, stop_reason, model? }`.
    pub const TURN_COMPLETE: &str = "session/turn_complete";
    /// Health check. Params: none. Result: `{ pong: true, version: String }`.
    pub const PING: &str = "ping";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_id_round_trip_string() {
        let id = RpcId::Str("abc".into());
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"abc\"");
        assert_eq!(serde_json::from_str::<RpcId>(&json).unwrap(), id);
    }

    #[test]
    fn rpc_id_round_trip_number() {
        let id = RpcId::Num(42);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "42");
        assert_eq!(serde_json::from_str::<RpcId>(&json).unwrap(), id);
    }

    #[test]
    fn request_serializes_to_jsonrpc_2_0() {
        let r = RpcRequest::new(RpcId::Num(1), method::PING, None);
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["method"], "ping");
        assert_eq!(v["id"], 1);
    }

    #[test]
    fn response_ok_omits_error() {
        let r = RpcResponse::ok(RpcId::Num(1), serde_json::json!({"pong": true}));
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert!(v.get("error").is_none());
        assert_eq!(v["result"]["pong"], true);
    }

    #[test]
    fn response_err_omits_result() {
        let r = RpcResponse::err(
            RpcId::Num(1),
            RpcError::new(RpcErrorCode::MethodNotFound, "no such method"),
        );
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert!(v.get("result").is_none());
        assert_eq!(v["error"]["code"], -32601);
    }

    #[test]
    fn notification_omits_id() {
        let n = RpcNotification::new(method::EVENT, None);
        let v: serde_json::Value = serde_json::to_value(&n).unwrap();
        assert!(v.get("id").is_none());
        assert_eq!(v["method"], "session/event");
    }

    #[test]
    fn rpc_error_code_values_match_jsonrpc_spec() {
        assert_eq!(RpcErrorCode::ParseError as i32, -32700);
        assert_eq!(RpcErrorCode::InvalidRequest as i32, -32600);
        assert_eq!(RpcErrorCode::MethodNotFound as i32, -32601);
        assert_eq!(RpcErrorCode::InvalidParams as i32, -32602);
        assert_eq!(RpcErrorCode::InternalError as i32, -32603);
    }

    #[test]
    fn untagged_message_parses_request_response_notification() {
        // Request.
        let req_json = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        match serde_json::from_str::<RpcMessage>(req_json).unwrap() {
            RpcMessage::Request(_) => {}
            other => panic!("expected Request, got {other:?}"),
        }
        // Response.
        let resp_json = r#"{"jsonrpc":"2.0","id":1,"result":{"pong":true}}"#;
        match serde_json::from_str::<RpcMessage>(resp_json).unwrap() {
            RpcMessage::Response(_) => {}
            other => panic!("expected Response, got {other:?}"),
        }
        // Notification.
        let n_json = r#"{"jsonrpc":"2.0","method":"session/event"}"#;
        match serde_json::from_str::<RpcMessage>(n_json).unwrap() {
            RpcMessage::Notification(_) => {}
            other => panic!("expected Notification, got {other:?}"),
        }
    }

    #[test]
    fn rpc_error_with_data_preserves_payload() {
        let e = RpcError::new(RpcErrorCode::AgentError, "oops")
            .with_data(serde_json::json!({"detail": "x"}));
        let v: serde_json::Value = serde_json::to_value(&e).unwrap();
        assert_eq!(v["data"]["detail"], "x");
    }
}
