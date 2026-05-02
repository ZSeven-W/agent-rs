//! Remote session driver — host-side trait + helper for building a
//! method dispatcher from a reactive set of handlers.
//!
//! The host implements [`RemoteSession`] for whatever it actually
//! does (open / close / send / cancel) and feeds incoming
//! [`super::protocol::RpcMessage`] frames into [`dispatch`]. Dispatch
//! returns an outbound [`super::protocol::RpcResponse`] (when the
//! frame was a request) plus zero or more notifications produced by
//! the handler — typically streamed `session/event` pushes.

use async_trait::async_trait;

use super::protocol::{
    method, RpcError, RpcErrorCode, RpcId, RpcMessage, RpcNotification, RpcRequest, RpcResponse,
};

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session: {0}")]
    Other(String),
}

/// What dispatch should do with an inbound frame:
/// - Request → produce a single Response.
/// - Notification → no Response (one-way).
///
/// Either path may also produce side-channel notifications (e.g.,
/// `session/event` streams), accumulated in `pushes`.
#[derive(Debug, Default)]
pub struct DispatchOutcome {
    pub response: Option<RpcResponse>,
    pub pushes: Vec<RpcNotification>,
}

/// Host-side handler. Methods correspond 1:1 with the wire methods
/// in [`super::protocol::method`]. All return JSON values that the
/// dispatcher wraps into RpcResponse / RpcError envelopes.
#[async_trait]
pub trait RemoteSession: std::fmt::Debug + Send + Sync {
    async fn open(&self, params: serde_json::Value) -> Result<serde_json::Value, RpcError>;
    async fn close(&self, params: serde_json::Value) -> Result<serde_json::Value, RpcError>;
    /// Returns the SYNCHRONOUS acknowledgement (typically `{"queued": true}`).
    /// The actual `session/event` notifications are produced
    /// out-of-band by the host; the caller is responsible for
    /// pumping them back to the wire.
    async fn send(&self, params: serde_json::Value) -> Result<serde_json::Value, RpcError>;
    async fn cancel(&self, params: serde_json::Value) -> Result<serde_json::Value, RpcError>;
    /// Crate version + ack. Default impl returns the agent crate's
    /// VERSION constant.
    async fn ping(&self, _params: serde_json::Value) -> Result<serde_json::Value, RpcError> {
        Ok(serde_json::json!({
            "pong": true,
            "version": crate::VERSION,
        }))
    }
}

/// Dispatch one inbound frame against a [`RemoteSession`].
/// Single-call entry point — call once per frame received from the
/// codec.
pub async fn dispatch(session: &dyn RemoteSession, inbound: RpcMessage) -> DispatchOutcome {
    match inbound {
        RpcMessage::Request(req) => {
            let response = handle_request(session, req).await;
            DispatchOutcome {
                response: Some(response),
                pushes: vec![],
            }
        }
        RpcMessage::Notification(_) => {
            // Notifications are one-way — silently accept. Hosts
            // that care about specific notification methods can
            // peel them off before calling dispatch.
            DispatchOutcome::default()
        }
        RpcMessage::Response(_) => {
            // Responses are inbound only when this side made a
            // request. Out of scope for the simple session driver;
            // hosts that initiate requests need a request/response
            // correlator (typical for bi-directional protocols).
            DispatchOutcome::default()
        }
    }
}

async fn handle_request(session: &dyn RemoteSession, req: RpcRequest) -> RpcResponse {
    let id = req.id.clone();
    let params = req.params.unwrap_or(serde_json::Value::Null);
    let result = match req.method.as_str() {
        method::OPEN => session.open(params).await,
        method::CLOSE => session.close(params).await,
        method::SEND => session.send(params).await,
        method::CANCEL => session.cancel(params).await,
        method::PING => session.ping(params).await,
        other => Err(RpcError::new(
            RpcErrorCode::MethodNotFound,
            format!("unknown method `{other}`"),
        )),
    };
    match result {
        Ok(v) => RpcResponse::ok(id, v),
        Err(e) => RpcResponse::err(id, e),
    }
}

/// Build an event-push notification (`session/event`).
pub fn event_notification(session_id: &str, event: serde_json::Value) -> RpcNotification {
    RpcNotification::new(
        method::EVENT,
        Some(serde_json::json!({
            "session_id": session_id,
            "event": event,
        })),
    )
}

/// Build a turn-complete notification.
pub fn turn_complete_notification(
    session_id: &str,
    stop_reason: Option<&str>,
    model: Option<&str>,
) -> RpcNotification {
    let mut params = serde_json::Map::new();
    params.insert("session_id".into(), session_id.into());
    if let Some(s) = stop_reason {
        params.insert("stop_reason".into(), s.into());
    }
    if let Some(m) = model {
        params.insert("model".into(), m.into());
    }
    RpcNotification::new(
        method::TURN_COMPLETE,
        Some(serde_json::Value::Object(params)),
    )
}

/// Discard the request id wrapper for tests + assertions.
#[doc(hidden)]
pub fn _id_for_test(n: i64) -> RpcId {
    RpcId::Num(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct EchoSession;

    #[async_trait]
    impl RemoteSession for EchoSession {
        async fn open(&self, p: serde_json::Value) -> Result<serde_json::Value, RpcError> {
            Ok(serde_json::json!({"opened": p}))
        }
        async fn close(&self, _p: serde_json::Value) -> Result<serde_json::Value, RpcError> {
            Ok(serde_json::json!({"closed": true}))
        }
        async fn send(&self, _p: serde_json::Value) -> Result<serde_json::Value, RpcError> {
            Ok(serde_json::json!({"queued": true}))
        }
        async fn cancel(&self, _p: serde_json::Value) -> Result<serde_json::Value, RpcError> {
            Ok(serde_json::json!({"canceled": true}))
        }
    }

    fn req(id: i64, method: &str, params: serde_json::Value) -> RpcMessage {
        RpcMessage::Request(RpcRequest::new(RpcId::Num(id), method, Some(params)))
    }

    #[tokio::test]
    async fn dispatch_open_returns_session_id_in_result() {
        let s = EchoSession;
        let out = dispatch(
            &s,
            req(1, method::OPEN, serde_json::json!({"session_id": "S1"})),
        )
        .await;
        let resp = out.response.unwrap();
        assert_eq!(resp.error, None);
        assert_eq!(resp.result.as_ref().unwrap()["opened"]["session_id"], "S1");
    }

    #[tokio::test]
    async fn dispatch_unknown_method_returns_method_not_found() {
        let s = EchoSession;
        let out = dispatch(&s, req(2, "no.such", serde_json::json!({}))).await;
        let resp = out.response.unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, RpcErrorCode::MethodNotFound as i32);
    }

    #[tokio::test]
    async fn dispatch_notification_produces_no_response() {
        let s = EchoSession;
        let n = RpcMessage::Notification(RpcNotification::new(method::EVENT, None));
        let out = dispatch(&s, n).await;
        assert!(out.response.is_none());
        assert!(out.pushes.is_empty());
    }

    #[tokio::test]
    async fn ping_returns_version() {
        let s = EchoSession;
        let out = dispatch(&s, req(3, method::PING, serde_json::json!({}))).await;
        let resp = out.response.unwrap();
        let r = resp.result.unwrap();
        assert_eq!(r["pong"], true);
        assert!(r["version"].as_str().is_some());
    }

    #[tokio::test]
    async fn dispatch_routes_close_send_cancel() {
        let s = EchoSession;
        for method in [method::CLOSE, method::SEND, method::CANCEL] {
            let out = dispatch(&s, req(1, method, serde_json::json!({}))).await;
            assert!(out.response.unwrap().error.is_none(), "method {method}");
        }
    }

    #[test]
    fn event_notification_carries_session_id_and_event() {
        let n = event_notification("S1", serde_json::json!({"kind": "text_delta"}));
        assert_eq!(n.method, "session/event");
        let p = n.params.unwrap();
        assert_eq!(p["session_id"], "S1");
        assert_eq!(p["event"]["kind"], "text_delta");
    }

    #[test]
    fn turn_complete_notification_optional_fields() {
        let n = turn_complete_notification("S1", Some("end_turn"), None);
        let p = n.params.unwrap();
        assert_eq!(p["stop_reason"], "end_turn");
        assert!(p.get("model").is_none());
    }
}
