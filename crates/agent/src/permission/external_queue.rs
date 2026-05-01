//! External approval queue (Phase 3 / Task 3.2).
#![allow(clippy::result_large_err)]
// PermissionDecision is intentionally large (carries the rule + reason
// for diagnostic surfacing); requests already cross a tokio task
// boundary so a Box on the Ok variant doesn't materially help.
//!
//! Bridges the in-process [`PermissionManager`](super::PermissionManager)
//! to an out-of-process approver — a UI panel, a Slack hook, a CLI
//! prompt, etc. The queue exposes two halves: callers `request()` a
//! decision (yields the current task until a response is sent), and a
//! single host task drains [`ExternalQueueReceiver::next`] and
//! [`ExternalRequest::respond`]s to each pending request.

use std::time::Duration;

use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::types::{DecisionReason, DenyDecision, PermissionDecision};

/// One pending request flowing from agent to approver.
#[derive(Debug)]
pub struct ExternalRequest {
    pub id: Uuid,
    pub tool: String,
    pub input: serde_json::Value,
    sender: oneshot::Sender<PermissionDecision>,
}

impl ExternalRequest {
    /// Send the approver's decision back to the waiting agent.
    /// Returns `Err(decision)` if the requester dropped its receiver
    /// (rare — usually means the surrounding task was aborted).
    pub fn respond(self, decision: PermissionDecision) -> Result<(), PermissionDecision> {
        self.sender.send(decision)
    }
}

/// The agent-facing handle. Cheap to clone; cloning shares the same
/// underlying mpsc sender. Drop the last clone to close the queue.
#[derive(Debug, Clone)]
pub struct ExternalQueue {
    sender: mpsc::UnboundedSender<ExternalRequest>,
}

/// The approver-facing handle. **Single-consumer** by construction —
/// only one task should drain it. Pair it with a long-running task
/// that calls `next().await` in a loop and forwards each request to
/// the UI / approver, replying via `req.respond(decision)`.
#[derive(Debug)]
pub struct ExternalQueueReceiver {
    receiver: mpsc::UnboundedReceiver<ExternalRequest>,
}

impl ExternalQueueReceiver {
    pub async fn next(&mut self) -> Option<ExternalRequest> {
        self.receiver.recv().await
    }
}

/// Construct a queue + receiver pair.
pub fn external_queue() -> (ExternalQueue, ExternalQueueReceiver) {
    let (tx, rx) = mpsc::unbounded_channel();
    (
        ExternalQueue { sender: tx },
        ExternalQueueReceiver { receiver: rx },
    )
}

#[derive(Debug, Error)]
pub enum ExternalQueueError {
    #[error("external queue closed (no receiver)")]
    Closed,
    #[error("external request cancelled (sender dropped)")]
    Cancelled,
    #[error("external request timed out")]
    Timeout,
}

impl ExternalQueue {
    /// Submit a request and await the response. Returns `Err(Closed)` if
    /// the receiver was dropped, `Err(Cancelled)` if the responder
    /// dropped the oneshot without sending.
    pub async fn request(
        &self,
        tool: impl Into<String>,
        input: serde_json::Value,
    ) -> Result<PermissionDecision, ExternalQueueError> {
        let (tx, rx) = oneshot::channel();
        let req = ExternalRequest {
            id: Uuid::new_v4(),
            tool: tool.into(),
            input,
            sender: tx,
        };
        self.sender
            .send(req)
            .map_err(|_| ExternalQueueError::Closed)?;
        rx.await.map_err(|_| ExternalQueueError::Cancelled)
    }

    /// Submit + await with timeout. On deadline, returns `Err(Timeout)`
    /// without further state change — callers can map this to a
    /// default-deny decision via [`timeout_default_deny`].
    pub async fn request_with_timeout(
        &self,
        tool: impl Into<String>,
        input: serde_json::Value,
        timeout: Duration,
    ) -> Result<PermissionDecision, ExternalQueueError> {
        let fut = self.request(tool, input);
        match tokio::time::timeout(timeout, fut).await {
            Ok(result) => result,
            Err(_) => Err(ExternalQueueError::Timeout),
        }
    }

    /// Request with timeout, mapping every error to a caller-supplied
    /// fallback decision. Use this when you want full control over the
    /// "fail-closed" message + reason on timeout/closed/cancelled.
    ///
    /// Plan calls for "超时 default-deny（可配置）"; the configurable
    /// piece is this `fallback` closure. The closure receives the tool
    /// name and the underlying error so it can include either in the
    /// surfaced decision.
    pub async fn request_with_fallback<F>(
        &self,
        tool: impl Into<String>,
        input: serde_json::Value,
        timeout: Duration,
        fallback: F,
    ) -> PermissionDecision
    where
        F: FnOnce(&str, ExternalQueueError) -> PermissionDecision,
    {
        let tool = tool.into();
        match self.request_with_timeout(tool.clone(), input, timeout).await {
            Ok(d) => d,
            Err(e) => fallback(&tool, e),
        }
    }

    /// Convenience: request with timeout, mapping every error to the
    /// built-in [`timeout_default_deny`] decision. Equivalent to
    /// calling [`Self::request_with_fallback`] with `timeout_default_deny`.
    pub async fn request_or_default_deny(
        &self,
        tool: impl Into<String>,
        input: serde_json::Value,
        timeout: Duration,
    ) -> PermissionDecision {
        self.request_with_fallback(tool, input, timeout, timeout_default_deny)
            .await
    }
}

/// Build a Deny decision for the case where the external approval did
/// not arrive in time (or the queue was closed). Used by
/// [`ExternalQueue::request_or_default_deny`] but also exposed for
/// callers who want to construct one manually.
pub fn timeout_default_deny(tool: &str, err: ExternalQueueError) -> PermissionDecision {
    PermissionDecision::Deny(DenyDecision {
        message_text: format!(
            "Tool '{tool}' denied: external approval did not arrive ({err})."
        ),
        reason: DecisionReason::other(format!("external_queue: {err}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::{AllowDecision, DecisionReason as DR};

    fn allow() -> PermissionDecision {
        PermissionDecision::Allow(AllowDecision {
            updated_input: None,
            reason: DR::other("test"),
        })
    }

    #[tokio::test]
    async fn request_response_round_trip() {
        let (queue, mut receiver) = external_queue();

        // Spawn approver task.
        tokio::spawn(async move {
            while let Some(req) = receiver.next().await {
                req.respond(allow()).unwrap();
            }
        });

        let res = queue
            .request("Bash", serde_json::json!({"cmd": "ls"}))
            .await
            .unwrap();
        assert!(res.is_allow());
    }

    #[tokio::test]
    async fn timeout_returns_timeout_error() {
        let (queue, _receiver) = external_queue();
        // Receiver task NEVER processes — should hit timeout.
        let result = queue
            .request_with_timeout(
                "Bash",
                serde_json::json!({}),
                Duration::from_millis(20),
            )
            .await;
        assert!(matches!(result, Err(ExternalQueueError::Timeout)));
    }

    #[tokio::test]
    async fn request_or_default_deny_on_timeout() {
        let (queue, _receiver) = external_queue();
        let decision = queue
            .request_or_default_deny(
                "Bash",
                serde_json::json!({}),
                Duration::from_millis(20),
            )
            .await;
        assert!(decision.is_deny());
    }

    #[tokio::test]
    async fn request_with_custom_fallback() {
        let (queue, _receiver) = external_queue();
        // Custom fallback: ask the user instead of denying.
        let decision = queue
            .request_with_fallback(
                "Bash",
                serde_json::json!({}),
                Duration::from_millis(20),
                |_tool, _err| {
                    PermissionDecision::Ask(crate::permission::AskDecision {
                        message_text: "approver offline; please approve".into(),
                        reason: Some(DR::other("custom fallback")),
                    })
                },
            )
            .await;
        assert!(decision.is_ask());
    }

    #[tokio::test]
    async fn closed_queue_returns_closed_error() {
        let (queue, receiver) = external_queue();
        drop(receiver);
        let result = queue.request("Bash", serde_json::json!({})).await;
        assert!(matches!(result, Err(ExternalQueueError::Closed)));
    }

    #[tokio::test]
    async fn approver_can_drop_oneshot_to_signal_cancellation() {
        let (queue, mut receiver) = external_queue();
        tokio::spawn(async move {
            // Drain one request and drop without responding.
            let _req = receiver.next().await.unwrap();
            // _req drops here, oneshot sender drops.
        });
        let result = queue.request("Bash", serde_json::json!({})).await;
        assert!(matches!(result, Err(ExternalQueueError::Cancelled)));
    }

    #[tokio::test]
    async fn three_pending_requests_resolved_in_order() {
        let (queue, mut receiver) = external_queue();
        tokio::spawn(async move {
            for i in 0..3 {
                let req = receiver.next().await.unwrap();
                req.respond(PermissionDecision::Allow(AllowDecision {
                    updated_input: None,
                    reason: DR::other(format!("approved {i}")),
                }))
                .unwrap();
            }
        });

        let q1 = queue.clone();
        let q2 = queue.clone();
        let q3 = queue;
        let (a, b, c) = tokio::join!(
            q1.request("A", serde_json::json!({})),
            q2.request("B", serde_json::json!({})),
            q3.request("C", serde_json::json!({})),
        );
        // All approved.
        assert!(a.unwrap().is_allow());
        assert!(b.unwrap().is_allow());
        assert!(c.unwrap().is_allow());
    }
}
