//! File-system permission sync for cross-process / cross-agent
//! approval flows (Phase 6 / Task 6.2).
//!
//! Workers write `PendingRequest` files into
//! `<team_root>/permissions/pending/<request_id>.json`. A leader
//! drains pending requests, asks an external approver (UI / Slack /
//! CLI prompt), and writes a `ResolvedResponse` to
//! `<team_root>/permissions/resolved/<request_id>.json`. Workers
//! poll for their response with a timeout — default-deny on
//! timeout via [`PermissionSync::request_or_default_deny`].
//!
//! Contract:
//! - `PermissionSync::request` is the worker side. Writes the
//!   pending file and polls `resolved/{id}.json` with a configurable
//!   interval. Returns the resolved decision or
//!   [`PermissionSyncError::Timeout`] / `Cancelled`.
//! - `PermissionSync::drain_pending` is the leader side: returns
//!   every pending request not yet resolved. Caller decides the
//!   decision.
//! - `PermissionSync::respond` writes the response file. Idempotent
//!   per request_id (later writes replace).
//!
//! `notify`-based fs-event watching is the design target; this batch
//! ships the polling fallback (sound semantics, ~30 ms latency on
//! local disk). Switching to `notify` is a follow-up that swaps
//! `poll_for_response` for an event-driven impl behind the same
//! method signature.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs;
use uuid::Uuid;

use crate::permission::PermissionDecision;

#[derive(Debug, Error)]
pub enum PermissionSyncError {
    #[error("permission sync io: {0}")]
    Io(#[from] std::io::Error),
    #[error("permission sync json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("response did not arrive within timeout")]
    Timeout,
    #[error("request cancelled")]
    Cancelled,
    #[error("permission sync: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingRequest {
    pub id: Uuid,
    pub from_agent: String,
    pub tool: String,
    pub input: serde_json::Value,
    pub timestamp_ms: u64,
}

impl PendingRequest {
    pub fn new(
        from_agent: impl Into<String>,
        tool: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            from_agent: from_agent.into(),
            tool: tool.into(),
            input,
            timestamp_ms: now_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedResponse {
    pub request_id: Uuid,
    pub decision: PermissionDecision,
    pub responded_at_ms: u64,
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
pub struct PermissionSync {
    pending_dir: PathBuf,
    resolved_dir: PathBuf,
}

impl PermissionSync {
    /// Construct under `<team_root>/permissions/`. Creates pending +
    /// resolved subdirectories on construction.
    pub async fn new(team_root: impl AsRef<Path>) -> Result<Self, PermissionSyncError> {
        let perm_root = team_root.as_ref().join("permissions");
        let pending_dir = perm_root.join("pending");
        let resolved_dir = perm_root.join("resolved");
        fs::create_dir_all(&pending_dir).await?;
        fs::create_dir_all(&resolved_dir).await?;
        Ok(Self {
            pending_dir,
            resolved_dir,
        })
    }

    pub fn pending_dir(&self) -> &Path {
        &self.pending_dir
    }

    pub fn resolved_dir(&self) -> &Path {
        &self.resolved_dir
    }

    /// Worker side: write pending request, poll for response, return
    /// decision. `poll_interval` controls how often the resolved
    /// directory is checked; sensible default is 50 ms.
    pub async fn request(
        &self,
        req: &PendingRequest,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<PermissionDecision, PermissionSyncError> {
        let pending_path = self.pending_dir.join(format!("{}.json", req.id));
        let body = serde_json::to_vec(req)?;
        // Atomic write: tmp → rename.
        let tmp = pending_path.with_extension("json.tmp");
        fs::write(&tmp, body).await?;
        fs::rename(&tmp, &pending_path).await?;

        let resolved_path = self.resolved_dir.join(format!("{}.json", req.id));
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            if tokio::fs::try_exists(&resolved_path).await? {
                let bytes = fs::read(&resolved_path).await?;
                let response: ResolvedResponse = serde_json::from_slice(&bytes)?;
                // Cleanup: remove the pending + resolved files now that
                // the worker has consumed the response.
                let _ = fs::remove_file(&pending_path).await;
                let _ = fs::remove_file(&resolved_path).await;
                return Ok(response.decision);
            }
            if tokio::time::Instant::now() >= deadline {
                // Cleanup pending so a slow leader's later write
                // doesn't pile up.
                let _ = fs::remove_file(&pending_path).await;
                return Err(PermissionSyncError::Timeout);
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Worker side convenience — caller-supplied default decision on
    /// timeout (typically Deny).
    pub async fn request_or_default(
        &self,
        req: &PendingRequest,
        timeout: Duration,
        poll_interval: Duration,
        default: PermissionDecision,
    ) -> PermissionDecision {
        match self.request(req, timeout, poll_interval).await {
            Ok(d) => d,
            Err(_) => default,
        }
    }

    /// Leader side: snapshot of every pending request (sorted by
    /// timestamp_ms ascending so older requests surface first).
    pub async fn drain_pending(&self) -> Result<Vec<PendingRequest>, PermissionSyncError> {
        let mut entries = fs::read_dir(&self.pending_dir).await?;
        let mut out = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let bytes = match fs::read(&path).await {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            let req: PendingRequest = match serde_json::from_slice(&bytes) {
                Ok(r) => r,
                Err(_) => continue, // tolerate partial writes
            };
            out.push(req);
        }
        out.sort_by_key(|r| r.timestamp_ms);
        Ok(out)
    }

    /// Leader side: write the response. Idempotent — later writes
    /// replace earlier ones for the same request_id.
    pub async fn respond(
        &self,
        request_id: Uuid,
        decision: PermissionDecision,
    ) -> Result<(), PermissionSyncError> {
        let response = ResolvedResponse {
            request_id,
            decision,
            responded_at_ms: now_ms(),
        };
        let body = serde_json::to_vec(&response)?;
        let path = self.resolved_dir.join(format!("{request_id}.json"));
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, body).await?;
        fs::rename(&tmp, &path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::{AllowDecision, DecisionReason, DenyDecision};
    use tempfile::tempdir;

    fn allow() -> PermissionDecision {
        PermissionDecision::Allow(AllowDecision {
            updated_input: None,
            reason: DecisionReason::other("test"),
        })
    }
    fn deny() -> PermissionDecision {
        PermissionDecision::Deny(DenyDecision {
            message_text: "test deny".into(),
            reason: DecisionReason::other("test"),
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn new_creates_pending_and_resolved_dirs() {
        let dir = tempdir().unwrap();
        let ps = PermissionSync::new(dir.path()).await.unwrap();
        assert!(ps.pending_dir().exists());
        assert!(ps.resolved_dir().exists());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn worker_request_resolved_by_leader() {
        let dir = tempdir().unwrap();
        let ps = PermissionSync::new(dir.path()).await.unwrap();
        let req = PendingRequest::new("worker-1", "Bash", serde_json::json!({"cmd": "ls"}));
        let req_id = req.id;
        let ps_leader = ps.clone();

        // Spawn leader that drains + responds.
        tokio::spawn(async move {
            // Wait briefly for the worker to write the request.
            tokio::time::sleep(Duration::from_millis(20)).await;
            let pending = ps_leader.drain_pending().await.unwrap();
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].id, req_id);
            ps_leader.respond(req_id, allow()).await.unwrap();
        });

        let decision = ps
            .request(&req, Duration::from_millis(2000), Duration::from_millis(20))
            .await
            .unwrap();
        assert!(decision.is_allow());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn worker_request_timeout_when_leader_silent() {
        let dir = tempdir().unwrap();
        let ps = PermissionSync::new(dir.path()).await.unwrap();
        let req = PendingRequest::new("worker-1", "Bash", serde_json::json!({}));
        let result = ps
            .request(&req, Duration::from_millis(50), Duration::from_millis(10))
            .await;
        assert!(matches!(result, Err(PermissionSyncError::Timeout)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn request_or_default_returns_caller_default_on_timeout() {
        let dir = tempdir().unwrap();
        let ps = PermissionSync::new(dir.path()).await.unwrap();
        let req = PendingRequest::new("w", "Bash", serde_json::json!({}));
        let decision = ps
            .request_or_default(
                &req,
                Duration::from_millis(40),
                Duration::from_millis(10),
                deny(),
            )
            .await;
        assert!(decision.is_deny());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn three_pending_resolved_two_allow_one_deny() {
        let dir = tempdir().unwrap();
        let ps = PermissionSync::new(dir.path()).await.unwrap();

        // Spawn leader that approves IDs containing "ok", denies the rest.
        let leader = ps.clone();
        let leader_task = tokio::spawn(async move {
            // Poll until 3 pending arrive.
            for _ in 0..200 {
                let pending = leader.drain_pending().await.unwrap();
                if pending.len() == 3 {
                    for req in pending {
                        let dec = if req.from_agent.contains("ok") {
                            allow()
                        } else {
                            deny()
                        };
                        leader.respond(req.id, dec).await.unwrap();
                    }
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!("leader timeout waiting for 3 pending");
        });

        let req_a = PendingRequest::new("ok-1", "T", serde_json::json!({}));
        let req_b = PendingRequest::new("ok-2", "T", serde_json::json!({}));
        let req_c = PendingRequest::new("bad-3", "T", serde_json::json!({}));
        let (a, b, c) = tokio::join!(
            ps.request(
                &req_a,
                Duration::from_millis(2000),
                Duration::from_millis(20)
            ),
            ps.request(
                &req_b,
                Duration::from_millis(2000),
                Duration::from_millis(20)
            ),
            ps.request(
                &req_c,
                Duration::from_millis(2000),
                Duration::from_millis(20)
            ),
        );
        leader_task.await.unwrap();

        assert!(a.unwrap().is_allow());
        assert!(b.unwrap().is_allow());
        assert!(c.unwrap().is_deny());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_pending_sorts_by_timestamp() {
        let dir = tempdir().unwrap();
        let ps = PermissionSync::new(dir.path()).await.unwrap();

        let mut older = PendingRequest::new("a", "T", serde_json::json!({}));
        older.timestamp_ms = 1000;
        let mut newer = PendingRequest::new("b", "T", serde_json::json!({}));
        newer.timestamp_ms = 5000;

        // Write newer first to verify sort.
        let p_newer = ps.pending_dir().join(format!("{}.json", newer.id));
        let p_older = ps.pending_dir().join(format!("{}.json", older.id));
        fs::write(&p_newer, serde_json::to_vec(&newer).unwrap())
            .await
            .unwrap();
        fs::write(&p_older, serde_json::to_vec(&older).unwrap())
            .await
            .unwrap();

        let drained = ps.drain_pending().await.unwrap();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].id, older.id);
        assert_eq!(drained[1].id, newer.id);
    }
}
