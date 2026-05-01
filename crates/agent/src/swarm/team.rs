//! Team — collection of [`SubAgent`]s with one designated leader,
//! sharing a single permission-sync root and one mailbox-root
//! filesystem location (Phase 6 / Task 6.3).

use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use tokio::fs;

use super::mailbox::{Mailbox, MailboxError, MailboxMessage};
use super::permission_sync::{PermissionSync, PermissionSyncError};
use super::sub_agent::SubAgent;
use super::task::SwarmTask;

#[derive(Debug, Error)]
pub enum TeamError {
    #[error("team mailbox: {0}")]
    Mailbox(#[from] MailboxError),
    #[error("team permission sync: {0}")]
    PermissionSync(#[from] PermissionSyncError),
    #[error("team io: {0}")]
    Io(#[from] std::io::Error),
    #[error("team: {0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct Team {
    pub name: String,
    pub root: PathBuf,
    pub leader_id: String,
    pub members: Vec<SubAgent>,
    pub permission_sync: Arc<PermissionSync>,
}

impl Team {
    /// Construct a team rooted at `<agent_root>/teams/{team_name}/`.
    /// Auto-creates the inbox + permissions directories. `leader_id`
    /// must match one of the member ids passed in or this returns
    /// [`TeamError::Other`].
    pub async fn new(
        agent_root: impl Into<PathBuf>,
        team_name: impl Into<String>,
        leader_id: impl Into<String>,
        member_specs: Vec<MemberSpec>,
    ) -> Result<Self, TeamError> {
        let team_name = team_name.into();
        let team_root = agent_root.into().join("teams").join(&team_name);
        fs::create_dir_all(&team_root).await?;

        let permission_sync = Arc::new(PermissionSync::new(&team_root).await?);

        let mut members = Vec::with_capacity(member_specs.len());
        for spec in member_specs {
            let mailbox = Mailbox::for_agent(&team_root, &spec.id).await?;
            members.push(SubAgent::new(
                spec.id,
                spec.role,
                mailbox,
                permission_sync.clone(),
            ));
        }

        let leader_id = leader_id.into();
        if !members.iter().any(|m| m.id == leader_id) {
            return Err(TeamError::Other(format!(
                "leader_id '{leader_id}' is not a member of team '{team_name}'"
            )));
        }

        Ok(Self {
            name: team_name,
            root: team_root,
            leader_id,
            members,
            permission_sync,
        })
    }

    pub fn leader(&self) -> Option<&SubAgent> {
        self.members.iter().find(|m| m.id == self.leader_id)
    }

    pub fn member(&self, id: &str) -> Option<&SubAgent> {
        self.members.iter().find(|m| m.id == id)
    }

    /// Push a [`SwarmTask`] to a member's mailbox as a payload-tagged
    /// envelope. Returns the message id for tracking.
    pub async fn delegate(&self, task: &SwarmTask) -> Result<uuid::Uuid, TeamError> {
        let assignee = task
            .assignee
            .as_deref()
            .ok_or_else(|| TeamError::Other("delegate: task has no assignee".into()))?;
        let target = self.member(assignee).ok_or_else(|| {
            TeamError::Other(format!(
                "delegate: assignee '{assignee}' is not a member of team '{}'",
                self.name
            ))
        })?;
        let payload = serde_json::json!({
            "kind": "task",
            "task": task,
        });
        let msg = MailboxMessage::new(self.leader_id.clone(), assignee, payload);
        let id = msg.id;
        target.mailbox.send(&msg).await?;
        Ok(id)
    }

    /// Stop every member. Idempotent.
    pub fn stop_all(&self) {
        for m in &self.members {
            m.stop();
        }
    }

    pub fn is_all_stopped(&self) -> bool {
        self.members.iter().all(|m| m.is_stopped())
    }
}

#[derive(Debug, Clone)]
pub struct MemberSpec {
    pub id: String,
    pub role: String,
}

impl MemberSpec {
    pub fn new(id: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: role.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn team_new_validates_leader_in_members() {
        let dir = tempdir().unwrap();
        let result = Team::new(
            dir.path().to_path_buf(),
            "design-squad",
            "ghost",
            vec![
                MemberSpec::new("alice", "leader"),
                MemberSpec::new("bob", "worker"),
            ],
        )
        .await;
        assert!(matches!(result, Err(TeamError::Other(_))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn team_creates_inbox_and_permission_dirs() {
        let dir = tempdir().unwrap();
        let team = Team::new(
            dir.path().to_path_buf(),
            "design-squad",
            "alice",
            vec![
                MemberSpec::new("alice", "leader"),
                MemberSpec::new("bob", "worker"),
            ],
        )
        .await
        .unwrap();
        assert!(team.root.exists());
        assert!(team.root.join("inboxes").exists());
        assert!(team.root.join("permissions/pending").exists());
        assert!(team.root.join("permissions/resolved").exists());
        assert_eq!(team.members.len(), 2);
        assert_eq!(team.leader().unwrap().id, "alice");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn delegate_lands_in_assignee_mailbox() {
        let dir = tempdir().unwrap();
        let team = Team::new(
            dir.path().to_path_buf(),
            "t1",
            "alice",
            vec![
                MemberSpec::new("alice", "leader"),
                MemberSpec::new("bob", "worker"),
            ],
        )
        .await
        .unwrap();
        let task = SwarmTask::new("alice", "ship it").assign_to("bob");
        let msg_id = team.delegate(&task).await.unwrap();

        let bob = team.member("bob").unwrap();
        let drained = bob.mailbox.drain().await.unwrap();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].id, msg_id);
        assert_eq!(drained[0].from, "alice");
        assert_eq!(drained[0].to, "bob");
        assert_eq!(drained[0].payload["kind"], "task");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn delegate_to_unknown_assignee_errors() {
        let dir = tempdir().unwrap();
        let team = Team::new(
            dir.path().to_path_buf(),
            "t1",
            "alice",
            vec![MemberSpec::new("alice", "leader")],
        )
        .await
        .unwrap();
        let task = SwarmTask::new("alice", "x").assign_to("ghost");
        let result = team.delegate(&task).await;
        assert!(matches!(result, Err(TeamError::Other(_))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stop_all_idempotent() {
        let dir = tempdir().unwrap();
        let team = Team::new(
            dir.path().to_path_buf(),
            "t1",
            "alice",
            vec![
                MemberSpec::new("alice", "leader"),
                MemberSpec::new("bob", "worker"),
            ],
        )
        .await
        .unwrap();
        assert!(!team.is_all_stopped());
        team.stop_all();
        team.stop_all();
        assert!(team.is_all_stopped());
    }
}
