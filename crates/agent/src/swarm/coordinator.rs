//! Coordinator — manages [`Team`] lifecycle (Phase 6 / Task 6.3).
//!
//! In Phase 6 batch O the Coordinator is intentionally minimal —
//! create / stop / poll a team — so consumers can drive the message
//! flow themselves. Phase 7+ may add a high-level "run team to
//! completion" loop that drains mailboxes, invokes QueryEngine per
//! tick, and watches permission-sync.

use std::path::PathBuf;

use super::mailbox::MailboxMessage;
use super::team::{MemberSpec, Team, TeamError};

#[derive(Debug)]
pub struct Coordinator {
    pub agent_root: PathBuf,
    pub teams: Vec<Team>,
}

impl Coordinator {
    pub fn new(agent_root: impl Into<PathBuf>) -> Self {
        Self {
            agent_root: agent_root.into(),
            teams: Vec::new(),
        }
    }

    /// Spawn a team under this coordinator. Returns a reference to
    /// the newly-created team for further interaction.
    pub async fn spawn_team(
        &mut self,
        team_name: impl Into<String>,
        leader_id: impl Into<String>,
        members: Vec<MemberSpec>,
    ) -> Result<&Team, TeamError> {
        let team = Team::new(self.agent_root.clone(), team_name, leader_id, members).await?;
        self.teams.push(team);
        Ok(self.teams.last().unwrap())
    }

    pub fn team(&self, name: &str) -> Option<&Team> {
        self.teams.iter().find(|t| t.name == name)
    }

    /// Stop a single team by name. No-op if not found.
    pub fn stop_team(&self, name: &str) {
        if let Some(t) = self.team(name) {
            t.stop_all();
        }
    }

    /// Stop every team this coordinator owns.
    pub fn stop_all(&self) {
        for t in &self.teams {
            t.stop_all();
        }
    }

    /// Drain every member mailbox across every team. Returns
    /// `Vec<(team, agent_id, MailboxMessage)>` for the caller to
    /// process. Useful for a single-tick poll loop.
    pub async fn poll_all_inboxes(&self) -> Vec<(String, String, MailboxMessage)> {
        let mut out = Vec::new();
        for team in &self.teams {
            for member in &team.members {
                if let Ok(msgs) = member.mailbox.drain().await {
                    for m in msgs {
                        out.push((team.name.clone(), member.id.clone(), m));
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    use super::super::task::SwarmTask;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn coordinator_spawns_and_stops_team() {
        let dir = tempdir().unwrap();
        let mut coord = Coordinator::new(dir.path().to_path_buf());
        let _ = coord
            .spawn_team(
                "t1",
                "alice",
                vec![
                    MemberSpec::new("alice", "leader"),
                    MemberSpec::new("bob", "worker"),
                ],
            )
            .await
            .unwrap();
        assert_eq!(coord.teams.len(), 1);
        assert!(coord.team("t1").is_some());
        coord.stop_all();
        assert!(coord.team("t1").unwrap().is_all_stopped());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn poll_all_inboxes_drains_every_team() {
        let dir = tempdir().unwrap();
        let mut coord = Coordinator::new(dir.path().to_path_buf());
        coord
            .spawn_team(
                "t1",
                "alice",
                vec![
                    MemberSpec::new("alice", "leader"),
                    MemberSpec::new("bob", "worker"),
                ],
            )
            .await
            .unwrap();
        coord
            .spawn_team(
                "t2",
                "carol",
                vec![
                    MemberSpec::new("carol", "leader"),
                    MemberSpec::new("dave", "worker"),
                ],
            )
            .await
            .unwrap();

        // Leader of t1 delegates to bob; leader of t2 delegates to dave.
        let t1 = coord.team("t1").unwrap();
        let t2 = coord.team("t2").unwrap();
        let task1 = SwarmTask::new("alice", "build").assign_to("bob");
        let task2 = SwarmTask::new("carol", "test").assign_to("dave");
        t1.delegate(&task1).await.unwrap();
        t2.delegate(&task2).await.unwrap();

        let drained = coord.poll_all_inboxes().await;
        assert_eq!(drained.len(), 2);
        let teams: std::collections::HashSet<_> = drained.iter().map(|(t, _, _)| t.clone()).collect();
        assert!(teams.contains("t1"));
        assert!(teams.contains("t2"));
    }
}
