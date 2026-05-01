//! Append-only message store with uuid index + parent-chain traversal.

use std::collections::HashMap;

use uuid::Uuid;

use super::Message;
use crate::error::AgentError;

/// In-memory message DAG. Append-only by construction — once pushed, a
/// message's slot is immutable (use a [`Message::Tombstone`] to logically
/// delete while keeping the DAG dangling-free).
#[derive(Debug, Clone, Default)]
pub struct MessageStore {
    messages: Vec<Message>,
    by_uuid: HashMap<Uuid, usize>,
}

impl MessageStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a message. Returns `Err(AgentError::DuplicateUuid)` if a
    /// message with the same uuid is already present.
    pub fn push(&mut self, message: Message) -> Result<(), AgentError> {
        let uuid = message.uuid();
        if self.by_uuid.contains_key(&uuid) {
            return Err(AgentError::DuplicateUuid(uuid));
        }
        let idx = self.messages.len();
        self.by_uuid.insert(uuid, idx);
        self.messages.push(message);
        Ok(())
    }

    pub fn get(&self, uuid: Uuid) -> Option<&Message> {
        self.by_uuid.get(&uuid).map(|&i| &self.messages[i])
    }

    /// Walk one step up the DAG.
    pub fn parent_of(&self, uuid: Uuid) -> Option<&Message> {
        let parent_uuid = self.get(uuid)?.parent_uuid()?;
        self.get(parent_uuid)
    }

    /// Walk all the way to the root, including the message at `uuid`.
    /// Returns `[uuid, parent, grandparent, ..., root]`.
    ///
    /// **Silent-truncation contract** (intentional): `push()` is
    /// permissive — it does not enforce that `parent_uuid` points to a
    /// message already in the store, because callers may replay or load
    /// partial subtrees. As a result, `ancestors()` may return a chain
    /// shorter than the conceptual lineage if a parent_uuid dangles.
    /// If `uuid` itself is not in the store, returns an empty vec.
    ///
    /// Use [`MessageStore::ancestors_checked`] when a dangling parent
    /// must be treated as data corruption.
    pub fn ancestors(&self, uuid: Uuid) -> Vec<&Message> {
        let mut chain = Vec::new();
        let mut current = Some(uuid);
        while let Some(u) = current {
            match self.get(u) {
                Some(msg) => {
                    chain.push(msg);
                    current = msg.parent_uuid();
                }
                None => break,
            }
        }
        chain
    }

    /// Strict variant of [`MessageStore::ancestors`].
    ///
    /// Errors with [`AgentError::InvalidMessage`] if `uuid` itself is not
    /// in the store, or if any walked message's `parent_uuid` points at a
    /// uuid not present in the store.
    pub fn ancestors_checked(&self, uuid: Uuid) -> Result<Vec<&Message>, AgentError> {
        let mut chain = Vec::new();
        let mut current = Some(uuid);
        while let Some(u) = current {
            let msg = self.get(u).ok_or_else(|| {
                AgentError::InvalidMessage(format!(
                    "dangling parent uuid {u} during DAG walk"
                ))
            })?;
            chain.push(msg);
            current = msg.parent_uuid();
        }
        Ok(chain)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ContentBlock, Header, Message};
    use super::*;

    fn user_msg(parent: Option<Uuid>, text: &str) -> Message {
        let header = match parent {
            Some(p) => Header::child_of(p),
            None => Header::new(),
        };
        Message::User {
            header,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[test]
    fn push_then_get() {
        let mut s = MessageStore::new();
        let m = user_msg(None, "hi");
        let id = m.uuid();
        s.push(m.clone()).unwrap();
        assert_eq!(s.get(id), Some(&m));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn duplicate_uuid_rejected() {
        let mut s = MessageStore::new();
        let m = user_msg(None, "hi");
        s.push(m.clone()).unwrap();
        match s.push(m) {
            Err(AgentError::DuplicateUuid(_)) => {}
            other => panic!("expected DuplicateUuid, got {other:?}"),
        }
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn three_message_parent_chain() {
        let mut s = MessageStore::new();
        let a = user_msg(None, "a");
        let a_id = a.uuid();
        let b = user_msg(Some(a_id), "b");
        let b_id = b.uuid();
        let c = user_msg(Some(b_id), "c");
        let c_id = c.uuid();
        s.push(a).unwrap();
        s.push(b).unwrap();
        s.push(c).unwrap();

        assert_eq!(s.parent_of(c_id).map(|m| m.uuid()), Some(b_id));
        assert_eq!(s.parent_of(b_id).map(|m| m.uuid()), Some(a_id));
        assert_eq!(s.parent_of(a_id).map(|m| m.uuid()), None);

        let chain: Vec<Uuid> = s.ancestors(c_id).iter().map(|m| m.uuid()).collect();
        assert_eq!(chain, vec![c_id, b_id, a_id]);
    }

    #[test]
    fn ancestors_of_unknown_is_empty() {
        let s = MessageStore::new();
        assert!(s.ancestors(Uuid::new_v4()).is_empty());
    }

    #[test]
    fn ancestors_checked_of_unknown_errors() {
        let s = MessageStore::new();
        let unknown = Uuid::new_v4();
        match s.ancestors_checked(unknown) {
            Err(AgentError::InvalidMessage(msg)) => assert!(msg.contains(&unknown.to_string())),
            other => panic!("expected InvalidMessage, got {other:?}"),
        }
    }

    #[test]
    fn ancestors_silently_truncates_on_dangling_parent() {
        let mut s = MessageStore::new();
        let dangling_uuid = Uuid::new_v4();
        let m = Message::User {
            header: super::super::Header {
                uuid: Uuid::new_v4(),
                parent_uuid: Some(dangling_uuid),
                timestamp_ms: 0,
            },
            content: vec![ContentBlock::Text { text: "orphan".into() }],
        };
        let m_id = m.uuid();
        s.push(m).unwrap();
        // ancestors stops at m, returning just [m] — silent truncation.
        let chain = s.ancestors(m_id);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].uuid(), m_id);
    }

    #[test]
    fn ancestors_checked_errors_on_dangling_parent() {
        let mut s = MessageStore::new();
        let dangling_uuid = Uuid::new_v4();
        let m = Message::User {
            header: super::super::Header {
                uuid: Uuid::new_v4(),
                parent_uuid: Some(dangling_uuid),
                timestamp_ms: 0,
            },
            content: vec![ContentBlock::Text { text: "orphan".into() }],
        };
        let m_id = m.uuid();
        s.push(m).unwrap();
        match s.ancestors_checked(m_id) {
            Err(AgentError::InvalidMessage(msg)) => {
                assert!(msg.contains(&dangling_uuid.to_string()));
            }
            other => panic!("expected InvalidMessage, got {other:?}"),
        }
    }

    #[test]
    fn iter_preserves_insertion_order() {
        let mut s = MessageStore::new();
        let a = user_msg(None, "a");
        let b = user_msg(Some(a.uuid()), "b");
        let texts: Vec<&str> = vec!["a", "b"];
        s.push(a).unwrap();
        s.push(b).unwrap();
        let got: Vec<String> = s
            .iter()
            .map(|m| match m {
                Message::User { content, .. } => match &content[0] {
                    ContentBlock::Text { text } => text.clone(),
                    _ => panic!(),
                },
                _ => panic!(),
            })
            .collect();
        assert_eq!(got, texts);
    }
}
