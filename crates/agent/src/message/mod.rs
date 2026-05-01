//! Message tagged union with DAG headers (Phase 1 / Task 1.1).
//!
//! The agent's memory is a directed acyclic graph of messages linked by
//! `parent_uuid`. Each message owns a [`Header`] (uuid + parent + timestamp)
//! and a payload that depends on the variant. [`MessageStore`] is the
//! in-memory store with O(1) uuid→message lookup and parent-chain walks.

mod store;

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use store::MessageStore;

/// Common header attached to every [`Message`] variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub uuid: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_uuid: Option<Uuid>,
    pub timestamp_ms: u64,
}

impl Header {
    /// New root header — no parent.
    pub fn new() -> Self {
        Self {
            uuid: Uuid::new_v4(),
            parent_uuid: None,
            timestamp_ms: now_ms(),
        }
    }

    /// New child header pointing at `parent_uuid`.
    pub fn child_of(parent_uuid: Uuid) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            parent_uuid: Some(parent_uuid),
            timestamp_ms: now_ms(),
        }
    }
}

impl Default for Header {
    fn default() -> Self {
        Self::new()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// One element of an Anthropic-style message body.
///
/// Mirrors Anthropic's content-block schema (text / image / tool_use /
/// tool_result / thinking) so wire-level translation in providers is a
/// rename-only operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
}

/// Tool-result content can be a single string OR a list of content blocks
/// (Anthropic supports recursive content for richer tool outputs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Tagged enum over the kinds of message agent runtime knows about.
///
/// Variants:
/// - `User` — user-authored content delivered to the LLM.
/// - `Assistant` — LLM response (may include `ContentBlock::ToolUse`).
/// - `System` — system prompt or system note (single text payload).
/// - `Progress` — internal progress note (e.g., "compacting context...").
/// - `Tombstone` — replaces a deleted message in the DAG so child
///   `parent_uuid` references stay valid; carries a human-readable reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    User {
        header: Header,
        content: Vec<ContentBlock>,
    },
    Assistant {
        header: Header,
        content: Vec<ContentBlock>,
    },
    System {
        header: Header,
        text: String,
    },
    Progress {
        header: Header,
        note: String,
    },
    Tombstone {
        header: Header,
        reason: String,
    },
}

impl Message {
    /// Borrow the header. Every variant has one.
    pub fn header(&self) -> &Header {
        match self {
            Self::User { header, .. }
            | Self::Assistant { header, .. }
            | Self::System { header, .. }
            | Self::Progress { header, .. }
            | Self::Tombstone { header, .. } => header,
        }
    }

    /// Mutable borrow on the header — useful for tests / migrations.
    pub fn header_mut(&mut self) -> &mut Header {
        match self {
            Self::User { header, .. }
            | Self::Assistant { header, .. }
            | Self::System { header, .. }
            | Self::Progress { header, .. }
            | Self::Tombstone { header, .. } => header,
        }
    }

    pub fn uuid(&self) -> Uuid {
        self.header().uuid
    }

    pub fn parent_uuid(&self) -> Option<Uuid> {
        self.header().parent_uuid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_new_has_no_parent() {
        let h = Header::new();
        assert!(h.parent_uuid.is_none());
        assert!(h.timestamp_ms > 0);
    }

    #[test]
    fn header_child_of_links_parent() {
        let parent = Header::new();
        let child = Header::child_of(parent.uuid);
        assert_eq!(child.parent_uuid, Some(parent.uuid));
        assert_ne!(child.uuid, parent.uuid);
    }

    #[test]
    fn message_serde_roundtrip_user() {
        let msg = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text {
                text: "hi".into(),
            }],
        };
        let j = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&j).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn message_serde_roundtrip_tool_use() {
        let msg = Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::ToolUse {
                id: "tu_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({ "path": "/tmp/x" }),
            }],
        };
        let j = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&j).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn message_serde_roundtrip_tool_result_text() {
        let msg = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: ToolResultContent::Text("ok".into()),
                is_error: false,
            }],
        };
        let j = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&j).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn message_serde_roundtrip_tool_result_blocks() {
        let msg = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: ToolResultContent::Blocks(vec![ContentBlock::Text {
                    text: "result line 1".into(),
                }]),
                is_error: true,
            }],
        };
        let j = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&j).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn message_serde_roundtrip_thinking() {
        let msg = Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Thinking {
                thinking: "let me think...".into(),
                signature: Some("sig123".into()),
            }],
        };
        let j = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&j).unwrap();
        assert_eq!(msg, back);
    }

    #[test]
    fn message_header_accessors() {
        let h = Header::new();
        let mut msg = Message::System {
            header: h.clone(),
            text: "hi".into(),
        };
        assert_eq!(msg.uuid(), h.uuid);
        assert_eq!(msg.parent_uuid(), None);
        assert_eq!(msg.header(), &h);
        msg.header_mut().timestamp_ms = 12345;
        assert_eq!(msg.header().timestamp_ms, 12345);
    }
}
