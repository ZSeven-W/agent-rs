//! Adjacent-message grouping for compaction (claude-code parity).
//!
//! Mirror of `services/compact/grouping.ts`. Identifies "atomic"
//! groups that must NOT be split across a compaction boundary —
//! primarily `assistant tool_use` followed by its `user tool_result`.
//! Splitting those would leave a tool_use referencing a result that
//! lives only in the summary, which breaks multi-turn semantics on
//! every provider.

use crate::message::{ContentBlock, Message};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    /// Single User message (no tool result attached).
    UserOnly,
    /// Single Assistant message (no tool_use attached).
    AssistantOnly,
    /// Assistant emitted tool_use + the immediately-following User
    /// message carrying the matching tool_result. Atomic.
    ToolDispatchSequence,
    /// System / Progress / Tombstone — passes through as a single-
    /// item group.
    Auxiliary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageGroup {
    /// Inclusive start index in the original messages slice.
    pub start: usize,
    /// Inclusive end index.
    pub end: usize,
    pub kind: GroupKind,
}

impl MessageGroup {
    pub fn len(&self) -> usize {
        self.end + 1 - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True if compaction must NOT split inside this group.
    pub fn is_atomic(&self) -> bool {
        matches!(self.kind, GroupKind::ToolDispatchSequence)
    }
}

/// Walk `messages` left-to-right and emit groups.
pub fn group_messages(messages: &[Message]) -> Vec<MessageGroup> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < messages.len() {
        match &messages[i] {
            Message::Assistant { content, .. } if has_tool_use(content) => {
                // Look ahead: is the NEXT message a User carrying
                // any matching tool_result?
                let next_idx = i + 1;
                if next_idx < messages.len() {
                    if let Message::User {
                        content: next_content,
                        ..
                    } = &messages[next_idx]
                    {
                        let assistant_ids = collect_tool_use_ids(content);
                        let result_ids = collect_tool_result_ids(next_content);
                        if !assistant_ids.is_empty()
                            && assistant_ids.iter().any(|id| result_ids.contains(id))
                        {
                            out.push(MessageGroup {
                                start: i,
                                end: next_idx,
                                kind: GroupKind::ToolDispatchSequence,
                            });
                            i = next_idx + 1;
                            continue;
                        }
                    }
                }
                out.push(MessageGroup {
                    start: i,
                    end: i,
                    kind: GroupKind::AssistantOnly,
                });
                i += 1;
            }
            Message::Assistant { .. } => {
                out.push(MessageGroup {
                    start: i,
                    end: i,
                    kind: GroupKind::AssistantOnly,
                });
                i += 1;
            }
            Message::User { .. } => {
                out.push(MessageGroup {
                    start: i,
                    end: i,
                    kind: GroupKind::UserOnly,
                });
                i += 1;
            }
            Message::System { .. } | Message::Progress { .. } | Message::Tombstone { .. } => {
                out.push(MessageGroup {
                    start: i,
                    end: i,
                    kind: GroupKind::Auxiliary,
                });
                i += 1;
            }
        }
    }
    out
}

fn has_tool_use(content: &[ContentBlock]) -> bool {
    content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
}

fn collect_tool_use_ids(content: &[ContentBlock]) -> Vec<String> {
    content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

fn collect_tool_result_ids(content: &[ContentBlock]) -> Vec<String> {
    content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
            _ => None,
        })
        .collect()
}

/// Pick a safe split index: find the largest `i` ≤ `target` such
/// that splitting at `i` doesn't sever an atomic group. Returns
/// `target` itself if no atomic group spans the candidate point.
pub fn safe_split_index(groups: &[MessageGroup], target: usize) -> usize {
    let mut adjusted = target;
    for g in groups {
        if g.is_atomic() && g.start < adjusted && adjusted <= g.end {
            // Splitting inside the atomic group — back off to its start.
            adjusted = g.start;
        }
    }
    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Header, ToolResultContent};

    fn user_text(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
    fn assistant_text(text: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
    fn assistant_with_tool_use(id: &str, name: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input: serde_json::json!({}),
            }],
        }
    }
    fn user_with_tool_result(id: &str, body: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: ToolResultContent::Text(body.into()),
                is_error: false,
            }],
        }
    }

    #[test]
    fn pure_text_yields_one_group_per_message() {
        let msgs = vec![user_text("hi"), assistant_text("hello"), user_text("ok")];
        let groups = group_messages(&msgs);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].kind, GroupKind::UserOnly);
        assert_eq!(groups[1].kind, GroupKind::AssistantOnly);
        assert_eq!(groups[2].kind, GroupKind::UserOnly);
    }

    #[test]
    fn tool_use_then_tool_result_groups_together() {
        let msgs = vec![
            user_text("please calc"),
            assistant_with_tool_use("tu_1", "calc"),
            user_with_tool_result("tu_1", "42"),
            assistant_text("got 42"),
        ];
        let groups = group_messages(&msgs);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].kind, GroupKind::UserOnly);
        assert_eq!(groups[1].kind, GroupKind::ToolDispatchSequence);
        assert_eq!(groups[1].start, 1);
        assert_eq!(groups[1].end, 2);
        assert!(groups[1].is_atomic());
        assert_eq!(groups[2].kind, GroupKind::AssistantOnly);
    }

    #[test]
    fn assistant_with_tool_use_no_result_is_assistant_only() {
        let msgs = vec![
            assistant_with_tool_use("tu_1", "calc"),
            assistant_text("hi"),
        ];
        let groups = group_messages(&msgs);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].kind, GroupKind::AssistantOnly);
    }

    #[test]
    fn safe_split_index_backs_off_inside_atomic_group() {
        let msgs = vec![
            user_text("a"),
            assistant_with_tool_use("tu_1", "calc"),
            user_with_tool_result("tu_1", "42"),
            user_text("b"),
        ];
        let groups = group_messages(&msgs);
        // Target index 2 (after assistant tool_use, before result) —
        // would split the atomic pair; should back off to start of
        // the atomic group (1).
        assert_eq!(safe_split_index(&groups, 2), 1);
        // Target 3 is past the atomic pair; no adjustment.
        assert_eq!(safe_split_index(&groups, 3), 3);
    }

    #[test]
    fn safe_split_index_passes_through_when_no_atomic_overlap() {
        let msgs = vec![user_text("a"), assistant_text("b"), user_text("c")];
        let groups = group_messages(&msgs);
        assert_eq!(safe_split_index(&groups, 1), 1);
        assert_eq!(safe_split_index(&groups, 2), 2);
    }
}
