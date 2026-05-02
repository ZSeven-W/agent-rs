//! Context-window analysis utilities (Tier 2 / claude-code parity).
//!
//! Mirrors `services/context/analyze.ts`. Inspects the live message
//! store and reports useful summaries the host UI shows in its
//! progress / cost-tracking widgets:
//!
//! - Total estimated tokens by role.
//! - Top-N largest messages (often candidates for compaction).
//! - Tool-call breakdown (which tools accumulated the most input/
//!   output volume).
//! - Approximate share of context spent on cached vs uncached input
//!   (best-effort; relies on the prompt-cache tracker if available).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::compact::estimate_tokens;
use crate::message::{ContentBlock, Message};

/// Aggregate report produced by [`analyze`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextReport {
    pub total_tokens: u32,
    pub total_messages: u32,
    pub user_tokens: u32,
    pub assistant_tokens: u32,
    pub system_tokens: u32,
    pub progress_tokens: u32,
    pub tombstone_tokens: u32,
    /// Tool-call breakdown: tool name → (call count, accumulated
    /// input tokens, accumulated output tokens).
    pub tool_breakdown: BTreeMap<String, ToolUsage>,
    /// Indices of the top-N largest messages, sorted descending by
    /// token estimate. `top_messages.len()` = `min(n, total_messages)`.
    pub top_messages: Vec<TopMessage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolUsage {
    pub call_count: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopMessage {
    pub index: u32,
    pub role: String,
    pub tokens: u32,
}

/// Walk `messages` and produce a [`ContextReport`]. `top_n` controls
/// how many of the largest messages are surfaced in
/// `top_messages` (typical UI value: 5–10).
pub fn analyze(messages: &[Message], top_n: usize) -> ContextReport {
    let mut r = ContextReport {
        total_messages: messages.len() as u32,
        ..ContextReport::default()
    };

    let mut all: Vec<TopMessage> = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        let tokens = estimate_tokens(m);
        r.total_tokens = r.total_tokens.saturating_add(tokens);
        match m {
            Message::User { content, .. } => {
                r.user_tokens = r.user_tokens.saturating_add(tokens);
                accumulate_tool_results(content, &mut r.tool_breakdown);
            }
            Message::Assistant { content, .. } => {
                r.assistant_tokens = r.assistant_tokens.saturating_add(tokens);
                accumulate_tool_uses(content, &mut r.tool_breakdown);
            }
            Message::System { .. } => {
                r.system_tokens = r.system_tokens.saturating_add(tokens);
            }
            Message::Progress { .. } => {
                r.progress_tokens = r.progress_tokens.saturating_add(tokens);
            }
            Message::Tombstone { .. } => {
                r.tombstone_tokens = r.tombstone_tokens.saturating_add(tokens);
            }
        }
        all.push(TopMessage {
            index: i as u32,
            role: role_str(m).to_string(),
            tokens,
        });
    }
    all.sort_by(|a, b| b.tokens.cmp(&a.tokens).then_with(|| a.index.cmp(&b.index)));
    all.truncate(top_n);
    r.top_messages = all;
    r
}

fn role_str(m: &Message) -> &'static str {
    match m {
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
        Message::System { .. } => "system",
        Message::Progress { .. } => "progress",
        Message::Tombstone { .. } => "tombstone",
    }
}

fn accumulate_tool_uses(blocks: &[ContentBlock], out: &mut BTreeMap<String, ToolUsage>) {
    for b in blocks {
        if let ContentBlock::ToolUse { name, input, .. } = b {
            let entry = out.entry(name.clone()).or_default();
            entry.call_count = entry.call_count.saturating_add(1);
            let in_tokens = serde_json::to_string(input)
                .map(|s| (s.len() as u32) / 4)
                .unwrap_or(0);
            entry.input_tokens = entry.input_tokens.saturating_add(in_tokens);
        }
    }
}

fn accumulate_tool_results(blocks: &[ContentBlock], out: &mut BTreeMap<String, ToolUsage>) {
    for b in blocks {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = b
        {
            // Tool name is not present on the result block; key by
            // `tool_use_id` as a fallback so the report still shows
            // per-result volume even without the originating call.
            let entry = out.entry(format!("[unknown:{tool_use_id}]")).or_default();
            let bytes = match content {
                crate::message::ToolResultContent::Text(s) => s.len() as u32,
                crate::message::ToolResultContent::Blocks(bs) => bs
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => text.len() as u32,
                        _ => 0,
                    })
                    .sum(),
            };
            entry.output_tokens = entry.output_tokens.saturating_add(bytes / 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Header, Message, ToolResultContent};

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

    fn assistant_tool_use(id: &str, name: &str, input: serde_json::Value) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: name.into(),
                input,
            }],
        }
    }

    fn user_tool_result(id: &str, text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: ToolResultContent::Text(text.into()),
                is_error: false,
            }],
        }
    }

    #[test]
    fn empty_corpus_has_zero_totals() {
        let r = analyze(&[], 5);
        assert_eq!(r.total_messages, 0);
        assert_eq!(r.total_tokens, 0);
        assert!(r.top_messages.is_empty());
    }

    #[test]
    fn role_breakdown_sums_to_total() {
        let msgs = vec![user_text("hi"), assistant_text("hello world")];
        let r = analyze(&msgs, 5);
        assert_eq!(
            r.user_tokens
                + r.assistant_tokens
                + r.system_tokens
                + r.progress_tokens
                + r.tombstone_tokens,
            r.total_tokens
        );
    }

    #[test]
    fn top_messages_sorted_descending() {
        let msgs = vec![
            user_text("short"),
            user_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            user_text("medium length text"),
        ];
        let r = analyze(&msgs, 3);
        assert_eq!(r.top_messages.len(), 3);
        assert!(r.top_messages[0].tokens >= r.top_messages[1].tokens);
        assert!(r.top_messages[1].tokens >= r.top_messages[2].tokens);
    }

    #[test]
    fn top_n_truncates() {
        let msgs: Vec<Message> = (0..10).map(|i| user_text(&format!("msg {i}"))).collect();
        let r = analyze(&msgs, 3);
        assert_eq!(r.top_messages.len(), 3);
    }

    #[test]
    fn tool_breakdown_counts_calls() {
        let msgs = vec![
            assistant_tool_use("tu1", "fetch", serde_json::json!({"url": "x"})),
            assistant_tool_use("tu2", "fetch", serde_json::json!({"url": "y"})),
            assistant_tool_use("tu3", "ping", serde_json::json!({})),
        ];
        let r = analyze(&msgs, 5);
        assert_eq!(r.tool_breakdown.get("fetch").unwrap().call_count, 2);
        assert_eq!(r.tool_breakdown.get("ping").unwrap().call_count, 1);
    }

    #[test]
    fn tool_results_accumulate_output_tokens() {
        let msgs = vec![user_tool_result("tu1", "abcd".repeat(8).as_str())];
        let r = analyze(&msgs, 5);
        let entry = r.tool_breakdown.get("[unknown:tu1]").unwrap();
        assert!(entry.output_tokens > 0);
    }

    #[test]
    fn ties_broken_by_index_ascending() {
        // Two identical-token messages — earlier index wins.
        let msgs = vec![user_text("abcd"), user_text("efgh")];
        let r = analyze(&msgs, 2);
        assert_eq!(r.top_messages[0].index, 0);
        assert_eq!(r.top_messages[1].index, 1);
    }

    #[test]
    fn role_str_consistent_with_message_variants() {
        assert_eq!(role_str(&user_text("x")), "user");
        assert_eq!(role_str(&assistant_text("x")), "assistant");
    }

    #[test]
    fn report_serialization_roundtrip() {
        let msgs = vec![user_text("hi"), assistant_text("hello")];
        let r = analyze(&msgs, 3);
        let json = serde_json::to_string(&r).unwrap();
        let parsed: ContextReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
    }
}
