//! Micro-compaction: surgical token reduction without an LLM call
//! (claude-code parity, Tier 1).
//!
//! Strategy: replace OLD `ContentBlock::ToolResult` payloads with a
//! marker string. Preserves the message DAG, the assistant's
//! `tool_use` references, and the structural shape of the
//! conversation; surrenders only the bulky tool output bytes that
//! the model has likely already integrated into its assistant
//! response.
//!
//! Mirror of `services/compact/microCompact.ts`. Where Claude Code
//! has a more elaborate time-based config (see `timeBasedMCConfig.ts`),
//! we ship a simpler turn-age + size threshold combo. The gates are
//! tunable via [`MicrocompactConfig`].

use uuid::Uuid;

use super::estimate_tokens;
use crate::message::{ContentBlock, Message, ToolResultContent};

/// Replacement payload that goes into a cleared tool result. Mirrors
/// `TIME_BASED_MC_CLEARED_MESSAGE` from Claude Code.
pub const CLEARED_PLACEHOLDER: &str = "[Old tool result content cleared]";

/// Knobs controlling which tool results are eligible for clearing.
#[derive(Debug, Clone, Copy)]
pub struct MicrocompactConfig {
    /// Tool results that appeared **at least this many turns ago**
    /// (counted as User messages back from the most recent) are
    /// candidates. Default 3.
    pub min_age_turns: u32,
    /// Always preserve the most recent N tool results regardless of
    /// age. Default 5.
    pub preserve_last_n: u32,
    /// Only clear tool results whose estimated token cost exceeds
    /// this. Cheap results aren't worth the structural mutation.
    /// Default 100.
    pub min_tokens_per_result: u32,
}

impl Default for MicrocompactConfig {
    fn default() -> Self {
        Self {
            min_age_turns: 3,
            preserve_last_n: 5,
            min_tokens_per_result: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicrocompactResult {
    /// Number of tool result blocks whose content was replaced.
    pub cleared_count: usize,
    /// Estimated tokens freed (`pre - post`).
    pub tokens_freed: u32,
    /// Tool-use IDs whose results were cleared (useful for telemetry
    /// and post-cleanup file restoration).
    pub cleared_tool_use_ids: Vec<String>,
    /// UUIDs of messages whose blocks were modified.
    pub modified_message_uuids: Vec<Uuid>,
}

/// Apply microcompaction in-place over `messages`.
pub fn microcompact(messages: &mut [Message], config: &MicrocompactConfig) -> MicrocompactResult {
    // First, identify how many User-anchored turns separate each
    // ToolResult block from "now". We walk forward counting User
    // messages, then backwards from the end so we can compute "turns
    // ago" for each tool_result block.
    let total_turns: u32 = messages
        .iter()
        .filter(|m| matches!(m, Message::User { .. }))
        .count() as u32;

    // Pass 1: gather (msg_index, block_index, age_turns, token_cost)
    // for every ToolResult block.
    let mut candidates: Vec<(usize, usize, u32, u32, String)> = Vec::new();
    let mut user_seen: u32 = 0;
    for (mi, msg) in messages.iter().enumerate() {
        if matches!(msg, Message::User { .. }) {
            user_seen = user_seen.saturating_add(1);
        }
        if let Message::User { content, .. } = msg {
            for (bi, block) in content.iter().enumerate() {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content: result_content,
                    ..
                } = block
                {
                    // age = total_turns - user_seen_so_far + 1, where
                    // current message hadn't yet been counted...
                    // user_seen here HAS been incremented for the
                    // current User message, so age = total - user_seen.
                    let age = total_turns.saturating_sub(user_seen);
                    let cost = estimate_block_tokens(result_content);
                    candidates.push((mi, bi, age, cost, tool_use_id.clone()));
                }
            }
        }
    }

    // Pass 2: filter — drop everything within `preserve_last_n` (most
    // recent), drop ages below `min_age_turns`, drop cheap blocks.
    let total_results = candidates.len();
    let preserve_floor =
        total_results.saturating_sub(config.preserve_last_n as usize);
    let mut to_clear: Vec<(usize, usize, String, u32)> = Vec::new();
    for (idx, (mi, bi, age, cost, tu_id)) in candidates.into_iter().enumerate() {
        if idx >= preserve_floor {
            // In the preserve window — skip.
            continue;
        }
        if age < config.min_age_turns {
            continue;
        }
        if cost < config.min_tokens_per_result {
            continue;
        }
        to_clear.push((mi, bi, tu_id, cost));
    }

    // Pass 3: mutate.
    let mut cleared = 0usize;
    let mut tokens_freed: u32 = 0;
    let mut cleared_ids: Vec<String> = Vec::new();
    let mut modified_uuids: Vec<Uuid> = Vec::new();
    for (mi, bi, tu_id, cost) in to_clear {
        let placeholder_cost: u32 = estimate_tokens_in_text(CLEARED_PLACEHOLDER);
        let freed = cost.saturating_sub(placeholder_cost);
        if let Some(Message::User { content, header, .. }) = messages.get_mut(mi) {
            if let Some(ContentBlock::ToolResult { content: c, .. }) = content.get_mut(bi) {
                *c = ToolResultContent::Text(CLEARED_PLACEHOLDER.into());
            }
            cleared = cleared.saturating_add(1);
            tokens_freed = tokens_freed.saturating_add(freed);
            cleared_ids.push(tu_id);
            if !modified_uuids.contains(&header.uuid) {
                modified_uuids.push(header.uuid);
            }
        }
    }

    MicrocompactResult {
        cleared_count: cleared,
        tokens_freed,
        cleared_tool_use_ids: cleared_ids,
        modified_message_uuids: modified_uuids,
    }
}

/// Per-block token estimate matching the rule in [`super::estimate_tokens`].
fn estimate_block_tokens(content: &ToolResultContent) -> u32 {
    match content {
        ToolResultContent::Text(t) => estimate_tokens_in_text(t),
        ToolResultContent::Blocks(bs) => bs
            .iter()
            .map(|b| {
                // Wrap in a synthetic User message to reuse estimator.
                let m = Message::User {
                    header: crate::message::Header::new(),
                    content: vec![b.clone()],
                };
                estimate_tokens(&m)
            })
            .fold(0u32, u32::saturating_add),
    }
}

fn estimate_tokens_in_text(text: &str) -> u32 {
    super::estimate_text_tokens(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Header;

    fn user_with_tool_result(tu_id: &str, payload_text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tu_id.into(),
                content: ToolResultContent::Text(payload_text.into()),
                is_error: false,
            }],
        }
    }

    fn user_text(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[test]
    fn cleared_placeholder_string_is_canonical() {
        assert_eq!(CLEARED_PLACEHOLDER, "[Old tool result content cleared]");
    }

    #[test]
    fn microcompact_no_op_under_min_age() {
        // Only 1 turn of history; min_age = 3 — nothing eligible.
        let mut messages = vec![user_with_tool_result("tu_1", &"x".repeat(800))];
        let cfg = MicrocompactConfig::default();
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 0);
    }

    #[test]
    fn microcompact_preserves_last_n() {
        // 6 turns, all containing tool_result, preserve_last_n = 5.
        // Only the oldest (1 result) should be eligible.
        let mut messages: Vec<Message> = (0..6)
            .map(|i| user_with_tool_result(&format!("tu_{i}"), &"x".repeat(800)))
            .collect();
        let cfg = MicrocompactConfig {
            min_age_turns: 0,
            preserve_last_n: 5,
            min_tokens_per_result: 0,
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 1);
        assert_eq!(result.cleared_tool_use_ids, vec!["tu_0"]);
    }

    #[test]
    fn microcompact_skips_cheap_results() {
        let mut messages = vec![
            user_with_tool_result("tu_old", "a"), // tiny payload
            user_text("filler"),
            user_text("filler"),
            user_text("filler"),
            user_text("now"),
        ];
        let cfg = MicrocompactConfig {
            min_age_turns: 1,
            preserve_last_n: 0,
            min_tokens_per_result: 100, // require >= 100
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 0);
    }

    #[test]
    fn microcompact_clears_old_large_results() {
        // 5 user turns; only the first has a tool result with 500
        // tokens of payload; min_age = 2; preserve_last_n = 0 (no
        // most-recent floor — age + size are the only gates).
        let big = "x".repeat(2_000); // ~500 tokens
        let mut messages = vec![
            user_with_tool_result("tu_old", &big),
            user_text("turn 2"),
            user_text("turn 3"),
            user_text("turn 4"),
            user_text("turn 5"),
        ];
        let cfg = MicrocompactConfig {
            min_age_turns: 2,
            preserve_last_n: 0,
            min_tokens_per_result: 100,
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 1);
        assert_eq!(result.cleared_tool_use_ids, vec!["tu_old"]);
        assert!(result.tokens_freed > 100);
        // Verify the content was actually replaced.
        if let Message::User { content, .. } = &messages[0] {
            if let ContentBlock::ToolResult {
                content: ToolResultContent::Text(t),
                ..
            } = &content[0]
            {
                assert_eq!(t, CLEARED_PLACEHOLDER);
            } else {
                panic!("expected Text content");
            }
        }
    }

    #[test]
    fn microcompact_handles_blocks_payload() {
        // ToolResultContent::Blocks variant should also estimate tokens.
        let inner_blocks = ToolResultContent::Blocks(vec![ContentBlock::Text {
            text: "a".repeat(800),
        }]);
        let mut messages = vec![
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: inner_blocks,
                    is_error: false,
                }],
            },
            user_text("turn 2"),
            user_text("turn 3"),
            user_text("now"),
        ];
        let cfg = MicrocompactConfig {
            min_age_turns: 1,
            preserve_last_n: 0,
            min_tokens_per_result: 50,
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 1);
    }
}
