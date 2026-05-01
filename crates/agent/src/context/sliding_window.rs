//! Sliding-window context strategy.
//!
//! Keeps the last `keep_turns` logical turns. A **logical turn** is one
//! User message and every following message up to (and not including)
//! the next User message — covers the standard "user asks, assistant
//! responds with tool calls and tool results" sequence.

use crate::context::ContextStrategy;
use crate::message::Message;

#[derive(Debug, Clone, Copy)]
pub struct SlidingWindowStrategy {
    keep_turns: usize,
}

impl SlidingWindowStrategy {
    pub fn new(keep_turns: usize) -> Self {
        Self {
            keep_turns: keep_turns.max(1),
        }
    }
}

impl ContextStrategy for SlidingWindowStrategy {
    fn select<'a>(&self, messages: &'a [Message], budget_tokens: u32) -> Vec<&'a Message> {
        // 1. Keep the last `keep_turns` turns by walking backwards and
        //    counting User messages.
        let total_turns = count_logical_turns(messages);
        let drop_turns = total_turns.saturating_sub(self.keep_turns);

        let mut user_seen = 0usize;
        let mut start_idx = 0usize;
        for (i, m) in messages.iter().enumerate() {
            if matches!(m, Message::User { .. }) {
                if user_seen >= drop_turns {
                    start_idx = i;
                    break;
                }
                user_seen += 1;
            }
        }
        let mut out: Vec<&Message> = messages[start_idx..].iter().collect();

        // 2. If still over budget, keep dropping from the front
        //    (oldest turn first) until we fit. We always preserve at
        //    least the most recent User message so the provider has
        //    something to respond to.
        let mut total = super::estimate_total_tokens(&out);
        while total > budget_tokens && out.len() > 1 {
            // Drop one full turn from the front: the leading User
            // message + everything until the next User message.
            let pop_until = out
                .iter()
                .skip(1)
                .position(|m| matches!(m, Message::User { .. }))
                .map(|p| p + 1) // +1 to keep the User we found
                .unwrap_or(out.len() - 1);
            out.drain(0..pop_until);
            total = super::estimate_total_tokens(&out);
        }

        out
    }
}

/// Count logical turns in `messages`. A turn = one User message and
/// the following assistant chain.
pub fn count_logical_turns(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|m| matches!(m, Message::User { .. }))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Header, Message};

    fn user(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
    fn assistant(text: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[test]
    fn count_logical_turns_basic() {
        let msgs = vec![
            user("u1"),
            assistant("a1"),
            user("u2"),
            assistant("a2"),
            user("u3"),
        ];
        assert_eq!(count_logical_turns(&msgs), 3);
    }

    #[test]
    fn keep_last_n_turns() {
        // 5 turns; keep 2 — should produce only u4/a4/u5 onwards.
        let msgs = vec![
            user("u1"),
            assistant("a1"),
            user("u2"),
            assistant("a2"),
            user("u3"),
            assistant("a3"),
            user("u4"),
            assistant("a4"),
            user("u5"),
        ];
        let s = SlidingWindowStrategy::new(2);
        let got = s.select(&msgs, 1_000_000);
        assert_eq!(got.len(), 3); // u4 + a4 + u5
        assert!(matches!(got[0], Message::User { .. }));
        if let Message::User { content, .. } = got[0] {
            if let ContentBlock::Text { text } = &content[0] {
                assert_eq!(text, "u4");
            }
        }
    }

    #[test]
    fn drops_oldest_when_over_budget() {
        // 3 large user messages; budget too small for 3.
        let big = "x".repeat(800); // ~200 tokens
        let msgs = vec![
            user(&big),
            user(&big),
            user(&big),
        ];
        let s = SlidingWindowStrategy::new(10); // keep up to 10 turns
        // budget too small for 3 turns (~600 tokens) — should drop
        // older ones until fits.
        let got = s.select(&msgs, 250);
        assert!(got.len() < 3);
        assert!(!got.is_empty());
    }

    #[test]
    fn keep_one_minimum_even_when_over_budget() {
        // Single huge message that exceeds budget on its own —
        // strategy must still return it (else provider has nothing
        // to respond to).
        let huge = "x".repeat(40_000);
        let msgs = vec![user(&huge)];
        let s = SlidingWindowStrategy::new(5);
        let got = s.select(&msgs, 100);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let s = SlidingWindowStrategy::new(5);
        let got = s.select(&[], 1000);
        assert!(got.is_empty());
    }

    #[test]
    fn keep_turns_zero_falls_back_to_one() {
        let msgs = vec![user("u1"), assistant("a1"), user("u2")];
        let s = SlidingWindowStrategy::new(0); // clamped to 1
        let got = s.select(&msgs, 1_000_000);
        // Should keep just the last turn (u2).
        assert_eq!(got.len(), 1);
    }
}
