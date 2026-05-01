//! Context selection strategies (Phase 4 / Task 4.1).
//!
//! [`ContextStrategy`] picks which subset of a [`MessageStore`] to send
//! to the provider on a given turn. Built-in strategies:
//!
//! - [`SlidingWindowStrategy`] — keep the last N logical turns (a turn
//!   = one User message + the Assistant response chain, including tool
//!   results). Drops oldest turns first when over the token budget.

mod sliding_window;

use crate::compact::estimate_tokens;
use crate::message::Message;

pub use sliding_window::{count_logical_turns, SlidingWindowStrategy};

/// Strategy for selecting which messages to forward to the provider.
///
/// Implementations must be deterministic + side-effect free — they
/// only inspect the store + budget and return references in original
/// order. The QueryLoop calls `select` once per turn before building
/// the [`StreamRequest`](crate::provider::StreamRequest).
pub trait ContextStrategy: std::fmt::Debug + Send + Sync {
    /// Select messages to include. Returned slice is in original
    /// (insertion) order; the caller serializes them as the
    /// `messages` field in the provider request.
    ///
    /// `budget_tokens` is the **soft** cap; the strategy SHOULD
    /// produce a selection whose estimated tokens fit under the cap,
    /// but may slightly exceed if even one message is over budget
    /// (the caller may then reactive-compact via Phase 4 Task 4.3).
    fn select<'a>(&self, messages: &'a [Message], budget_tokens: u32) -> Vec<&'a Message>;
}

/// Pass-through strategy — sends every message in the store.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassThroughStrategy;

impl ContextStrategy for PassThroughStrategy {
    fn select<'a>(&self, messages: &'a [Message], _budget_tokens: u32) -> Vec<&'a Message> {
        messages.iter().collect()
    }
}

/// Sum the estimated tokens of `messages` using
/// [`crate::compact::estimate_tokens`].
pub fn estimate_total_tokens(messages: &[&Message]) -> u32 {
    messages
        .iter()
        .map(|m| estimate_tokens(m))
        .fold(0u32, u32::saturating_add)
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

    #[test]
    fn pass_through_returns_all() {
        let msgs = vec![user("a"), user("b"), user("c")];
        let s = PassThroughStrategy;
        let got = s.select(&msgs, 10_000);
        assert_eq!(got.len(), 3);
    }
}
