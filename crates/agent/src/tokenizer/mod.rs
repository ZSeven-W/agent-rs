//! Tokenizer (Tier 4 / claude-code parity).
//!
//! Mirrors `services/tokens/`. Production-grade token counting needs
//! a true tokenizer (tiktoken / sentencepiece / etc.), but pulling
//! one in pulls a multi-MB binary blob into the dep tree. This
//! module provides:
//!
//! - [`Tokenizer`] trait — the surface every estimator implements.
//! - [`HeuristicTokenizer`] — the existing 4-ASCII / 1-CJK estimator
//!   from [`crate::compact::estimate_text_tokens`], wrapped as a
//!   trait impl. Default for hosts that don't need exact counts.
//! - [`WordSplitTokenizer`] — alternative implementation closer to
//!   real BPE behaviour for English: splits on whitespace +
//!   punctuation, counts subwords by simple length-based rules.
//!   No fixtures; deterministic + fast.
//!
//! Real tiktoken-style tokenizers are pluggable: implement
//! [`Tokenizer`] and pass it to anywhere that takes a `&dyn Tokenizer`.

use crate::message::{ContentBlock, Message, ToolResultContent};

/// Pluggable tokenizer surface.
pub trait Tokenizer: std::fmt::Debug + Send + Sync {
    fn count_text(&self, text: &str) -> u32;

    fn count_message(&self, msg: &Message) -> u32 {
        match msg {
            Message::User { content, .. } | Message::Assistant { content, .. } => content
                .iter()
                .map(|b| self.count_block(b))
                .fold(0, u32::saturating_add),
            Message::System { text, .. }
            | Message::Progress { note: text, .. }
            | Message::Tombstone { reason: text, .. } => self.count_text(text).saturating_add(2),
        }
    }

    fn count_block(&self, block: &ContentBlock) -> u32 {
        match block {
            ContentBlock::Text { text } => self.count_text(text),
            ContentBlock::Image { source } => match source {
                crate::message::ImageSource::Base64 { data, .. } => {
                    ((data.len() as u32) / 4).saturating_add(8)
                }
                crate::message::ImageSource::Url { url } => self.count_text(url).saturating_add(8),
                crate::message::ImageSource::File { file_id } => {
                    self.count_text(file_id).saturating_add(8)
                }
            },
            ContentBlock::Document { source } => match source {
                crate::message::DocumentSource::Base64 { data, .. } => {
                    ((data.len() as u32) / 4).saturating_add(16)
                }
                crate::message::DocumentSource::Url { url } => {
                    self.count_text(url).saturating_add(16)
                }
                crate::message::DocumentSource::File { file_id } => {
                    self.count_text(file_id).saturating_add(16)
                }
            },
            ContentBlock::ToolUse { id, name, input } => {
                let json_len = serde_json::to_string(input).map(|s| s.len()).unwrap_or(0);
                self.count_text(id)
                    .saturating_add(self.count_text(name))
                    .saturating_add((json_len as u32) / 4)
                    .saturating_add(4)
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                let inner = match content {
                    ToolResultContent::Text(t) => self.count_text(t),
                    ToolResultContent::Blocks(bs) => bs
                        .iter()
                        .map(|b| self.count_block(b))
                        .fold(0, u32::saturating_add),
                };
                self.count_text(tool_use_id)
                    .saturating_add(inner)
                    .saturating_add(4)
            }
            ContentBlock::Thinking { thinking, .. } => self.count_text(thinking).saturating_add(2),
        }
    }

    fn count_messages(&self, msgs: &[Message]) -> u32 {
        msgs.iter()
            .map(|m| self.count_message(m))
            .fold(0, u32::saturating_add)
    }
}

/// Default — wraps the existing `compact::estimate_text_tokens`
/// (4-ASCII / 1-CJK rule).
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicTokenizer;

impl Tokenizer for HeuristicTokenizer {
    fn count_text(&self, text: &str) -> u32 {
        crate::compact::estimate_text_tokens(text)
    }
}

/// Word-and-punctuation splitting tokenizer. Closer to real BPE for
/// English at the cost of ASCII-only accuracy. Counts each
/// alphanumeric run as 1 token if ≤8 chars, else `len / 4` tokens
/// (mimicking how BPE breaks long words). Punctuation runs each
/// count as 1 token. CJK chars count 1 each.
#[derive(Debug, Clone, Copy, Default)]
pub struct WordSplitTokenizer;

impl Tokenizer for WordSplitTokenizer {
    fn count_text(&self, text: &str) -> u32 {
        let mut total: u32 = 0;
        let mut alnum_run = 0u32;
        for ch in text.chars() {
            if is_cjk(ch) {
                if alnum_run > 0 {
                    total = total.saturating_add(token_for_run(alnum_run));
                    alnum_run = 0;
                }
                total = total.saturating_add(1);
            } else if ch.is_alphanumeric() {
                alnum_run = alnum_run.saturating_add(1);
            } else if ch.is_whitespace() {
                if alnum_run > 0 {
                    total = total.saturating_add(token_for_run(alnum_run));
                    alnum_run = 0;
                }
            } else {
                if alnum_run > 0 {
                    total = total.saturating_add(token_for_run(alnum_run));
                    alnum_run = 0;
                }
                // Each punctuation char = 1 token.
                total = total.saturating_add(1);
            }
        }
        if alnum_run > 0 {
            total = total.saturating_add(token_for_run(alnum_run));
        }
        total
    }
}

fn token_for_run(len: u32) -> u32 {
    if len <= 8 {
        1
    } else {
        len.div_ceil(4)
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3040..=0x30FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xAC00..=0xD7AF
        | 0xF900..=0xFAFF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
    )
}

#[cfg(test)]
mod tests {
    use crate::message::{ContentBlock, Header, Message};

    use super::*;

    fn user(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[test]
    fn heuristic_tokenizer_matches_legacy_estimator() {
        let t = HeuristicTokenizer;
        assert_eq!(t.count_text("abcdefgh"), 2);
        assert_eq!(t.count_text("你好"), 2);
    }

    #[test]
    fn word_split_short_words_are_one_token() {
        let t = WordSplitTokenizer;
        assert_eq!(t.count_text("hello world"), 2);
        assert_eq!(t.count_text("hi"), 1);
    }

    #[test]
    fn word_split_long_words_split_into_subwords() {
        let t = WordSplitTokenizer;
        // 16-char word → ceil(16/4) = 4 subword tokens.
        assert_eq!(t.count_text("aaaaaaaaaaaaaaaa"), 4);
    }

    #[test]
    fn word_split_punctuation_counts_each() {
        let t = WordSplitTokenizer;
        // "hi!" → "hi" (1) + "!" (1) = 2.
        assert_eq!(t.count_text("hi!"), 2);
        // "hi !!" → "hi" (1) + "!" (1) + "!" (1) = 3
        assert_eq!(t.count_text("hi !!"), 3);
    }

    #[test]
    fn word_split_cjk_one_per_char() {
        let t = WordSplitTokenizer;
        assert_eq!(t.count_text("你好世界"), 4);
    }

    #[test]
    fn count_message_dispatches_to_blocks() {
        let t = HeuristicTokenizer;
        let m = user("hello world!");
        assert!(t.count_message(&m) > 0);
    }

    #[test]
    fn count_messages_sums_individuals() {
        let t = HeuristicTokenizer;
        let msgs = vec![user("hi"), user("there")];
        let total = t.count_messages(&msgs);
        let sum: u32 = msgs.iter().map(|m| t.count_message(m)).sum();
        assert_eq!(total, sum);
    }

    #[test]
    fn empty_text_zero_tokens() {
        let t = WordSplitTokenizer;
        assert_eq!(t.count_text(""), 0);
        assert_eq!(t.count_text("   "), 0);
    }

    #[test]
    fn word_split_mixed_runs() {
        let t = WordSplitTokenizer;
        // "Hello, World! 你好"
        // → "Hello"(1) "," (1) "World" (1) "!" (1) "你"(1) "好"(1) = 6
        assert_eq!(t.count_text("Hello, World! 你好"), 6);
    }

    #[test]
    fn token_for_run_boundary() {
        assert_eq!(token_for_run(1), 1);
        assert_eq!(token_for_run(8), 1);
        assert_eq!(token_for_run(9), 3); // ceil(9/4)
        assert_eq!(token_for_run(12), 3);
        assert_eq!(token_for_run(16), 4);
    }
}
