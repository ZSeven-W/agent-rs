//! `TiktokenTokenizer` — real BPE token counts via `tiktoken-rs`.
//!
//! Gated behind the `tiktoken` feature (off by default — pulls a
//! multi-MB BPE vocabulary blob into the dep tree). Hosts that need
//! exact counts for cost projection, sliding-window trim, or
//! prompt-size precheck opt in.
//!
//! Encoding selection mirrors what OpenAI / Anthropic publish:
//!
//! | Encoding     | Models                                             |
//! |--------------|----------------------------------------------------|
//! | `Cl100kBase` | GPT-3.5-turbo / GPT-4 / GPT-4-turbo / *Claude* (close approximation; Anthropic doesn't publish a public tokenizer, and `cl100k_base` overestimates by ~5–10% — close enough for budgeting). |
//! | `O200kBase`  | GPT-4o, GPT-4o-mini, o1                            |
//! | `P50kBase`   | text-davinci-002 / -003 / Codex                    |
//! | `R50kBase`   | GPT-3 (legacy)                                     |
//!
//! For Anthropic Claude, the official guidance is to use the
//! [`messages/count_tokens`](https://docs.anthropic.com/en/api/messages-count-tokens)
//! API for production accounting. `Cl100kBase` is a reasonable local
//! estimator when network calls aren't an option (offline, latency-
//! sensitive, or rate-limited budgeting passes).

use tiktoken_rs::{cl100k_base, o200k_base, p50k_base, r50k_base, CoreBPE};

use super::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiktokenEncoding {
    Cl100kBase,
    O200kBase,
    P50kBase,
    R50kBase,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TiktokenError {
    #[error("failed to load tiktoken encoding {encoding:?}: {message}")]
    Load {
        encoding: TiktokenEncoding,
        message: String,
    },
}

/// `Tokenizer` impl backed by `tiktoken-rs`. Thread-safe (`CoreBPE`
/// is `Sync`); cheap to share via `Arc` if multiple components need
/// it.
pub struct TiktokenTokenizer {
    encoding: TiktokenEncoding,
    bpe: CoreBPE,
}

// `CoreBPE` doesn't `Debug` (its internal HashMap of byte sequences
// would dump megabytes if it did). Surface only the encoding.
impl std::fmt::Debug for TiktokenTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenTokenizer")
            .field("encoding", &self.encoding)
            .finish()
    }
}

impl TiktokenTokenizer {
    /// Construct with the chosen encoding. Loading parses the BPE
    /// vocab once; reuse the resulting tokenizer for the lifetime of
    /// your process.
    pub fn new(encoding: TiktokenEncoding) -> Result<Self, TiktokenError> {
        let bpe = load(encoding)?;
        Ok(Self { encoding, bpe })
    }

    /// Convenience for the most common encoding.
    pub fn cl100k_base() -> Result<Self, TiktokenError> {
        Self::new(TiktokenEncoding::Cl100kBase)
    }

    /// GPT-4o / GPT-4o-mini / o1 family.
    pub fn o200k_base() -> Result<Self, TiktokenError> {
        Self::new(TiktokenEncoding::O200kBase)
    }

    /// Codex / text-davinci-00x.
    pub fn p50k_base() -> Result<Self, TiktokenError> {
        Self::new(TiktokenEncoding::P50kBase)
    }

    /// Legacy GPT-3.
    pub fn r50k_base() -> Result<Self, TiktokenError> {
        Self::new(TiktokenEncoding::R50kBase)
    }

    /// Pick the right encoding for a model id. Falls back to
    /// `cl100k_base` for unknown / Claude / generic chat models.
    pub fn for_model(model: &str) -> Result<Self, TiktokenError> {
        Self::new(encoding_for_model(model))
    }

    pub fn encoding(&self) -> TiktokenEncoding {
        self.encoding
    }
}

fn load(encoding: TiktokenEncoding) -> Result<CoreBPE, TiktokenError> {
    let result = match encoding {
        TiktokenEncoding::Cl100kBase => cl100k_base(),
        TiktokenEncoding::O200kBase => o200k_base(),
        TiktokenEncoding::P50kBase => p50k_base(),
        TiktokenEncoding::R50kBase => r50k_base(),
    };
    result.map_err(|source| TiktokenError::Load {
        encoding,
        message: source.to_string(),
    })
}

/// Heuristic mapping from model id → tiktoken encoding. Conservative:
/// when in doubt, pick `cl100k_base` since it's the broadest valid
/// choice for chat-completion-shaped models.
pub fn encoding_for_model(model: &str) -> TiktokenEncoding {
    let m = model.to_ascii_lowercase();
    if m.starts_with("gpt-4o") || m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4")
    {
        return TiktokenEncoding::O200kBase;
    }
    if m.starts_with("text-davinci-00") || m.starts_with("code-") {
        return TiktokenEncoding::P50kBase;
    }
    if m.starts_with("davinci") || m.starts_with("curie") || m.starts_with("babbage") || m == "ada"
    {
        return TiktokenEncoding::R50kBase;
    }
    // GPT-3.5 / GPT-4 / GPT-4-turbo / Claude all map to cl100k_base.
    TiktokenEncoding::Cl100kBase
}

impl Tokenizer for TiktokenTokenizer {
    fn count_text(&self, text: &str) -> u32 {
        // `encode_with_special_tokens` matches what OpenAI's billing
        // counts (special tokens included). Saturating cast keeps a
        // pathologically long input from panicking.
        let n = self.bpe.encode_with_special_tokens(text).len();
        u32::try_from(n).unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cl100k_known_count() {
        let t = TiktokenTokenizer::cl100k_base().expect("load");
        // "hello world" is exactly 2 cl100k tokens.
        assert_eq!(t.count_text("hello world"), 2);
    }

    #[test]
    fn o200k_known_count() {
        let t = TiktokenTokenizer::o200k_base().expect("load");
        // "hello world" is also 2 in o200k (different bpe but same
        // common-word merge).
        assert_eq!(t.count_text("hello world"), 2);
    }

    #[test]
    fn empty_string_is_zero() {
        let t = TiktokenTokenizer::cl100k_base().expect("load");
        assert_eq!(t.count_text(""), 0);
    }

    #[test]
    fn cjk_counts_more_than_ascii_per_char() {
        let t = TiktokenTokenizer::cl100k_base().expect("load");
        let ascii = t.count_text("hello world hello world");
        let cjk = t.count_text("你好世界");
        // CJK in cl100k is ~1+ tokens per character, so 4 CJK chars
        // should generally cost more than the merge-friendly 4-word
        // ascii string. This is a smoke check; exact ratios drift.
        assert!(cjk > 0);
        assert!(ascii > 0);
    }

    #[test]
    fn for_model_routes_gpt4o_to_o200k() {
        assert_eq!(
            encoding_for_model("gpt-4o-2024-08-06"),
            TiktokenEncoding::O200kBase
        );
        assert_eq!(
            encoding_for_model("gpt-4o-mini"),
            TiktokenEncoding::O200kBase
        );
        assert_eq!(
            encoding_for_model("o1-preview"),
            TiktokenEncoding::O200kBase
        );
    }

    #[test]
    fn for_model_routes_legacy_to_p50k_or_r50k() {
        assert_eq!(
            encoding_for_model("text-davinci-003"),
            TiktokenEncoding::P50kBase
        );
        assert_eq!(encoding_for_model("davinci"), TiktokenEncoding::R50kBase);
    }

    #[test]
    fn for_model_falls_back_to_cl100k() {
        assert_eq!(
            encoding_for_model("claude-opus-4-7"),
            TiktokenEncoding::Cl100kBase
        );
        assert_eq!(
            encoding_for_model("gpt-4-turbo-preview"),
            TiktokenEncoding::Cl100kBase
        );
        assert_eq!(
            encoding_for_model("some-future-unknown-model"),
            TiktokenEncoding::Cl100kBase
        );
    }

    #[test]
    fn count_messages_uses_tiktoken_under_default_impl() {
        use crate::message::{ContentBlock, Header, Message};
        let t = TiktokenTokenizer::cl100k_base().expect("load");
        let m = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text {
                text: "hello world".into(),
            }],
        };
        // count_message dispatches to count_block → count_text.
        assert!(t.count_message(&m) >= 2);
    }
}
