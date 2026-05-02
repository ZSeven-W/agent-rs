//! Cross-provider effort / thinking-budget config (Tier 1 / claude-code parity).
//!
//! Maps a provider-agnostic [`EffortLevel`] to:
//!
//! - **Anthropic**: `extended_thinking.budget_tokens` for Sonnet 4.5+
//!   thinking-capable models. Higher levels reserve more tokens for
//!   the thinking phase.
//! - **OpenAI / OpenAI-compatible**: `reasoning_effort` field on the
//!   chat-completion request (`"none" | "low" | "medium" | "high"`),
//!   for o-series and o-series-compatible models.
//!
//! The host picks a level; the provider adapter picks the wire
//! representation.

use serde::{Deserialize, Serialize};

/// Effort tier. `None` is provider-default (often "off / no thinking").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EffortLevel {
    #[default]
    None,
    Low,
    Medium,
    High,
}

impl EffortLevel {
    /// OpenAI `reasoning_effort` field value.
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// Anthropic budget heuristic. Returns `None` for [`Self::None`]
    /// (skip the field entirely; provider defaults apply).
    ///
    /// Tier values are pragmatic starting points consistent with the
    /// Anthropic docs' guidance ("≥1024 tokens, tune per use case"):
    /// 1k / 4k / 16k. Hosts that want exact control should bypass
    /// [`EffortLevel`] and pass [`EffortBudget::AnthropicTokens`]
    /// directly with a custom value.
    pub fn as_anthropic_budget(self) -> Option<u32> {
        match self {
            Self::None => None,
            Self::Low => Some(1024),
            Self::Medium => Some(4096),
            Self::High => Some(16_384),
        }
    }
}

/// Wire-shape budget passed to a provider adapter. Adapters choose
/// the field with `match`; this type is the union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffortBudget {
    /// Anthropic style — explicit token budget for the extended-thinking
    /// phase.
    AnthropicTokens(u32),
    /// OpenAI style — discrete level string.
    OpenAiLevel(EffortLevel),
}

impl EffortBudget {
    pub fn from_level_for_anthropic(level: EffortLevel) -> Option<Self> {
        level.as_anthropic_budget().map(Self::AnthropicTokens)
    }

    pub fn for_openai(level: EffortLevel) -> Self {
        Self::OpenAiLevel(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_level_strings_are_stable() {
        assert_eq!(EffortLevel::None.as_openai_str(), "none");
        assert_eq!(EffortLevel::Low.as_openai_str(), "low");
        assert_eq!(EffortLevel::Medium.as_openai_str(), "medium");
        assert_eq!(EffortLevel::High.as_openai_str(), "high");
    }

    #[test]
    fn anthropic_budgets_grow_monotonically() {
        let l = EffortLevel::Low.as_anthropic_budget().unwrap();
        let m = EffortLevel::Medium.as_anthropic_budget().unwrap();
        let h = EffortLevel::High.as_anthropic_budget().unwrap();
        assert!(l < m);
        assert!(m < h);
    }

    #[test]
    fn anthropic_none_returns_none_budget() {
        assert!(EffortLevel::None.as_anthropic_budget().is_none());
    }

    #[test]
    fn from_level_for_anthropic_skips_none() {
        assert!(EffortBudget::from_level_for_anthropic(EffortLevel::None).is_none());
    }

    #[test]
    fn default_is_none() {
        assert_eq!(EffortLevel::default(), EffortLevel::None);
    }
}
