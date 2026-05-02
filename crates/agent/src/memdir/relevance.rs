//! Memory relevance scoring (Tier 1 / claude-code parity).
//!
//! Mirrors `memdir/findRelevantMemories.ts`. Cheap mention-based
//! scoring (no embedding model required) so memory recall can run
//! synchronously alongside every conversation turn.
//!
//! Score components (all positive contributions; sum then multiplied
//! by an age-decay factor):
//!
//! - **Token overlap** — count of distinct query tokens that appear
//!   in the memory's name + description + body.
//! - **Type bias** — the caller can hint a preferred [`MemoryType`]
//!   (e.g., "feedback first when the user is correcting").
//! - **Length penalty** — extremely long memories (>1000 lines) are
//!   slightly down-weighted because they're more likely to be
//!   transcripts than insights.
//!
//! Output is a sorted, top-N filtered list. Stable ordering on score
//! ties (lex-by-name).

use super::age::AgeBucket;
use super::Memory;
use super::MemoryType;

#[derive(Debug, Clone, Copy)]
pub struct RelevanceConfig {
    /// Maximum results to return.
    pub max_results: usize,
    /// Optional bias toward a specific memory type.
    pub prefer_type: Option<MemoryType>,
    /// Multiplier applied to memories matching `prefer_type`.
    pub type_bias_multiplier: f32,
    /// Minimum score after age decay; weaker matches are dropped.
    pub min_score: f32,
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            max_results: 5,
            prefer_type: None,
            type_bias_multiplier: 1.5,
            min_score: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredMemory<'a> {
    pub memory: &'a Memory,
    pub score: f32,
    pub matched_tokens: Vec<String>,
}

/// Find the top-N memories most relevant to `query`, given a slice of
/// loaded memories. Pure function — no I/O.
pub fn find_relevant<'a>(
    query: &str,
    memories: &'a [Memory],
    cfg: &RelevanceConfig,
) -> Vec<ScoredMemory<'a>> {
    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<ScoredMemory<'a>> = memories
        .iter()
        .map(|m| score_one(m, &query_tokens, cfg))
        .filter(|s| s.score >= cfg.min_score)
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.memory.name.cmp(&b.memory.name))
    });
    scored.truncate(cfg.max_results);
    scored
}

fn score_one<'a>(
    memory: &'a Memory,
    query_tokens: &[String],
    cfg: &RelevanceConfig,
) -> ScoredMemory<'a> {
    let haystack: String =
        format!("{}\n{}\n{}", memory.name, memory.description, memory.body).to_lowercase();
    let mut matched: Vec<String> = Vec::new();
    let mut raw = 0.0f32;
    for tok in query_tokens {
        if haystack.contains(tok) {
            matched.push(tok.clone());
            raw += 1.0;
        }
    }
    if memory.body.lines().count() > 1000 {
        raw *= 0.7;
    }
    if let Some(prefer) = cfg.prefer_type {
        if memory.kind == prefer {
            raw *= cfg.type_bias_multiplier;
        }
    }
    let bucket = AgeBucket::from_age(memory.age);
    let final_score = raw * bucket.relevance_multiplier();
    ScoredMemory {
        memory,
        score: final_score,
        matched_tokens: matched,
    }
}

/// Lowercase + split on non-alnum. Filter trivially-short stop tokens.
fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut tokens: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 3)
        .filter(|s| !STOPWORDS.contains(s))
        .map(|s| s.to_string())
        .collect();
    tokens.sort();
    tokens.dedup();
    tokens
}

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "but", "with", "this", "that", "from", "are", "was", "you", "your", "our",
    "his", "her", "its", "any", "not", "all", "can", "use", "use", "into", "what", "who", "why",
    "how", "when", "where", "have", "has", "had", "will", "would", "could", "should", "about",
];

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use super::*;

    fn mem(kind: MemoryType, name: &str, desc: &str, body: &str, age: Duration) -> Memory {
        Memory {
            kind,
            name: name.to_string(),
            description: desc.to_string(),
            body: body.to_string(),
            path: PathBuf::from(format!("/m/{name}.md")),
            age,
        }
    }

    #[test]
    fn empty_query_returns_empty() {
        let m = mem(MemoryType::User, "x", "y", "z", Duration::ZERO);
        let memories = vec![m];
        let r = find_relevant("", &memories, &RelevanceConfig::default());
        assert!(r.is_empty());
    }

    #[test]
    fn only_stopwords_returns_empty() {
        let m = mem(MemoryType::User, "x", "y", "z", Duration::ZERO);
        let memories = vec![m];
        let r = find_relevant("the and for", &memories, &RelevanceConfig::default());
        assert!(r.is_empty());
    }

    #[test]
    fn matches_in_name_or_description() {
        let m1 = mem(
            MemoryType::User,
            "Alice prefers Rust",
            "language preference",
            "",
            Duration::ZERO,
        );
        let m2 = mem(
            MemoryType::User,
            "Bob loves Python",
            "different language",
            "",
            Duration::ZERO,
        );
        let memories = vec![m1, m2];
        let r = find_relevant("rust", &memories, &RelevanceConfig::default());
        assert_eq!(r.len(), 1);
        assert!(r[0].memory.name.contains("Alice"));
    }

    #[test]
    fn prefer_type_biases_score() {
        let user_m = mem(
            MemoryType::User,
            "user-msg",
            "rust preference",
            "",
            Duration::ZERO,
        );
        let feedback_m = mem(
            MemoryType::Feedback,
            "feedback-msg",
            "rust preference",
            "",
            Duration::ZERO,
        );
        let memories = vec![user_m, feedback_m];
        let cfg = RelevanceConfig {
            prefer_type: Some(MemoryType::Feedback),
            ..Default::default()
        };
        let r = find_relevant("rust preference", &memories, &cfg);
        assert_eq!(r[0].memory.kind, MemoryType::Feedback);
    }

    #[test]
    fn age_decays_score() {
        let fresh = mem(
            MemoryType::User,
            "fresh",
            "rust language",
            "",
            Duration::from_secs(60 * 60),
        );
        let ancient = mem(
            MemoryType::User,
            "ancient",
            "rust language",
            "",
            Duration::from_secs(60 * 60 * 24 * 730), // 2 years
        );
        let memories = vec![fresh, ancient];
        // Use a permissive min_score so both pass through and we can
        // compare their scores directly.
        let cfg = RelevanceConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let r = find_relevant("rust language", &memories, &cfg);
        assert_eq!(r.len(), 2);
        assert!(r[0].score > r[1].score);
        assert_eq!(r[0].memory.name, "fresh");
    }

    #[test]
    fn max_results_truncates() {
        let memories: Vec<Memory> = (0..10)
            .map(|i| {
                mem(
                    MemoryType::User,
                    &format!("m{i}"),
                    "rust language",
                    "",
                    Duration::ZERO,
                )
            })
            .collect();
        let cfg = RelevanceConfig {
            max_results: 3,
            ..Default::default()
        };
        let r = find_relevant("rust language", &memories, &cfg);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn min_score_filters_weak_matches() {
        let m = mem(
            MemoryType::User,
            "x",
            "totally unrelated",
            "",
            Duration::from_secs(60 * 60 * 24 * 730),
        );
        let memories = vec![m];
        let cfg = RelevanceConfig {
            min_score: 10.0,
            ..Default::default()
        };
        let r = find_relevant("anything", &memories, &cfg);
        assert!(r.is_empty());
    }

    #[test]
    fn matched_tokens_reported() {
        let m = mem(
            MemoryType::User,
            "rust prefs",
            "language: rust",
            "",
            Duration::ZERO,
        );
        let memories = vec![m];
        let r = find_relevant("rust language", &memories, &RelevanceConfig::default());
        assert!(!r[0].matched_tokens.is_empty());
        assert!(
            r[0].matched_tokens.contains(&"rust".to_string())
                || r[0].matched_tokens.contains(&"language".to_string())
        );
    }
}
