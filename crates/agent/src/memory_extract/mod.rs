//! Background memory extraction (Tier 3 / claude-code parity).
//!
//! Mirrors `services/memdir/sessionMemory.ts`. Walks an assistant
//! message (or compaction analysis block) and pulls out candidate
//! memory entries that the host then promotes to durable
//! [`crate::memdir::Memory`] files.
//!
//! No LLM call is required — extraction is heuristic, looking for
//! the same DECISION/OBSERVATION/CONSTRAINT/OPEN_QUESTION/REFERENCE
//! prefixes that [`crate::compact::session_memory::extract_memories_from_analysis`]
//! already recognises, plus a few additional patterns:
//!
//! - `User prefers …`, `User wants …` → User memory.
//! - `Reminder:` / `Note:` followed by an instruction → Feedback.
//! - URL / file-path-bearing lines → Reference.
//!
//! Hosts that need stronger extraction can plug a real LLM-driven
//! extractor in front of this module — the candidate shape is
//! intentionally generic.

use serde::{Deserialize, Serialize};

use crate::memdir::MemoryType;

/// One extracted candidate. Hosts decide whether to promote into
/// real memory files (calling [`crate::compact::session_memory::promote_to_store`]
/// or writing through their own persistence layer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryCandidate {
    pub kind: MemoryType,
    /// Suggested 1-line description (frontmatter `description`).
    pub description: String,
    /// Body text — the substantive content of the memory.
    pub body: String,
    /// Confidence score in [0.0, 1.0]. Heuristics return 1.0 for
    /// strong-signal patterns (explicit "User prefers" / DECISION:),
    /// 0.5 for weaker matches.
    pub confidence: f32,
}

/// Run the heuristic extractor over `text` (typically an assistant
/// message or compaction analysis block). Returns extracted
/// candidates sorted by confidence descending.
pub fn extract(text: &str) -> Vec<MemoryCandidate> {
    let mut out: Vec<MemoryCandidate> = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        // Strong-signal prefixed forms (compaction-analysis style).
        if let Some(rest) = strip_prefix_ci(line, "DECISION:") {
            out.push(candidate(MemoryType::Project, "decision", rest, 1.0));
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "OBSERVATION:") {
            out.push(candidate(MemoryType::Project, "observation", rest, 1.0));
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "CONSTRAINT:") {
            out.push(candidate(MemoryType::Project, "constraint", rest, 1.0));
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "OPEN_QUESTION:")
            .or_else(|| strip_prefix_ci(line, "OPEN QUESTION:"))
        {
            out.push(candidate(MemoryType::Project, "open question", rest, 1.0));
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "REFERENCE:") {
            out.push(candidate(MemoryType::Reference, "reference", rest, 1.0));
            continue;
        }
        // Weaker-signal natural-language patterns.
        if starts_with_any_ci(line, &["user prefers ", "user wants ", "user is "]) {
            out.push(candidate(MemoryType::User, "user", line, 0.7));
            continue;
        }
        if starts_with_any_ci(line, &["reminder:", "note:", "todo:"]) {
            let rest = line.split_once(':').map(|(_, r)| r).unwrap_or(line).trim();
            out.push(candidate(MemoryType::Feedback, "reminder", rest, 0.6));
            continue;
        }
        // URL / repo-link-bearing line → Reference.
        if (line.contains("http://") || line.contains("https://") || line.contains("file://"))
            && line.len() < 400
        {
            out.push(candidate(MemoryType::Reference, "link", line, 0.5));
            continue;
        }
    }
    // Stable sort by confidence DESC, body lexicographically as
    // tiebreak — deterministic across runs.
    out.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.body.cmp(&b.body))
    });
    out
}

fn candidate(kind: MemoryType, description: &str, body: &str, confidence: f32) -> MemoryCandidate {
    MemoryCandidate {
        kind,
        description: description.to_string(),
        body: body.trim().to_string(),
        confidence,
    }
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() < prefix.len() {
        return None;
    }
    if s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(s[prefix.len()..].trim_start())
    } else {
        None
    }
}

fn starts_with_any_ci(s: &str, prefixes: &[&str]) -> bool {
    prefixes
        .iter()
        .any(|p| s.len() >= p.len() && s[..p.len()].eq_ignore_ascii_case(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_decision_with_high_confidence() {
        let out = extract("DECISION: Use Postgres instead of MySQL.");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, MemoryType::Project);
        assert!(out[0].confidence >= 0.99);
        assert!(out[0].body.contains("Postgres"));
    }

    #[test]
    fn extracts_user_preference() {
        let out = extract("User prefers terse responses without markdown headers.");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, MemoryType::User);
    }

    #[test]
    fn extracts_reminder_as_feedback() {
        let out = extract("Reminder: don't forget to run lint before commit.");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, MemoryType::Feedback);
    }

    #[test]
    fn extracts_url_as_reference() {
        let out = extract("See https://example.com/docs for details.");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, MemoryType::Reference);
    }

    #[test]
    fn empty_input_yields_no_candidates() {
        assert!(extract("").is_empty());
        assert!(extract("\n\n   \n").is_empty());
    }

    #[test]
    fn unrecognised_text_yields_nothing() {
        let out = extract("The weather is nice today.");
        assert!(out.is_empty());
    }

    #[test]
    fn results_sorted_by_confidence_desc() {
        let text = "See https://x\nDECISION: A\nReminder: do thing\n";
        let out = extract(text);
        assert!(out[0].confidence >= out[1].confidence);
        assert!(out[1].confidence >= out[2].confidence);
    }

    #[test]
    fn case_insensitive_prefix_match() {
        let out = extract("decision: lowercase still works");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, MemoryType::Project);
    }

    #[test]
    fn open_question_with_or_without_underscore() {
        for text in ["OPEN_QUESTION: Why X?", "Open question: Why X?"] {
            let out = extract(text);
            assert_eq!(out.len(), 1, "input {text}");
            assert_eq!(out[0].kind, MemoryType::Project);
        }
    }

    #[test]
    fn very_long_link_lines_skipped() {
        let body = "x".repeat(500);
        let text = format!("see https://example.com {body}");
        // Line too long → not extracted as a link.
        let out = extract(&text);
        assert!(out.is_empty());
    }

    #[test]
    fn candidate_serde_roundtrip() {
        let c = MemoryCandidate {
            kind: MemoryType::User,
            description: "x".into(),
            body: "y".into(),
            confidence: 0.7,
        };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: MemoryCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn extracts_multiple_distinct_kinds_from_one_input() {
        let text = "User prefers vim.\nDECISION: ship Friday.\nSee https://example.com\n";
        let out = extract(text);
        let kinds: std::collections::HashSet<_> = out.iter().map(|c| c.kind).collect();
        assert!(kinds.contains(&MemoryType::User));
        assert!(kinds.contains(&MemoryType::Project));
        assert!(kinds.contains(&MemoryType::Reference));
    }
}
