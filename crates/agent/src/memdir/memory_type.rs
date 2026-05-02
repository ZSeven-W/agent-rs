//! Memory taxonomy (Tier 1 / claude-code parity).
//!
//! Mirrors `memdir/memoryTypes.ts`. Four types capture the kinds of
//! information worth persisting across conversations:
//!
//! - **User** — durable facts about the human collaborator (role,
//!   expertise, preferences, environment).
//! - **Feedback** — guidance the user has given on how to approach
//!   work (do this / don't do that), useful for steering future
//!   suggestions.
//! - **Project** — non-code, non-git context about ongoing work
//!   (deadlines, why decisions were made, who is doing what).
//! - **Reference** — pointers to external systems (Linear projects,
//!   Slack channels, dashboards) and how to use them.
//!
//! See the auto-memory section in `CLAUDE.md` for the detailed scope
//! rules each type carries.

use serde::{Deserialize, Serialize};

/// One of the four memory categories. New variants would constitute a
/// SemVer break — `#[non_exhaustive]` keeps the door open for future
/// types (e.g. "incident", "personal-pref").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemoryType {
    User,
    Feedback,
    Project,
    Reference,
}

impl MemoryType {
    /// Stable lowercase string used in YAML frontmatter `type:` and
    /// in JSON serialization (`#[serde(rename_all = "snake_case")]`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Feedback => "feedback",
            Self::Project => "project",
            Self::Reference => "reference",
        }
    }

    pub fn from_str_ci(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "user" => Some(Self::User),
            "feedback" => Some(Self::Feedback),
            "project" => Some(Self::Project),
            "reference" => Some(Self::Reference),
            _ => None,
        }
    }

    /// Human-readable scope description — what this type is FOR.
    /// Surfaced to the model so it can pick the right kind when
    /// asked to remember something.
    pub fn description(self) -> &'static str {
        match self {
            Self::User => {
                "durable facts about the human (role, expertise, preferences, environment)"
            }
            Self::Feedback => {
                "guidance from the user about how to approach work — corrections + confirmations"
            }
            Self::Project => {
                "ongoing-work context not derivable from code or git (deadlines, motivation, owners)"
            }
            Self::Reference => "pointers to external systems (Linear, Slack, Grafana) and how to use them",
        }
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Things a memory entry MUST NOT contain. Code patterns / file paths
/// / git history are out — they're derivable from the project state.
/// Validates the `body` text + frontmatter `name`/`description`. The
/// caller decides whether a violation is fatal (reject) or advisory
/// (log + accept).
pub fn validate_body(body: &str) -> Vec<ValidationWarning> {
    let mut out = Vec::new();
    let lower = body.to_lowercase();
    if lower.contains("```") && body.matches("```").count() >= 2 {
        out.push(ValidationWarning::ContainsCodeBlock);
    }
    if body.lines().count() > 200 {
        out.push(ValidationWarning::TooLong);
    }
    if body.trim().is_empty() {
        out.push(ValidationWarning::Empty);
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationWarning {
    /// Body has fenced code blocks — typically signals stored code,
    /// which the auto-memory rules forbid.
    ContainsCodeBlock,
    /// Body is over 200 lines — likely a copy-pasted log instead of
    /// a memory.
    TooLong,
    /// Body has no non-whitespace content.
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_strings() {
        for t in [
            MemoryType::User,
            MemoryType::Feedback,
            MemoryType::Project,
            MemoryType::Reference,
        ] {
            assert_eq!(MemoryType::from_str_ci(t.as_str()), Some(t));
            assert_eq!(MemoryType::from_str_ci(&t.as_str().to_uppercase()), Some(t));
        }
    }

    #[test]
    fn from_str_ci_unknown_returns_none() {
        assert!(MemoryType::from_str_ci("cookie").is_none());
        assert!(MemoryType::from_str_ci("").is_none());
    }

    #[test]
    fn validate_empty_body_warns() {
        let w = validate_body("   \n\n\t");
        assert!(w.contains(&ValidationWarning::Empty));
    }

    #[test]
    fn validate_long_body_warns() {
        let body: String = "line\n".repeat(220);
        assert!(validate_body(&body).contains(&ValidationWarning::TooLong));
    }

    #[test]
    fn validate_code_block_warns() {
        let body = "Here is some code:\n```\nfn main() {}\n```\nthat's it.";
        assert!(validate_body(body).contains(&ValidationWarning::ContainsCodeBlock));
    }

    #[test]
    fn validate_clean_body_no_warnings() {
        let body = "Short, plain prose memory entry.";
        assert!(validate_body(body).is_empty());
    }

    #[test]
    fn description_is_non_empty() {
        for t in [
            MemoryType::User,
            MemoryType::Feedback,
            MemoryType::Project,
            MemoryType::Reference,
        ] {
            assert!(!t.description().is_empty());
        }
    }
}
