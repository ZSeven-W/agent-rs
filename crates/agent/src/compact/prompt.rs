//! Summarization prompts (Phase 7+ / claude-code parity).
//!
//! Mirrors `services/compact/prompt.ts` from the Claude Code reference
//! source. The compact orchestrator hands these to the LLM as a system
//! prompt; the model produces a structured `<analysis>` + `<summary>`
//! response that [`super::summarize::compact_conversation`] then parses.

/// Preamble forbidding tool calls during compaction. Compaction always
/// runs in `maxTurns: 1` mode — a denied tool call wastes the turn.
/// Mirror of `NO_TOOLS_PREAMBLE` from Claude Code's prompt.ts.
pub const NO_TOOLS_PREAMBLE: &str = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.";

/// Tag the model uses to wrap its preliminary reasoning.
pub const ANALYSIS_OPEN: &str = "<analysis>";
pub const ANALYSIS_CLOSE: &str = "</analysis>";

/// Tag the model uses to wrap the final summary that will replace the
/// older messages.
pub const SUMMARY_OPEN: &str = "<summary>";
pub const SUMMARY_CLOSE: &str = "</summary>";

/// Direction of a partial compaction. Mirrors Claude Code's
/// `PartialCompactDirection`.
///
/// - `Full` (default) — summarize the entire prior conversation.
/// - `EarliestHalf` — summarize the older half, preserve the newer.
/// - `LatestHalf` — summarize the newer half, preserve the older.
///   Rare; used when the most-recent burst is a finished side-quest
///   that should fold into history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartialCompactDirection {
    #[default]
    Full,
    EarliestHalf,
    LatestHalf,
}

/// Build the system prompt for one compaction request. Returned text
/// is meant to be passed as [`crate::provider::StreamRequest::system`].
///
/// Components in order:
/// 1. [`NO_TOOLS_PREAMBLE`] — tool-call lockout.
/// 2. The role-and-format instructions (what to put in `<analysis>` /
///    `<summary>`, and how to optimize the summary for cache reuse).
/// 3. Optional `custom_instructions` from the caller (e.g., "Focus on
///    the test failures we just discussed").
/// 4. Direction-specific instruction when `direction != Full`.
pub fn summarization_prompt(
    direction: PartialCompactDirection,
    custom_instructions: Option<&str>,
) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(NO_TOOLS_PREAMBLE);
    s.push_str("\n\n");

    s.push_str(
        "You are summarizing the conversation above so it can fit back into the model's
context window. Produce TWO blocks, in order:

<analysis>
- A 3-7 bullet recap of WHAT decisions were made, WHAT was attempted,
  WHAT failed, WHAT succeeded.
- Mention specific file paths, function names, error messages,
  identifiers — anything a future turn will need to reference.
- Do NOT invent details. If something is unknown, say so.
</analysis>

<summary>
- A self-contained, third-person narrative summary of the conversation
  so far. Written so a fresh assistant could pick up exactly where the
  previous one left off.
- Preserve every action item, intent, constraint, and discovered fact.
- 200-1500 words. Shorter is better as long as nothing critical is
  dropped.
- Begin with one sentence stating the OVERALL goal of the session.
- Do not reference 'the conversation above' — write as if the summary
  IS the new conversation start.
- Drop trivia, small talk, repeated tool errors, and exact tool output
  bytes — keep only the conclusions.
</summary>",
    );

    if let Some(extra) = custom_instructions {
        s.push_str("\n\nAdditional caller-supplied instructions:\n");
        s.push_str(extra);
    }

    match direction {
        PartialCompactDirection::Full => {}
        PartialCompactDirection::EarliestHalf => {
            s.push_str(
                "\n\nNOTE: Compact only the EARLIEST half of the conversation. The most recent half remains verbatim — your summary will sit before it.",
            );
        }
        PartialCompactDirection::LatestHalf => {
            s.push_str(
                "\n\nNOTE: Compact only the LATEST half of the conversation. The earliest half remains verbatim — your summary will sit after it.",
            );
        }
    }

    s
}

/// Extracted analysis + summary blocks from the model's response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSummary {
    pub analysis: String,
    pub summary: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseSummaryError {
    #[error("response missing required <analysis> tag")]
    MissingAnalysis,
    #[error("response missing required <summary> tag")]
    MissingSummary,
    #[error("response has unbalanced or out-of-order tags")]
    Malformed,
}

/// Parse a model response containing `<analysis>...</analysis>` and
/// `<summary>...</summary>` blocks. Tags must each appear exactly
/// once, with `<analysis>` first.
///
/// Tolerates leading/trailing whitespace and prose before the first
/// tag (the model sometimes warms up with "Here is my analysis:" or
/// similar). Inner content is trimmed.
pub fn parse_summary_response(text: &str) -> Result<ParsedSummary, ParseSummaryError> {
    let analysis_open = text
        .find(ANALYSIS_OPEN)
        .ok_or(ParseSummaryError::MissingAnalysis)?;
    let analysis_close = text[analysis_open + ANALYSIS_OPEN.len()..]
        .find(ANALYSIS_CLOSE)
        .map(|i| analysis_open + ANALYSIS_OPEN.len() + i)
        .ok_or(ParseSummaryError::Malformed)?;
    let summary_open = text[analysis_close + ANALYSIS_CLOSE.len()..]
        .find(SUMMARY_OPEN)
        .map(|i| analysis_close + ANALYSIS_CLOSE.len() + i)
        .ok_or(ParseSummaryError::MissingSummary)?;
    let summary_close = text[summary_open + SUMMARY_OPEN.len()..]
        .find(SUMMARY_CLOSE)
        .map(|i| summary_open + SUMMARY_OPEN.len() + i)
        .ok_or(ParseSummaryError::Malformed)?;

    let analysis =
        text[analysis_open + ANALYSIS_OPEN.len()..analysis_close].trim().to_string();
    let summary = text[summary_open + SUMMARY_OPEN.len()..summary_close].trim().to_string();

    if analysis.is_empty() {
        return Err(ParseSummaryError::Malformed);
    }
    if summary.is_empty() {
        return Err(ParseSummaryError::Malformed);
    }

    Ok(ParsedSummary { analysis, summary })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tools_preamble_explicit() {
        assert!(NO_TOOLS_PREAMBLE.contains("Do NOT call any tools"));
        assert!(NO_TOOLS_PREAMBLE.contains("Tool calls will be REJECTED"));
    }

    #[test]
    fn summarization_prompt_default_direction() {
        let p = summarization_prompt(PartialCompactDirection::Full, None);
        assert!(p.contains("<analysis>"));
        assert!(p.contains("<summary>"));
        assert!(!p.contains("EARLIEST half"));
        assert!(!p.contains("LATEST half"));
    }

    #[test]
    fn summarization_prompt_with_custom_instructions() {
        let p = summarization_prompt(
            PartialCompactDirection::Full,
            Some("Focus on the test failures."),
        );
        assert!(p.contains("Focus on the test failures."));
    }

    #[test]
    fn summarization_prompt_partial_directions() {
        let earliest =
            summarization_prompt(PartialCompactDirection::EarliestHalf, None);
        assert!(earliest.contains("EARLIEST half"));
        let latest = summarization_prompt(PartialCompactDirection::LatestHalf, None);
        assert!(latest.contains("LATEST half"));
    }

    #[test]
    fn parse_summary_happy_path() {
        let response = "<analysis>
- We tried fix A.
- It failed because Y.
</analysis>
<summary>
The session aimed to fix bug X. Fix A was tried and failed because Y.
</summary>";
        let parsed = parse_summary_response(response).unwrap();
        assert!(parsed.analysis.contains("fix A"));
        assert!(parsed.summary.starts_with("The session"));
    }

    #[test]
    fn parse_summary_tolerates_prose_prefix() {
        let response = "Here is my analysis:\n\n<analysis>x</analysis>\nNow the summary:\n<summary>y</summary>";
        let parsed = parse_summary_response(response).unwrap();
        assert_eq!(parsed.analysis, "x");
        assert_eq!(parsed.summary, "y");
    }

    #[test]
    fn parse_summary_missing_analysis_errors() {
        let response = "<summary>only summary</summary>";
        match parse_summary_response(response) {
            Err(ParseSummaryError::MissingAnalysis) => {}
            other => panic!("expected MissingAnalysis, got {other:?}"),
        }
    }

    #[test]
    fn parse_summary_missing_summary_errors() {
        let response = "<analysis>only analysis</analysis>";
        match parse_summary_response(response) {
            Err(ParseSummaryError::MissingSummary) => {}
            other => panic!("expected MissingSummary, got {other:?}"),
        }
    }

    #[test]
    fn parse_summary_empty_block_errors() {
        let response = "<analysis></analysis><summary>real</summary>";
        match parse_summary_response(response) {
            Err(ParseSummaryError::Malformed) => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn parse_summary_summary_before_analysis_errors() {
        // Out-of-order: summary tag comes BEFORE analysis tag.
        let response = "<summary>s</summary><analysis>a</analysis>";
        match parse_summary_response(response) {
            // The parser searches for </analysis> AFTER <analysis>.
            // Since <summary> closes first, this presents as either
            // Malformed or MissingSummary depending on the input.
            Err(ParseSummaryError::Malformed) | Err(ParseSummaryError::MissingSummary) => {}
            other => panic!("expected Malformed or MissingSummary, got {other:?}"),
        }
    }
}
