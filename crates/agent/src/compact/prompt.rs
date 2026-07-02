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

/// Final synthetic user message appended to the summarize request. The
/// conversation being summarized usually ENDS mid-task (assistant tool call →
/// tool result), which invites the model to keep working on that task instead
/// of answering the system prompt — DeepSeek models were seen emitting raw
/// DSML tool markup exactly this way. A closing user turn re-anchors the
/// request on the summarize instruction.
pub const SUMMARIZE_NOW_NUDGE: &str = "Stop working on the task above — do not call any tools. \
Respond now with the compaction output only: an <analysis> block followed by a <summary> block, \
as plain text.";

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
    #[error("response is raw tool-call markup, not a summary")]
    RawToolMarkup,
}

/// Parse a model response into an analysis + summary, LENIENTLY.
///
/// The prompt asks for `<analysis>...</analysis>` then `<summary>...</summary>`,
/// and a well-behaved model emits exactly that (extracted precisely here). But
/// models not tuned for this prompt routinely skip the `<analysis>` wrapper, or
/// emit a bare summary, or get cut off mid-tag. Compaction must not throw away a
/// usable summary over a missing wrapper — a degraded summary that preserves the
/// conversation beats a hard failure that blocks compaction. So:
///
/// - Each block is located independently (order-independent).
/// - A missing `<analysis>` ⇒ empty analysis (it's only used for diagnostics /
///   file-restoration, which degrade to a no-op).
/// - A `<summary>` opened but never closed ⇒ take everything after the tag.
/// - No usable `<summary>` at all ⇒ fall back to the whole response with any
///   `<analysis>` block stripped.
///
/// Only an effectively empty response errors (the caller already rejects a
/// blank response before reaching here).
pub fn parse_summary_response(text: &str) -> Result<ParsedSummary, ParseSummaryError> {
    let analysis = extract_block(text, ANALYSIS_OPEN, ANALYSIS_CLOSE).unwrap_or_default();

    let tagged = extract_block(text, SUMMARY_OPEN, SUMMARY_CLOSE)
        .or_else(|| {
            // `<summary>` opened but not closed (truncated): take the tail.
            text.find(SUMMARY_OPEN)
                .map(|o| text[o + SUMMARY_OPEN.len()..].trim().to_string())
        })
        .filter(|s| !s.is_empty());

    let summary = match tagged {
        Some(s) => s,
        None => {
            // The model ignored the format. A prose response still makes a
            // usable summary — but raw tool-call markup (a model continuing
            // the conversation's task instead of summarizing, e.g. DeepSeek
            // emitting `<｜DSML｜tool_calls>` as text on a tool-less request)
            // must be REJECTED: splicing it into the store would replace the
            // real transcript with garbage, unrecoverably.
            let body = strip_block(text, ANALYSIS_OPEN, ANALYSIS_CLOSE);
            let body = body.trim();
            if looks_like_tool_markup(body) {
                return Err(ParseSummaryError::RawToolMarkup);
            }
            if body.is_empty() {
                analysis.clone()
            } else {
                body.to_string()
            }
        }
    };

    if summary.is_empty() {
        return Err(ParseSummaryError::MissingSummary);
    }
    Ok(ParsedSummary { analysis, summary })
}

/// Special-token fences that mark a tool call emitted as literal text:
/// ChatML-style `<|...|>` or DeepSeek's fullwidth `<｜...｜>` (also seen
/// doubled: `<｜｜DSML｜｜tool_calls>`). A real summary never STARTS with a
/// token fence, so only the leading characters are checked — prose that
/// merely mentions such tokens later in the text still passes.
fn looks_like_tool_markup(body: &str) -> bool {
    let t = body.trim_start();
    t.starts_with("<|") || t.starts_with("<｜")
}

/// Trimmed content between the first `open` and the next `close` after it.
/// `None` if either tag is absent.
fn extract_block(text: &str, open: &str, close: &str) -> Option<String> {
    let start = text.find(open)? + open.len();
    let end = text[start..].find(close).map(|i| start + i)?;
    Some(text[start..end].trim().to_string())
}

/// `text` with the first `open..close` block (inclusive) removed.
fn strip_block(text: &str, open: &str, close: &str) -> String {
    if let Some(o) = text.find(open) {
        if let Some(rel) = text[o..].find(close) {
            let end = o + rel + close.len();
            return format!("{}{}", &text[..o], &text[end..]);
        }
    }
    text.to_string()
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
    fn fallback_rejects_raw_tool_markup() {
        // Regression: DeepSeek answered a summarize request by "calling" a
        // tool as literal DSML text. The lenient no-tags fallback must not
        // splice that into the transcript as the summary.
        let dsml = "<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name=\"Bash\">\
                    \n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>";
        assert!(matches!(
            parse_summary_response(dsml),
            Err(ParseSummaryError::RawToolMarkup)
        ));
        // ChatML-style token fences are rejected the same way.
        let chatml = "<|tool_call|>{\"name\":\"bash\"}<|end|>";
        assert!(matches!(
            parse_summary_response(chatml),
            Err(ParseSummaryError::RawToolMarkup)
        ));
        // Untagged PROSE still passes via the lenient fallback, even when it
        // mentions token fences later in the text.
        let prose = "We fixed the parser. The model kept emitting <|junk|> mid-run.";
        assert_eq!(parse_summary_response(prose).unwrap().summary, prose);
        // A tagged summary is trusted as-is (the model followed the format).
        let tagged = "<summary>Ran <|weird|> experiments; all pass.</summary>";
        assert!(parse_summary_response(tagged).is_ok());
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
        let earliest = summarization_prompt(PartialCompactDirection::EarliestHalf, None);
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
    fn parse_summary_lenient_missing_analysis_uses_empty_analysis() {
        // Models not tuned for this prompt routinely skip <analysis>.
        let response = "<summary>only summary</summary>";
        let parsed = parse_summary_response(response).unwrap();
        assert_eq!(parsed.analysis, "");
        assert_eq!(parsed.summary, "only summary");
    }

    #[test]
    fn parse_summary_lenient_no_tags_uses_whole_body() {
        // The model ignored the format entirely — don't hard-fail, summarize
        // with the raw text so compaction still reclaims context.
        let response = "We set up the project and fixed the build.";
        let parsed = parse_summary_response(response).unwrap();
        assert_eq!(parsed.summary, "We set up the project and fixed the build.");
    }

    #[test]
    fn parse_summary_lenient_only_analysis_falls_back_to_it() {
        let response = "<analysis>only analysis</analysis>";
        let parsed = parse_summary_response(response).unwrap();
        // No <summary>, and the body minus the analysis block is empty, so the
        // analysis text becomes the summary rather than erroring.
        assert_eq!(parsed.summary, "only analysis");
    }

    #[test]
    fn parse_summary_lenient_unclosed_summary_takes_tail() {
        // Truncated mid-response (e.g. token limit) — keep what we got.
        let response = "<analysis>a</analysis><summary>the tail that never closed";
        let parsed = parse_summary_response(response).unwrap();
        assert_eq!(parsed.analysis, "a");
        assert_eq!(parsed.summary, "the tail that never closed");
    }

    #[test]
    fn parse_summary_empty_analysis_still_extracts_summary() {
        let response = "<analysis></analysis><summary>real</summary>";
        let parsed = parse_summary_response(response).unwrap();
        assert_eq!(parsed.analysis, "");
        assert_eq!(parsed.summary, "real");
    }

    #[test]
    fn parse_summary_blank_response_errors() {
        // Truly nothing usable → error (the caller also guards this upstream).
        match parse_summary_response("   \n  ") {
            Err(ParseSummaryError::MissingSummary) => {}
            other => panic!("expected MissingSummary, got {other:?}"),
        }
    }
}
