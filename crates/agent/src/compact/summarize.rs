//! LLM-driven conversation summarization (claude-code parity, Tier 1).
//!
//! Calls the configured [`Provider`] with a special compaction system
//! prompt, collects the streamed text, parses
//! `<analysis>...</analysis><summary>...</summary>`, and builds a
//! [`CompactionResult`] that the caller can splice back into the
//! [`MessageStore`].

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use thiserror::Error;
use uuid::Uuid;

use super::{
    estimate_tokens,
    prompt::{
        parse_summary_response, summarization_prompt, ParseSummaryError, ParsedSummary,
        PartialCompactDirection,
    },
};
use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, Header, Message, MessageStore};
use crate::provider::{Provider, StreamRequest};
use crate::stream::Event;

/// Sentinel text the boundary marker carries — same string as
/// [`super::insert_boundary_marker`] uses.
pub const COMPACT_BOUNDARY_TEXT: &str = "CONTEXT SUMMARY BELOW";

/// Reserved-tokens budget Claude Code keeps for the compaction
/// response itself (p99.99 of compact summary output is ~17K tokens).
pub const MAX_OUTPUT_TOKENS_FOR_SUMMARY: u32 = 20_000;

#[derive(Debug, Error)]
pub enum CompactError {
    #[error("compact: not enough messages to compact")]
    NotEnoughMessages,
    #[error("compact: provider error: {0}")]
    Provider(#[from] AgentError),
    #[error("compact: parse error: {0}")]
    Parse(#[from] ParseSummaryError),
    #[error("compact: aborted")]
    Aborted,
    #[error("compact: provider stream produced no text")]
    EmptyResponse,
}

/// Outcome of a successful [`compact_conversation`] call.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The boundary `Message::System` carrying [`COMPACT_BOUNDARY_TEXT`].
    pub boundary_message: Message,
    /// The `Message::System` carrying the LLM-produced summary.
    pub summary_message: Message,
    /// `<analysis>` block from the model — typically NOT pushed back
    /// into the message store; useful for diagnostics, telemetry,
    /// and the post-cleanup file-restoration step.
    pub analysis: String,
    /// Full `<summary>` text. Equal to `summary_message` text.
    pub summary: String,
    /// UUIDs of the messages this compaction is meant to replace. The
    /// caller decides whether to delete + tombstone them, or simply
    /// surface the summary alongside.
    pub replaced_uuids: Vec<Uuid>,
    /// Estimated token total of the messages BEFORE compaction.
    pub pre_compact_tokens: u32,
    /// Estimated token total of `(boundary + summary)` AFTER.
    pub post_compact_tokens: u32,
    /// Direction of compaction performed.
    pub direction: PartialCompactDirection,
}

/// Compact `messages` into a single summary System message.
///
/// **Behavior**:
/// 1. Validate at least 2 messages exist (single message is not
///    compactable).
/// 2. Compute pre_compact tokens via [`estimate_tokens`].
/// 3. Build a [`StreamRequest`] with `messages` as the body and
///    [`summarization_prompt`] as the system. `max_output_tokens` is
///    capped by [`MAX_OUTPUT_TOKENS_FOR_SUMMARY`].
/// 4. Call `provider.stream` and accumulate every `TextDelta` into a
///    single response string. Aborted streams short-circuit to
///    [`CompactError::Aborted`]. Stream-level provider errors bubble
///    up as [`CompactError::Provider`].
/// 5. Parse the response via
///    [`super::prompt::parse_summary_response`].
/// 6. Build the [`CompactionResult`] with boundary +  summary
///    Message::System nodes and the parsed analysis text.
///
/// The caller is responsible for splicing the result back into a
/// [`MessageStore`] (e.g., via [`apply_compaction_to_store`]).
pub async fn compact_conversation(
    messages: &[Message],
    provider: &dyn Provider,
    model: impl Into<String>,
    custom_instructions: Option<&str>,
    direction: PartialCompactDirection,
    abort: AbortController,
) -> Result<CompactionResult, CompactError> {
    if messages.len() < 2 {
        return Err(CompactError::NotEnoughMessages);
    }

    let pre_compact_tokens: u32 = messages
        .iter()
        .map(estimate_tokens)
        .fold(0u32, u32::saturating_add);

    let system = summarization_prompt(direction, custom_instructions);

    // Slice into compact-target depending on direction. For Full, send
    // every message. For EarliestHalf, send the first half. For
    // LatestHalf, send the second half.
    let mid = messages.len() / 2;
    let (target_slice, replaced_uuids): (Vec<Message>, Vec<Uuid>) = match direction {
        PartialCompactDirection::Full => (
            messages.to_vec(),
            messages.iter().map(|m| m.uuid()).collect(),
        ),
        PartialCompactDirection::EarliestHalf => (
            messages[..mid].to_vec(),
            messages[..mid].iter().map(|m| m.uuid()).collect(),
        ),
        PartialCompactDirection::LatestHalf => (
            messages[mid..].to_vec(),
            messages[mid..].iter().map(|m| m.uuid()).collect(),
        ),
    };

    let req = StreamRequest::new(model.into(), target_slice)
        .with_system(system)
        .with_max_output_tokens(MAX_OUTPUT_TOKENS_FOR_SUMMARY);

    let mut stream = provider
        .stream(req, abort.clone())
        .await
        .map_err(CompactError::Provider)?;

    let collected = Arc::new(Mutex::new(String::new()));
    let mut emitted_error: Option<AgentError> = None;
    while let Some(item) = stream.next().await {
        if abort.is_aborted() {
            return Err(CompactError::Aborted);
        }
        match item {
            Ok(Event::TextDelta { delta }) => {
                if let Ok(mut buf) = collected.lock() {
                    buf.push_str(&delta);
                }
            }
            Ok(Event::Result { .. }) => {
                // Stream completed; loop will end at next None.
            }
            Ok(Event::Error { code, message }) => {
                emitted_error = Some(AgentError::provider(
                    "compact",
                    format!("stream error code={code}: {message}"),
                ));
                break;
            }
            Ok(_) => {
                // Other events (Usage, Thinking, ToolUse) — ignore for
                // compaction. NO_TOOLS_PREAMBLE should prevent
                // ToolUse; if one slips through we drop it.
            }
            Err(e) => {
                emitted_error = Some(e);
                break;
            }
        }
    }

    if let Some(err) = emitted_error {
        return Err(CompactError::Provider(err));
    }

    let response_text = collected.lock().map(|b| b.clone()).unwrap_or_default();
    if response_text.trim().is_empty() {
        return Err(CompactError::EmptyResponse);
    }

    let ParsedSummary { analysis, summary } = parse_summary_response(&response_text)?;

    let boundary_message = Message::System {
        header: Header::new(),
        text: COMPACT_BOUNDARY_TEXT.into(),
    };
    // Render the summary as a User message so providers that drop
    // System messages from the body (notably Anthropic, which renders
    // System content only via the request-level `system` parameter)
    // still see the compaction summary in the conversation. Prefixed
    // with a marker so consumers and the model can recognize it as
    // synthesized context rather than literal user input.
    let summary_message = Message::User {
        header: Header::new(),
        content: vec![ContentBlock::Text {
            text: format!("[Context summary]\n{summary}"),
        }],
    };
    let post_compact_tokens =
        estimate_tokens(&boundary_message).saturating_add(estimate_tokens(&summary_message));

    Ok(CompactionResult {
        boundary_message,
        summary_message,
        analysis,
        summary,
        replaced_uuids,
        pre_compact_tokens,
        post_compact_tokens,
        direction,
    })
}

/// Apply a [`CompactionResult`] to a [`MessageStore`]: tombstone every
/// `replaced_uuids` (so child messages still resolve through the DAG)
/// and append boundary + summary system messages at the end.
///
/// Tombstones are non-destructive — the original UUIDs are preserved
/// so `parent_uuid` references in unrelated branches remain valid.
/// Future renderers / replay tools can recognize tombstoned messages
/// via [`Message::Tombstone`].
pub fn apply_compaction_to_store(
    store: &mut MessageStore,
    result: &CompactionResult,
) -> Result<(), AgentError> {
    // Snapshot existing messages, tombstone replaced UUIDs in place,
    // and insert boundary + summary AT the compaction boundary so the
    // chronological order makes sense to the model:
    //
    // - Full          → append at end (everything is replaced).
    // - EarliestHalf  → insert AFTER the last replaced message
    //                   (i.e., right BEFORE the first preserved).
    // - LatestHalf    → insert BEFORE the first replaced message
    //                   (i.e., right AFTER the last preserved).
    //
    // This keeps the synthetic summary chronologically next to the
    // window it summarizes, instead of sitting at the end and
    // appearing to come after the user's most recent input.
    let snapshot: Vec<Message> = store.iter().cloned().collect();
    let total = snapshot.len();
    let insert_at: usize = match result.direction {
        PartialCompactDirection::Full => total,
        PartialCompactDirection::EarliestHalf => snapshot
            .iter()
            .position(|m| !result.replaced_uuids.contains(&m.uuid()))
            .unwrap_or(total),
        PartialCompactDirection::LatestHalf => snapshot
            .iter()
            .position(|m| result.replaced_uuids.contains(&m.uuid()))
            .unwrap_or(total),
    };

    let mut new_store = MessageStore::new();
    for (i, msg) in snapshot.into_iter().enumerate() {
        if i == insert_at {
            new_store.push(result.boundary_message.clone())?;
            new_store.push(result.summary_message.clone())?;
        }
        let to_push = if result.replaced_uuids.contains(&msg.uuid()) {
            let header = msg.header().clone();
            Message::Tombstone {
                header,
                reason: "compacted".into(),
            }
        } else {
            msg
        };
        new_store.push(to_push)?;
    }
    if insert_at == total {
        new_store.push(result.boundary_message.clone())?;
        new_store.push(result.summary_message.clone())?;
    }
    *store = new_store;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use futures::stream;

    use super::*;
    use crate::message::ContentBlock;
    use crate::provider::{ProviderCapabilities, StreamRequest};
    use crate::stream::EventStream;

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

    #[derive(Debug)]
    struct ScriptedProvider {
        events: StdMutex<Vec<Event>>,
    }

    impl ScriptedProvider {
        fn new(events: Vec<Event>) -> Self {
            Self {
                events: StdMutex::new(events),
            }
        }
        fn from_text(text: &str) -> Self {
            Self::new(vec![
                Event::TextDelta { delta: text.into() },
                Event::Result {
                    data: Default::default(),
                },
            ])
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        fn id(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        async fn stream(
            &self,
            _req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            let events: Vec<Event> = self
                .events
                .lock()
                .map(|mut g| std::mem::take(&mut *g))
                .unwrap();
            Ok(Box::new(stream::iter(events.into_iter().map(Ok))))
        }
    }

    fn happy_response() -> &'static str {
        "<analysis>
- User asked X
- Assistant did Y
</analysis>
<summary>
The session was about X. Y happened.
</summary>"
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_conversation_happy_path() {
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("hi"), assistant("hello"), user("ok bye")];
        let result = compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .unwrap();
        assert!(result.analysis.contains("User asked X"));
        assert!(result.summary.starts_with("The session was about X"));
        assert_eq!(result.replaced_uuids.len(), 3);
        assert!(result.pre_compact_tokens > 0);
        assert!(result.post_compact_tokens > 0);
        // Boundary text is the canonical CONTEXT SUMMARY BELOW.
        match &result.boundary_message {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected System boundary"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_too_few_messages() {
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("only one")];
        match compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        {
            Err(CompactError::NotEnoughMessages) => {}
            other => panic!("expected NotEnoughMessages, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_empty_response_errors() {
        let provider = ScriptedProvider::from_text("");
        let messages = vec![user("a"), user("b")];
        match compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        {
            Err(CompactError::EmptyResponse) => {}
            other => panic!("expected EmptyResponse, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_tagless_response_is_summarized_leniently() {
        // A model that ignores the <analysis>/<summary> format must NOT abort
        // compaction — the plain text becomes the summary so context is still
        // reclaimed (regression guard for the deepseek `/compact` failure).
        let provider = ScriptedProvider::from_text("Just plain text, no tags here.");
        let messages = vec![user("a"), user("b")];
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .expect("tagless response should compact leniently, not error");
        assert_eq!(result.summary, "Just plain text, no tags here.");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_blank_response_still_errors() {
        // A truly empty response is a real failure (nothing to summarize).
        let provider = ScriptedProvider::from_text("   ");
        let messages = vec![user("a"), user("b")];
        assert!(matches!(
            compact_conversation(
                &messages,
                &provider,
                "m",
                None,
                PartialCompactDirection::Full,
                AbortController::new(),
            )
            .await,
            Err(CompactError::EmptyResponse)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_partial_earliest_half_keeps_recent_uuids_intact() {
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user("first");
        let m2 = user("second");
        let m3 = user("third");
        let m4 = user("fourth");
        let messages = vec![m1.clone(), m2.clone(), m3.clone(), m4.clone()];
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::EarliestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        // Earliest half = first 2; replaced_uuids should only include those.
        assert_eq!(result.replaced_uuids.len(), 2);
        assert!(result.replaced_uuids.contains(&m1.uuid()));
        assert!(result.replaced_uuids.contains(&m2.uuid()));
        assert!(!result.replaced_uuids.contains(&m3.uuid()));
        assert!(!result.replaced_uuids.contains(&m4.uuid()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_compaction_tombstones_replaced_messages() {
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user("first");
        let m2 = assistant("second");
        let mut store = MessageStore::new();
        store.push(m1.clone()).unwrap();
        store.push(m2.clone()).unwrap();
        let messages: Vec<Message> = store.iter().cloned().collect();

        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .unwrap();

        apply_compaction_to_store(&mut store, &result).unwrap();

        // Originals tombstoned, boundary + summary appended.
        assert_eq!(store.len(), 4);
        let collected: Vec<&Message> = store.iter().collect();
        assert!(matches!(collected[0], Message::Tombstone { reason, .. } if reason == "compacted"));
        assert!(matches!(collected[1], Message::Tombstone { reason, .. } if reason == "compacted"));
        // UUIDs preserved on tombstones.
        assert_eq!(collected[0].uuid(), m1.uuid());
        assert_eq!(collected[1].uuid(), m2.uuid());
        // Boundary + summary at end. Boundary is System; summary is now
        // User (so providers that skip System still see it).
        match collected[2] {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected System boundary at index 2"),
        }
        match collected[3] {
            Message::User { content, .. } => {
                let text = content
                    .iter()
                    .find_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .expect("summary user msg has text block");
                assert!(text.starts_with("[Context summary]"));
                assert!(text.contains("The session"));
            }
            _ => panic!("expected User summary at index 3"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_compaction_earliest_half_inserts_at_boundary() {
        // 4 messages, EarliestHalf compacts the first 2; boundary +
        // summary insert AFTER the tombstones, BEFORE the preserved
        // tail. Final layout: [Tomb, Tomb, System, User, m3, m4].
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user("a");
        let m2 = assistant("b");
        let m3 = user("c");
        let m4 = assistant("d");
        let mut store = MessageStore::new();
        for m in [&m1, &m2, &m3, &m4] {
            store.push(m.clone()).unwrap();
        }
        let messages: Vec<Message> = store.iter().cloned().collect();
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::EarliestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        apply_compaction_to_store(&mut store, &result).unwrap();

        let collected: Vec<&Message> = store.iter().collect();
        assert_eq!(collected.len(), 6);
        assert!(matches!(collected[0], Message::Tombstone { .. }));
        assert!(matches!(collected[1], Message::Tombstone { .. }));
        match collected[2] {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected boundary at idx 2"),
        }
        assert!(matches!(collected[3], Message::User { .. }));
        // Preserved m3, m4 remain at the tail with original UUIDs.
        assert_eq!(collected[4].uuid(), m3.uuid());
        assert_eq!(collected[5].uuid(), m4.uuid());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_compaction_latest_half_inserts_before_replaced() {
        // 4 messages, LatestHalf compacts the last 2; boundary +
        // summary insert BEFORE the replaced tail, AFTER the preserved
        // head. Final layout: [m1, m2, System, User, Tomb, Tomb].
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user("a");
        let m2 = assistant("b");
        let m3 = user("c");
        let m4 = assistant("d");
        let mut store = MessageStore::new();
        for m in [&m1, &m2, &m3, &m4] {
            store.push(m.clone()).unwrap();
        }
        let messages: Vec<Message> = store.iter().cloned().collect();
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::LatestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        apply_compaction_to_store(&mut store, &result).unwrap();

        let collected: Vec<&Message> = store.iter().collect();
        assert_eq!(collected.len(), 6);
        assert_eq!(collected[0].uuid(), m1.uuid());
        assert_eq!(collected[1].uuid(), m2.uuid());
        match collected[2] {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected boundary at idx 2"),
        }
        assert!(matches!(collected[3], Message::User { .. }));
        assert!(matches!(collected[4], Message::Tombstone { .. }));
        assert!(matches!(collected[5], Message::Tombstone { .. }));
    }
}
