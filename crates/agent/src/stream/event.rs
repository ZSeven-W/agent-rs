use std::collections::HashMap;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::AgentError;

/// A single event emitted during a query turn.
///
/// Tagged union with `kind` discriminator. New variants will be added in
/// later phases (`#[non_exhaustive]` keeps the SemVer door open for that).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Event {
    /// Streaming text delta from the assistant.
    TextDelta { delta: String },
    /// Streaming "extended thinking" delta.
    Thinking { delta: String },
    /// Assistant requested a tool call.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool produced a result for a prior `ToolUse`.
    ToolResult {
        id: String,
        ok: bool,
        output: serde_json::Value,
    },
    /// Final result envelope. Emitted at most once per turn.
    Result { data: ResultData },
    /// Token usage for the turn so far. May be emitted multiple times.
    Usage {
        input_tokens: u32,
        output_tokens: u32,
        cache_read: u32,
        cache_create: u32,
    },
    /// Provider-side or runtime error. Recoverable cases continue; fatal
    /// cases are followed by stream termination.
    Error { code: String, message: String },
    /// Forward-compatibility catch-all. Any `kind` value not recognized by
    /// this crate version deserializes to `Unknown` instead of failing the
    /// whole stream. Used by older consumers when newer producers emit
    /// event kinds added in later phases (typical scenario: a long-running
    /// daemon ships ahead of its WASM/web frontend).
    ///
    /// **Payload is dropped** on read because `#[serde(other)]` requires a
    /// unit variant. Consumers that need to forward raw unknown events
    /// should peel them off the wire (raw `serde_json::Value`) before
    /// deserializing into [`Event`].
    #[serde(other)]
    Unknown,
}

/// Final-result metadata.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ResultData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Marker trait for streams that produce `Result<Event, AgentError>` items.
///
/// Blanket-implemented for every `Stream` matching the bounds, so callers
/// can write `Box<dyn EventStream>` instead of the longer
/// `Pin<Box<dyn Stream<Item = ..>>>` form for already-`Unpin` streams
/// (e.g., `futures::stream::iter` results, or `Box::pin(...)` wrappers).
pub trait EventStream:
    Stream<Item = Result<Event, AgentError>> + Unpin + Send
{
}

impl<T> EventStream for T where
    T: Stream<Item = Result<Event, AgentError>> + Unpin + Send
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{stream, StreamExt};

    fn roundtrip(e: &Event) -> Event {
        let j = serde_json::to_string(e).unwrap();
        serde_json::from_str(&j).unwrap()
    }

    #[test]
    fn text_delta_roundtrip() {
        let e = Event::TextDelta { delta: "hi".into() };
        assert_eq!(e, roundtrip(&e));
    }

    #[test]
    fn thinking_roundtrip() {
        let e = Event::Thinking {
            delta: "let me think...".into(),
        };
        assert_eq!(e, roundtrip(&e));
    }

    #[test]
    fn tool_use_roundtrip() {
        let e = Event::ToolUse {
            id: "tu_1".into(),
            name: "read_file".into(),
            input: serde_json::json!({"path": "/tmp/x"}),
        };
        assert_eq!(e, roundtrip(&e));
    }

    #[test]
    fn tool_result_roundtrip() {
        let e = Event::ToolResult {
            id: "tu_1".into(),
            ok: true,
            output: serde_json::json!({"text": "ok"}),
        };
        assert_eq!(e, roundtrip(&e));
    }

    #[test]
    fn result_with_metadata_roundtrip() {
        let mut metadata = HashMap::new();
        metadata.insert("retries".into(), serde_json::json!(2));
        let e = Event::Result {
            data: ResultData {
                stop_reason: Some("end_turn".into()),
                model: Some("claude-opus-4-7".into()),
                metadata,
            },
        };
        assert_eq!(e, roundtrip(&e));
    }

    #[test]
    fn usage_roundtrip() {
        let e = Event::Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read: 200,
            cache_create: 0,
        };
        assert_eq!(e, roundtrip(&e));
    }

    #[test]
    fn error_roundtrip() {
        let e = Event::Error {
            code: "rate_limit".into(),
            message: "slow down".into(),
        };
        assert_eq!(e, roundtrip(&e));
    }

    #[test]
    fn json_kind_tag_is_snake_case() {
        let e = Event::TextDelta { delta: "x".into() };
        let j = serde_json::to_value(&e).unwrap();
        assert_eq!(j["kind"], "text_delta");
    }

    #[test]
    fn unknown_kind_deserializes_to_unknown_variant() {
        // A future variant, not yet defined in this crate version.
        let j = r#"{"kind": "future_variant_v2", "payload": {"x": 1}}"#;
        let parsed: Event = serde_json::from_str(j).unwrap();
        assert!(matches!(parsed, Event::Unknown));
    }

    #[tokio::test]
    async fn event_stream_blanket_impl() {
        // futures::stream::iter is Unpin + Send, so it gets EventStream for free.
        let s = stream::iter(vec![
            Ok(Event::TextDelta { delta: "a".into() }),
            Ok(Event::TextDelta { delta: "b".into() }),
        ]);
        // Compile-time check that it satisfies the trait bound.
        fn assert_event_stream<E: EventStream>(_: &E) {}
        assert_event_stream(&s);

        let collected: Vec<_> = s.collect().await;
        assert_eq!(collected.len(), 2);
        for r in collected {
            assert!(r.is_ok());
        }
    }
}
