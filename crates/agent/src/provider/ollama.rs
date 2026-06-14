//! Ollama provider (Phase 5 / Task 5.2).
//!
//! Wraps `ollama-rs` 0.3 to talk to a local (or remote) Ollama
//! daemon. Streams text deltas from `send_chat_messages_stream`;
//! tool calls are surfaced from `response.message.tool_calls` and
//! emitted as [`crate::stream::Event::ToolUse`]. Synthetic ids
//! (`ollama_tc_<n>`) bridge Ollama's id-less ToolCall to the
//! [`crate::tool::Tool`] dispatch loop, which is id-based.
//!
//! Feature-gated behind `ollama`.

#![allow(clippy::result_large_err)]

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::StreamExt;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use ollama_rs::generation::tools::{ToolCall, ToolCallFunction, ToolInfo};
use ollama_rs::Ollama;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, Message, ToolResultContent};
use crate::provider::{Provider, ProviderCapabilities, StreamRequest, ToolChoice, ToolDefinition};
use crate::stream::{Event, EventStream, ResultData};

const DEFAULT_HOST: &str = "http://localhost";
const DEFAULT_PORT: u16 = 11434;

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.into(),
            port: DEFAULT_PORT,
        }
    }
}

impl OllamaConfig {
    /// Default localhost:11434 (Ollama's standard daemon port).
    pub fn local() -> Self {
        Self::default()
    }

    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    config: OllamaConfig,
    capabilities: ProviderCapabilities,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config,
            capabilities: ProviderCapabilities {
                // tools are wired through `ChatMessageRequest::tools`
                // and tool_calls are surfaced from streaming responses.
                supports_tool_use: true,
                supports_prompt_caching: false,
                supports_thinking: false,
                supports_images: false,
                // Ollama context cap is per-model; this is a sane
                // default for llama3 / mistral. Callers should
                // override for larger models like deepseek-r1:32b
                // (128k) or qwen2.5:72b (32k).
                max_context_tokens: 8_192,
                needs_placeholder_text_before_tool_use: false,
            },
        }
    }

    pub fn local() -> Self {
        Self::new(OllamaConfig::default())
    }

    fn build_client(&self) -> Ollama {
        Ollama::new(self.config.host.clone(), self.config.port)
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn id(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }

    async fn stream(
        &self,
        req: StreamRequest,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        let client = self.build_client();

        let messages = render_messages(&req.system, &req.messages);
        let request_model = req.model.clone();
        // ToolChoice::None is honored by simply not sending tools; the
        // other variants don't have a wire representation in Ollama —
        // the daemon decides on its own whether to call. We log the
        // unsupported variants at trace level for diagnosability.
        let suppress_tools = matches!(req.tool_choice, Some(ToolChoice::None));
        let mut request = ChatMessageRequest::new(request_model.clone(), messages);
        if !req.tools.is_empty() && !suppress_tools {
            let tools = render_tools(&req.tools)
                .map_err(|e| AgentError::provider("ollama", format!("invalid tool schema: {e}")))?;
            request = request.tools(tools);
            if let Some(choice) = &req.tool_choice {
                if !matches!(choice, ToolChoice::Auto) {
                    tracing::trace!(
                        target: "agent::provider::ollama",
                        ?choice,
                        "Ollama has no tool_choice on the wire; ignored"
                    );
                }
            }
        }

        let mut sse = client
            .send_chat_messages_stream(request)
            .await
            .map_err(|e| AgentError::provider("ollama", format!("model={request_model}: {e}")))?;

        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();
        let error_model = request_model;

        tokio::spawn(async move {
            let mut model: Option<String> = None;
            let mut last_done = false;
            // Ollama's `ToolCall` has no id field (the daemon matches
            // by message position), so we synthesize stable IDs for
            // downstream `tool_result` correlation.
            let mut tool_call_seq: u32 = 0;

            loop {
                tokio::select! {
                    biased;
                    _ = abort.cancelled() => {
                        let _ = tx.unbounded_send(Err(AgentError::Aborted(
                            abort.reason().unwrap_or_else(|| "aborted".into()),
                        )));
                        return;
                    }
                    next = sse.next() => {
                        let Some(item) = next else { break };
                        match item {
                            Ok(response) => {
                                if model.is_none() && !response.model.is_empty() {
                                    model = Some(response.model.clone());
                                }
                                let text = response.message.content;
                                if !text.is_empty() {
                                    let _ = tx.unbounded_send(Ok(Event::TextDelta {
                                        delta: text,
                                    }));
                                }
                                for tc in response.message.tool_calls {
                                    let id = format!("ollama_tc_{tool_call_seq}");
                                    tool_call_seq += 1;
                                    let _ = tx.unbounded_send(Ok(Event::ToolUse {
                                        id,
                                        name: tc.function.name,
                                        input: tc.function.arguments,
                                    }));
                                }
                                if response.done {
                                    last_done = true;
                                }
                            }
                            Err(()) => {
                                // ollama-rs surfaces stream errors as
                                // a unit type (`Result<_, ()>`) — no
                                // structured payload available, so we
                                // include at least the requested model
                                // in the message for diagnostics.
                                let _ = tx.unbounded_send(Err(AgentError::provider(
                                    "ollama",
                                    format!(
                                        "stream error from ollama daemon (model={error_model})"
                                    ),
                                )));
                                return;
                            }
                        }
                    }
                }
            }

            let stop_reason = if last_done { Some("stop".into()) } else { None };
            let _ = tx.unbounded_send(Ok(Event::Result {
                data: ResultData {
                    stop_reason,
                    model,
                    metadata: Default::default(),
                },
            }));
        });

        Ok(Box::new(rx))
    }
}

/// Convert provider-neutral tool definitions into ollama-rs `ToolInfo`.
///
/// `ToolInfo::new::<P, T>()` is `pub(crate)` and requires a compile-time
/// `Parameters` type, so we round-trip through JSON to construct the
/// `Schema`-bearing struct without a custom schemars type. `ToolInfo`
/// implements `Deserialize` and the inner `Schema(Value)` accepts any
/// JSON Schema object.
fn render_tools(tools: &[ToolDefinition]) -> Result<Vec<ToolInfo>, serde_json::Error> {
    tools
        .iter()
        .map(|t| {
            // ollama-rs `ToolType` is a unit enum derived as PascalCase
            // for both serialize and deserialize, so the tag is the
            // literal string "Function" — not "function". Round-tripping
            // through JSON is the only public way to construct
            // `ToolInfo` (its `new::<P,T>()` constructor is `pub(crate)`
            // and demands a compile-time `Parameters` type).
            serde_json::from_value::<ToolInfo>(serde_json::json!({
                "type": "Function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }))
        })
        .collect()
}

fn render_messages(system: &Option<String>, messages: &[Message]) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(sys) = system {
        out.push(ChatMessage::system(sys.clone()));
    }
    for msg in messages {
        match msg {
            Message::User { content, .. } => {
                let pure_tool_results = !content.is_empty()
                    && content
                        .iter()
                        .all(|b| matches!(b, ContentBlock::ToolResult { .. }));
                if pure_tool_results {
                    // Each tool_result becomes its own `tool` role
                    // message — skip the (would-be-empty) user message.
                } else {
                    let text = content_to_plain_text(content);
                    if !text.is_empty() {
                        out.push(ChatMessage::user(text));
                    }
                }
                for block in content {
                    if let ContentBlock::ToolResult {
                        content: result_content,
                        ..
                    } = block
                    {
                        let text = match result_content {
                            ToolResultContent::Text(t) => t.clone(),
                            ToolResultContent::Blocks(bs) => content_to_plain_text(bs),
                        };
                        // Ollama's ChatMessage::tool() doesn't carry a
                        // tool_use_id; the daemon matches by position
                        // in the message history.
                        out.push(ChatMessage::tool(text));
                    }
                }
            }
            Message::Assistant { content, .. } => {
                let text = content_to_plain_text(content);
                let tool_calls: Vec<ToolCall> = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { name, input, .. } => Some(ToolCall {
                            function: ToolCallFunction {
                                name: name.clone(),
                                arguments: input.clone(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();
                // Skip emitting the assistant message altogether if it
                // would carry no content AND no tool_calls — pushing a
                // blank ChatMessage would be wire-rejected by Ollama.
                if !text.is_empty() || !tool_calls.is_empty() {
                    out.push(ChatMessage {
                        role: MessageRole::Assistant,
                        content: text,
                        tool_calls,
                        images: None,
                        thinking: None,
                    });
                }
            }
            Message::System { text, .. } => {
                out.push(ChatMessage::system(text.clone()));
            }
            Message::Progress { .. } | Message::Tombstone { .. } => {
                // Agent-internal — skip.
            }
        }
    }
    out
}

fn content_to_plain_text(blocks: &[ContentBlock]) -> String {
    let mut buf = String::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(text);
            }
            ContentBlock::Thinking { thinking, .. } => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(thinking);
            }
            ContentBlock::Image { .. } => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str("[image attachment]");
            }
            ContentBlock::Document { .. } => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str("[document attachment]");
            }
            ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => {
                // ToolUse: render via tool_calls when wired (skipped now).
                // Image: Ollama supports vision via message.images Vec<Image>;
                //   not yet threaded.
                // ToolResult: surfaced as separate `tool` role message.
            }
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Header;

    #[test]
    fn config_defaults_to_local() {
        let c = OllamaConfig::default();
        assert_eq!(c.host, "http://localhost");
        assert_eq!(c.port, 11434);
    }

    #[test]
    fn provider_id_is_ollama() {
        let p = OllamaProvider::local();
        assert_eq!(p.id(), "ollama");
    }

    #[test]
    fn capabilities_advertise_tool_use() {
        // tools are now wired through ChatMessageRequest::tools().
        // Caching and thinking remain off — Ollama daemon doesn't
        // expose either at the wire level.
        let p = OllamaProvider::local();
        let caps = p.capabilities();
        assert!(caps.supports_tool_use);
        assert!(!caps.supports_prompt_caching);
        assert!(!caps.supports_thinking);
    }

    #[test]
    fn render_tools_produces_valid_tool_info() {
        let defs = vec![ToolDefinition::new(
            "calc",
            "perform arithmetic",
            serde_json::json!({"type": "object", "properties": {"a": {"type": "number"}}}),
        )];
        let tools = render_tools(&defs).expect("render");
        assert_eq!(tools.len(), 1);
        // Round-trip through serde to inspect the wire shape.
        let json = serde_json::to_value(&tools[0]).expect("serialize");
        assert_eq!(json["type"], "Function");
        assert_eq!(json["function"]["name"], "calc");
        assert_eq!(json["function"]["description"], "perform arithmetic");
        assert_eq!(json["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn render_messages_assistant_tool_use_becomes_tool_calls() {
        // Round-trip through the renderer must preserve tool_use blocks
        // as Ollama tool_calls so the next turn's tool_result has
        // something to refer back to. Previously this was dropped
        // silently — bug (j) from codex round-2 review.
        let messages = vec![Message::Assistant {
            header: Header::new(),
            content: vec![
                ContentBlock::Text {
                    text: "let me think...".into(),
                },
                ContentBlock::ToolUse {
                    id: "tu_42".into(),
                    name: "calc".into(),
                    input: serde_json::json!({"a": 1}),
                },
            ],
        }];
        let rendered = render_messages(&None, &messages);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].role, MessageRole::Assistant);
        assert_eq!(rendered[0].content, "let me think...");
        assert_eq!(rendered[0].tool_calls.len(), 1);
        assert_eq!(rendered[0].tool_calls[0].function.name, "calc");
        assert_eq!(
            rendered[0].tool_calls[0].function.arguments,
            serde_json::json!({"a": 1})
        );
    }

    #[test]
    fn render_messages_assistant_tool_use_without_text_still_renders() {
        // An assistant turn that's tool-use only (no preceding text)
        // must still surface as a ChatMessage with tool_calls populated.
        let messages = vec![Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::ToolUse {
                id: "tu_1".into(),
                name: "search".into(),
                input: serde_json::json!({"q": "rust"}),
            }],
        }];
        let rendered = render_messages(&None, &messages);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].content, "");
        assert_eq!(rendered[0].tool_calls.len(), 1);
    }

    #[test]
    fn render_tools_rejects_non_object_schema() {
        // schemars 1.x's Schema validates shape — a bare boolean false
        // is a valid Schema (matches nothing) but a non-object/non-bool
        // payload should be rejected. We don't enforce object-ness
        // ourselves; we surface whatever Schema's validator says.
        let defs = vec![ToolDefinition::new(
            "bad",
            "",
            serde_json::Value::Number(serde_json::Number::from(7)),
        )];
        let err = render_tools(&defs).expect_err("number is not a valid schema");
        let msg = err.to_string();
        assert!(
            msg.to_ascii_lowercase().contains("schema") || msg.contains("expected"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn render_messages_handles_user_assistant_system() {
        let messages = vec![
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
            },
            Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            },
            Message::System {
                header: Header::new(),
                text: "be brief".into(),
            },
        ];
        let rendered = render_messages(&Some("top-system".into()), &messages);
        assert_eq!(rendered.len(), 4); // top-system + user + assistant + inline-system
    }

    #[test]
    fn render_messages_skips_progress_and_tombstone() {
        let messages = vec![
            Message::Progress {
                header: Header::new(),
                note: "n/a".into(),
            },
            Message::Tombstone {
                header: Header::new(),
                reason: "deleted".into(),
            },
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
            },
        ];
        let rendered = render_messages(&None, &messages);
        // Only the User message survives.
        assert_eq!(rendered.len(), 1);
    }

    #[test]
    fn render_messages_pure_tool_result_user_emits_only_tool_role() {
        let messages = vec![Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: ToolResultContent::Text("output".into()),
                is_error: false,
            }],
        }];
        let rendered = render_messages(&None, &messages);
        // Just one tool message — no user shell.
        assert_eq!(rendered.len(), 1);
    }

    #[test]
    fn content_to_plain_text_concatenates() {
        let blocks = vec![
            ContentBlock::Text {
                text: "first".into(),
            },
            ContentBlock::Text {
                text: "second".into(),
            },
        ];
        assert_eq!(content_to_plain_text(&blocks), "first\nsecond");
    }

    #[test]
    fn content_to_plain_text_surfaces_image_and_document_placeholders() {
        use crate::message::DocumentSource;
        let blocks = vec![
            ContentBlock::Text { text: "hi".into() },
            ContentBlock::Image {
                source: crate::message::ImageSource::File {
                    file_id: "file_a".into(),
                },
            },
            ContentBlock::Document {
                source: DocumentSource::File {
                    file_id: "file_b".into(),
                },
            },
        ];
        let out = content_to_plain_text(&blocks);
        assert!(out.contains("hi"));
        assert!(out.contains("[image attachment]"), "got {out}");
        assert!(out.contains("[document attachment]"), "got {out}");
    }

    /// Real-Ollama integration test, gated by OLLAMA_TEST_MODEL env
    /// var. Run with a local Ollama daemon + a small model pulled:
    /// `ollama pull tinyllama; OLLAMA_TEST_MODEL=tinyllama cargo test
    /// --features ollama ollama_real_local -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "requires local Ollama daemon + OLLAMA_TEST_MODEL env var"]
    async fn ollama_real_local() {
        let Ok(model) = std::env::var("OLLAMA_TEST_MODEL") else {
            eprintln!("OLLAMA_TEST_MODEL not set; skipping");
            return;
        };
        let provider = OllamaProvider::local();
        let req = StreamRequest::new(
            model,
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text {
                    text: "Reply with the single word: ok".into(),
                }],
            }],
        );
        let mut stream = provider.stream(req, AbortController::new()).await.unwrap();
        let mut got_text = String::new();
        let mut got_result = false;
        use futures::StreamExt as _;
        while let Some(item) = stream.next().await {
            match item.unwrap() {
                Event::TextDelta { delta } => got_text.push_str(&delta),
                Event::Result { .. } => got_result = true,
                _ => {}
            }
        }
        assert!(got_result);
        assert!(!got_text.is_empty());
    }
}
