//! Ollama provider (Phase 5 / Task 5.2).
//!
//! Wraps `ollama-rs` 0.3 to talk to a local (or remote) Ollama
//! daemon. Streams text deltas from `send_chat_messages_stream`;
//! tool calling is documented but capability-flagged off until
//! [`crate::provider::StreamRequest`] gains a `tools` field.
//!
//! Feature-gated behind `ollama`.

#![allow(clippy::result_large_err)]

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::StreamExt;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::Ollama;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, Message, ToolResultContent};
use crate::provider::{Provider, ProviderCapabilities, StreamRequest};
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
                // Ollama supports tools on the wire (ChatMessageRequest
                // has a `tools` field) but our StreamRequest doesn't
                // surface tool definitions yet — flip on when that
                // arrives. Same scope boundary as openai_compat.rs.
                supports_tool_use: false,
                supports_prompt_caching: false,
                supports_thinking: false,
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
        let request = ChatMessageRequest::new(request_model.clone(), messages);

        let mut sse = client
            .send_chat_messages_stream(request)
            .await
            .map_err(|e| AgentError::provider("ollama", format!("model={request_model}: {e}")))?;

        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();
        let error_model = request_model;

        tokio::spawn(async move {
            let mut model: Option<String> = None;
            let mut last_done = false;

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
                if !text.is_empty() {
                    out.push(ChatMessage::assistant(text));
                }
                // Tool_use blocks: Ollama's ChatMessage doesn't have a
                // dedicated tool_calls field exposed at this trait
                // level; capability flag is off and we don't emit a
                // text marker (would mislead the model). When tools
                // are wired through StreamRequest, build the typed
                // tool_calls payload then.
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
            ContentBlock::ToolUse { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolResult { .. } => {
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
    fn capabilities_are_conservative() {
        let p = OllamaProvider::local();
        let caps = p.capabilities();
        assert!(!caps.supports_tool_use);
        assert!(!caps.supports_prompt_caching);
        assert!(!caps.supports_thinking);
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
