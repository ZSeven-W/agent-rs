//! OpenAI-compatible provider (Phase 5 / Task 5.1).
//!
//! Thin wrapper around `async-openai` 0.36 covering the standard
//! Chat Completions streaming protocol. Multiple vendors share this
//! shape with cosmetic differences:
//!
//! | Vendor      | base_url                                    | dialect notes |
//! |-------------|---------------------------------------------|---------------|
//! | OpenAI      | <https://api.openai.com/v1>                 | standard       |
//! | DeepSeek    | <https://api.deepseek.com>                  | reasoning_effort, R1 thinking deltas |
//! | Moonshot    | <https://api.moonshot.cn/v1>                | standard tool_calls |
//! | OpenRouter  | <https://openrouter.ai/api/v1>              | model `org/name` |
//! | Groq        | <https://api.groq.com/openai/v1>            | standard |
//! | LM Studio   | <http://localhost:1234/v1>                  | local, no auth |
//!
//! Feature-gated behind `openai`.

#![allow(clippy::result_large_err)]

use std::collections::HashMap;

use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, FinishReason, FunctionCall,
};
use async_openai::Client;
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::StreamExt;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, Message, ToolResultContent};
use crate::provider::{Provider, ProviderCapabilities, StreamRequest};
use crate::stream::{Event, EventStream, ResultData};

/// Vendor dialect — currently informational; the provider behaves the
/// same across dialects because the wire shape is identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAiDialect {
    #[default]
    Standard,
    /// DeepSeek — supports reasoning_effort + R1 thinking deltas.
    /// (Phase 5 batch L treats them as plain text deltas; richer
    /// thinking surface lands later.)
    DeepSeek,
    /// Moonshot Kimi — standard tool_calls shape.
    Moonshot,
    /// OpenRouter — model name format `org/name`.
    OpenRouter,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    pub api_key: String,
    pub base_url: String,
    pub dialect: OpenAiDialect,
}

impl OpenAiCompatConfig {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            dialect: OpenAiDialect::Standard,
        }
    }

    pub fn with_dialect(mut self, dialect: OpenAiDialect) -> Self {
        self.dialect = dialect;
        self
    }

    /// Convenience for OpenAI's official endpoint.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "https://api.openai.com/v1")
    }
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "https://api.deepseek.com").with_dialect(OpenAiDialect::DeepSeek)
    }
    pub fn moonshot(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "https://api.moonshot.cn/v1").with_dialect(OpenAiDialect::Moonshot)
    }
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "https://openrouter.ai/api/v1").with_dialect(OpenAiDialect::OpenRouter)
    }
    pub fn groq(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "https://api.groq.com/openai/v1")
    }
    pub fn lm_studio() -> Self {
        Self::new("lm-studio", "http://localhost:1234/v1")
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatProvider {
    id: String,
    config: OpenAiCompatConfig,
    capabilities: ProviderCapabilities,
}

impl OpenAiCompatProvider {
    pub fn new(config: OpenAiCompatConfig) -> Self {
        let dialect = config.dialect;
        let id = match dialect {
            OpenAiDialect::Standard => "openai".to_string(),
            OpenAiDialect::DeepSeek => "openai-compat:deepseek".to_string(),
            OpenAiDialect::Moonshot => "openai-compat:moonshot".to_string(),
            OpenAiDialect::OpenRouter => "openai-compat:openrouter".to_string(),
        };
        Self {
            id,
            config,
            capabilities: ProviderCapabilities {
                // tool_use is wire-supported but `StreamRequest` has no
                // `tools` field today (deferred follow-up). Flip to true
                // once tool definitions are threaded through the request
                // builder; advertising true now would mislead callers
                // who gate on this flag.
                supports_tool_use: false,
                supports_prompt_caching: false,
                // DeepSeek R1 emits `reasoning_content` separately on the
                // delta, but `async-openai` 0.36 doesn't surface that
                // field; until we hand-deserialize it via `byot`,
                // thinking tokens are silently dropped — so don't claim
                // support.
                supports_thinking: false,
                max_context_tokens: 128_000,
                needs_placeholder_text_before_tool_use: false,
            },
        }
    }

    fn build_client(&self) -> Client<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_key(self.config.api_key.clone())
            .with_api_base(self.config.base_url.clone());
        Client::with_config(config)
    }
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn id(&self) -> &str {
        &self.id
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

        let messages = render_messages(&req.system, &req.messages)?;
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(&req.model)
            .messages(messages)
            .max_tokens(req.max_output_tokens)
            .stream(true);
        if let Some(temp) = req.temperature {
            builder.temperature(temp);
        }
        if !req.stop_sequences.is_empty() {
            builder.stop(req.stop_sequences.clone());
        }
        // Tool definitions are not part of `StreamRequest` in Phase 1
        // batch B (deferred to a follow-up). When added, build them here
        // by constructing `async_openai::types::chat::ChatCompletionTool`
        // with a `FunctionObject` payload and call `builder.tools(...)`.

        let request = builder
            .build()
            .map_err(|e| AgentError::provider("openai-compat", format!("build request: {e}")))?;

        let mut sse = client
            .chat()
            .create_stream(request)
            .await
            .map_err(|e| AgentError::provider("openai-compat", e.to_string()))?;

        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();

        tokio::spawn(async move {
            let mut tool_call_acc: HashMap<u32, ToolCallAccumulator> = HashMap::new();
            let mut model: Option<String> = None;
            let mut stop_reason: Option<String> = None;

            loop {
                tokio::select! {
                    biased;
                    _ = abort.cancelled() => {
                        let _ = tx.unbounded_send(Err(AgentError::Aborted(
                            abort.reason().unwrap_or_else(|| "aborted".into()),
                        )));
                        return;
                    }
                    item = sse.next() => {
                        let Some(item) = item else { break };
                        match item {
                            Ok(response) => {
                                if model.is_none() && !response.model.is_empty() {
                                    model = Some(response.model.clone());
                                }
                                for choice in response.choices {
                                    if let Some(content) = choice.delta.content {
                                        if !content.is_empty() {
                                            let _ = tx.unbounded_send(Ok(Event::TextDelta {
                                                delta: content,
                                            }));
                                        }
                                    }
                                    if let Some(chunks) = choice.delta.tool_calls {
                                        for chunk in chunks {
                                            accumulate_tool_call(&mut tool_call_acc, chunk);
                                        }
                                    }
                                    if let Some(finish) = choice.finish_reason {
                                        stop_reason = Some(finish_reason_str(finish).to_string());
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = tx.unbounded_send(Err(AgentError::provider(
                                    "openai-compat",
                                    e.to_string(),
                                )));
                                return;
                            }
                        }
                    }
                }
            }

            // Emit accumulated tool calls in their delivery order.
            let mut tool_indices: Vec<u32> = tool_call_acc.keys().copied().collect();
            tool_indices.sort();
            for idx in tool_indices {
                if let Some(acc) = tool_call_acc.remove(&idx) {
                    if let Some(event) = acc.into_event() {
                        if tx.unbounded_send(Ok(event)).is_err() {
                            return;
                        }
                    }
                }
            }

            // Emit the terminal Result event.
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

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccumulator {
    fn into_event(self) -> Option<Event> {
        let id = self.id?;
        let name = self.name?;
        let input = if self.arguments.is_empty() {
            serde_json::Value::Object(Default::default())
        } else {
            serde_json::from_str(&self.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()))
        };
        Some(Event::ToolUse { id, name, input })
    }
}

fn accumulate_tool_call(
    acc: &mut HashMap<u32, ToolCallAccumulator>,
    chunk: ChatCompletionMessageToolCallChunk,
) {
    let entry = acc.entry(chunk.index).or_default();
    if let Some(id) = chunk.id {
        entry.id = Some(id);
    }
    if let Some(function) = chunk.function {
        if let Some(name) = function.name {
            entry.name = Some(name);
        }
        if let Some(args) = function.arguments {
            entry.arguments.push_str(&args);
        }
    }
}

fn render_messages(
    system: &Option<String>,
    messages: &[Message],
) -> Result<Vec<ChatCompletionRequestMessage>, AgentError> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(sys) = system {
        let m = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys.clone())
            .build()
            .map_err(|e| AgentError::provider("openai-compat", format!("system: {e}")))?;
        out.push(m.into());
    }
    for msg in messages {
        match msg {
            Message::User { content, .. } => {
                // Tool results convert to `tool` role messages; the
                // residual non-tool-result content (text/image) becomes
                // a `user` message. Skip the user message entirely if
                // every block was a ToolResult — the tool messages
                // alone carry the turn.
                let user_text = content_to_plain_text(content);
                let any_non_tool_result = content
                    .iter()
                    .any(|b| !matches!(b, ContentBlock::ToolResult { .. }));
                if any_non_tool_result && !user_text.is_empty() {
                    let m = ChatCompletionRequestUserMessageArgs::default()
                        .content(user_text)
                        .build()
                        .map_err(|e| AgentError::provider("openai-compat", format!("user: {e}")))?;
                    out.push(m.into());
                }
                for block in content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content: result_content,
                        ..
                    } = block
                    {
                        let text = match result_content {
                            ToolResultContent::Text(t) => t.clone(),
                            ToolResultContent::Blocks(bs) => content_to_plain_text(bs),
                        };
                        let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(tool_use_id.clone())
                            .content(text)
                            .build()
                            .map_err(|e| {
                                AgentError::provider("openai-compat", format!("tool_result: {e}"))
                            })?;
                        out.push(tool_msg.into());
                    }
                }
            }
            Message::Assistant { content, .. } => {
                // Assistant ToolUse blocks render as a structured
                // `tool_calls` array on the assistant message — NOT
                // as a `[tool_call: name]` text marker (that broke
                // multi-turn tool conversations because the provider
                // couldn't recognize the prior tool invocation).
                let assistant_text = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let tool_calls: Vec<ChatCompletionMessageToolCalls> = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, name, input } => {
                            Some(ChatCompletionMessageToolCalls::Function(
                                ChatCompletionMessageToolCall {
                                    id: id.clone(),
                                    function: FunctionCall {
                                        name: name.clone(),
                                        arguments: input.to_string(),
                                    },
                                },
                            ))
                        }
                        _ => None,
                    })
                    .collect();
                if assistant_text.is_empty() && tool_calls.is_empty() {
                    // OpenAI rejects assistant messages with neither
                    // `content` nor `tool_calls`. Happens when the
                    // assistant block list contains only Images (we
                    // don't yet base64-encode them into the request)
                    // or got fully tombstoned. Skip rather than push
                    // an invalid-payload message.
                    tracing::debug!(
                        "openai-compat: skipped assistant message with no content or tool_calls",
                    );
                    continue;
                }
                let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
                if !assistant_text.is_empty() {
                    builder.content(assistant_text);
                }
                if !tool_calls.is_empty() {
                    builder.tool_calls(tool_calls);
                }
                let m = builder.build().map_err(|e| {
                    AgentError::provider("openai-compat", format!("assistant: {e}"))
                })?;
                out.push(m.into());
            }
            Message::System { text, .. } => {
                let m = ChatCompletionRequestSystemMessageArgs::default()
                    .content(text.clone())
                    .build()
                    .map_err(|e| AgentError::provider("openai-compat", format!("system: {e}")))?;
                out.push(m.into());
            }
            Message::Progress { .. } | Message::Tombstone { .. } => {
                // Agent-internal — skip.
            }
        }
    }
    Ok(out)
}

fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::FunctionCall => "function_call",
    }
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
                // ToolUse on assistant messages is rendered as
                // structured `tool_calls` in the assistant builder
                // (see render_messages), NOT as plain text — that
                // broke multi-turn tool conversations.
                // Images: simple wrapper doesn't translate base64 yet
                // (ChatCompletionRequestUserMessageContent::Array
                // upgrade is a follow-up).
                // Tool results: surfaced as separate `tool` role
                // messages (see render_messages).
            }
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Header};

    #[test]
    fn config_helpers_set_defaults() {
        let c = OpenAiCompatConfig::openai("sk-test");
        assert_eq!(c.base_url, "https://api.openai.com/v1");
        assert_eq!(c.dialect, OpenAiDialect::Standard);
        let c = OpenAiCompatConfig::deepseek("sk-test");
        assert_eq!(c.dialect, OpenAiDialect::DeepSeek);
        let c = OpenAiCompatConfig::moonshot("sk-test");
        assert_eq!(c.dialect, OpenAiDialect::Moonshot);
        let c = OpenAiCompatConfig::openrouter("sk-test");
        assert_eq!(c.dialect, OpenAiDialect::OpenRouter);
    }

    #[test]
    fn provider_id_reflects_dialect() {
        let p = OpenAiCompatProvider::new(OpenAiCompatConfig::openai("k"));
        assert_eq!(p.id(), "openai");
        let p = OpenAiCompatProvider::new(OpenAiCompatConfig::deepseek("k"));
        assert_eq!(p.id(), "openai-compat:deepseek");
    }

    #[test]
    fn capabilities_are_conservative_until_features_implemented() {
        // Both supports_tool_use and supports_thinking are off in
        // batch L because (a) StreamRequest has no `tools` field yet
        // (deferred), and (b) async-openai 0.36 doesn't surface
        // DeepSeek's `reasoning_content` field. Flipping these on
        // before the corresponding code paths exist would mislead
        // capability-gated callers.
        let p = OpenAiCompatProvider::new(OpenAiCompatConfig::openai("k"));
        assert!(!p.capabilities().supports_tool_use);
        assert!(!p.capabilities().supports_thinking);
        let p = OpenAiCompatProvider::new(OpenAiCompatConfig::deepseek("k"));
        assert!(!p.capabilities().supports_tool_use);
        assert!(!p.capabilities().supports_thinking);
    }

    #[test]
    fn finish_reason_string_mapping() {
        assert_eq!(finish_reason_str(FinishReason::Stop), "stop");
        assert_eq!(finish_reason_str(FinishReason::Length), "length");
        assert_eq!(finish_reason_str(FinishReason::ToolCalls), "tool_calls");
        assert_eq!(
            finish_reason_str(FinishReason::ContentFilter),
            "content_filter"
        );
        assert_eq!(
            finish_reason_str(FinishReason::FunctionCall),
            "function_call"
        );
    }

    #[test]
    fn render_messages_strips_progress_and_tombstone() {
        let messages = vec![
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
            },
            Message::Progress {
                header: Header::new(),
                note: "compacting".into(),
            },
            Message::Tombstone {
                header: Header::new(),
                reason: "deleted".into(),
            },
            Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            },
        ];
        let rendered = render_messages(&None, &messages).unwrap();
        // Only User + Assistant survive (2 messages).
        assert_eq!(rendered.len(), 2);
    }

    #[test]
    fn render_messages_emits_only_tool_message_when_user_is_pure_tool_result() {
        let messages = vec![Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: ToolResultContent::Text("output".into()),
                is_error: false,
            }],
        }];
        let rendered = render_messages(&None, &messages).unwrap();
        // Only the tool role message — no double user message.
        assert_eq!(rendered.len(), 1);
    }

    #[test]
    fn render_messages_keeps_user_when_mixed_text_and_tool_result() {
        let messages = vec![Message::User {
            header: Header::new(),
            content: vec![
                ContentBlock::Text {
                    text: "follow-up".into(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: ToolResultContent::Text("output".into()),
                    is_error: false,
                },
            ],
        }];
        let rendered = render_messages(&None, &messages).unwrap();
        // user (with the text) + tool_result.
        assert_eq!(rendered.len(), 2);
    }

    #[test]
    fn render_messages_skips_empty_assistant() {
        // Assistant with only an Image block (which our renderer
        // doesn't base64-translate yet) produces empty content +
        // empty tool_calls. We must skip rather than emit an invalid
        // message.
        let messages = vec![Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Image {
                source: crate::message::ImageSource::Url {
                    url: "https://example.com/x.png".into(),
                },
            }],
        }];
        let rendered = render_messages(&None, &messages).unwrap();
        assert_eq!(rendered.len(), 0);
    }

    #[test]
    fn render_messages_assistant_tool_use_becomes_tool_calls() {
        // Multi-turn pattern: assistant emits a tool_use block; the
        // request must include it as structured `tool_calls`, not as
        // text — otherwise the next turn's tool_result has nothing
        // to refer back to.
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
        let rendered = render_messages(&None, &messages).unwrap();
        assert_eq!(rendered.len(), 1);
        // Serialize to JSON and inspect — the rendered shape includes
        // `tool_calls` not a "[tool_call: ...]" text marker.
        let json = serde_json::to_value(&rendered[0]).unwrap();
        assert!(
            json["tool_calls"]
                .as_array()
                .map(|a| !a.is_empty())
                .unwrap_or(false),
            "expected tool_calls array, got {json}"
        );
        assert!(
            !json["content"]
                .as_str()
                .unwrap_or("")
                .contains("[tool_call:"),
            "assistant text should not contain tool_call marker"
        );
    }

    #[test]
    fn content_to_plain_text_concatenates_with_newlines() {
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
    fn tool_call_accumulator_assembles_streamed_chunks() {
        let mut acc = HashMap::new();
        accumulate_tool_call(
            &mut acc,
            ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call_1".into()),
                r#type: Some(async_openai::types::chat::FunctionType::Function),
                function: Some(async_openai::types::chat::FunctionCallStream {
                    name: Some("calc".into()),
                    arguments: Some("{\"a\":".into()),
                }),
            },
        );
        accumulate_tool_call(
            &mut acc,
            ChatCompletionMessageToolCallChunk {
                index: 0,
                id: None,
                r#type: None,
                function: Some(async_openai::types::chat::FunctionCallStream {
                    name: None,
                    arguments: Some("1}".into()),
                }),
            },
        );
        let event = acc.remove(&0).unwrap().into_event().unwrap();
        match event {
            Event::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "calc");
                assert_eq!(input, serde_json::json!({"a": 1}));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    /// Real-API integration test, gated by OPENAI_API_KEY env var.
    /// Skipped by default; run with `cargo test --features openai
    /// openai_real_api -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY env var; hits real OpenAI API"]
    async fn openai_real_api_hello() {
        let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
            eprintln!("OPENAI_API_KEY not set; skipping real-API test");
            return;
        };
        let provider = OpenAiCompatProvider::new(OpenAiCompatConfig::openai(api_key));
        let req = StreamRequest::new(
            "gpt-5.4-codex",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text {
                    text: "Reply with just the word: ok".into(),
                }],
            }],
        )
        .with_max_output_tokens(64);
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
