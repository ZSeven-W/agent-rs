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
    ChatCompletionMessageToolCalls, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionStreamOptions, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionTools, CreateChatCompletionRequestArgs,
    FinishReason, FunctionCall, FunctionName, FunctionObject, ImageUrl, ToolChoiceOptions,
};
use async_openai::Client;
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::StreamExt;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, Message, ToolResultContent};
use crate::provider::{Provider, ProviderCapabilities, StreamRequest, ToolChoice, ToolDefinition};
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
    pub supports_images: bool,
}

impl OpenAiCompatConfig {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            dialect: OpenAiDialect::Standard,
            supports_images: false,
        }
    }

    pub fn with_dialect(mut self, dialect: OpenAiDialect) -> Self {
        self.dialect = dialect;
        self
    }

    pub fn with_supports_images(mut self, supports_images: bool) -> Self {
        self.supports_images = supports_images;
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
        let supports_images = config.supports_images;
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
                supports_tool_use: true,
                supports_prompt_caching: false,
                // DeepSeek R1 emits `reasoning_content` separately on the
                // delta, but `async-openai` 0.36 doesn't surface that
                // field; until we hand-deserialize it via `byot`,
                // thinking tokens are silently dropped — so don't claim
                // support.
                supports_thinking: false,
                supports_images,
                max_context_tokens: 128_000,
                needs_placeholder_text_before_tool_use: false,
            },
        }
    }

    fn build_client(&self) -> Client<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_key(self.config.api_key.clone())
            .with_api_base(self.config.base_url.clone());
        // Bounded connect/idle-read timeouts (shared policy with the
        // Anthropic provider) so a black-holed endpoint can't hang a turn.
        Client::with_config(config).with_http_client(super::default_http_client())
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
            .stream(true)
            // Ask for a final usage chunk so we can surface token + cache
            // counts (DeepSeek reports prompt_tokens_details.cached_tokens here).
            .stream_options(ChatCompletionStreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        if let Some(temp) = req.temperature {
            builder.temperature(temp);
        }
        if !req.stop_sequences.is_empty() {
            builder.stop(req.stop_sequences.clone());
        }
        if !req.tools.is_empty() {
            builder.tools(render_tools(&req.tools));
        }
        if let Some(choice) = &req.tool_choice {
            // OpenAI rejects `tool_choice` if no tools are present (the
            // server returns "tool_choice: none/required is not allowed
            // when tools is not specified"). Drop it silently to match
            // Anthropic's behavior.
            if !req.tools.is_empty() {
                builder.tool_choice(render_tool_choice(choice));
            }
        }

        let request = builder
            .build()
            .map_err(|e| AgentError::provider("openai-compat", format!("build request: {e}")))?;

        // Wrap request initiation in an abort select — without it, a hung
        // connection blocks past the user's cancel until transport timeouts.
        let chat = client.chat();
        let create = chat.create_stream(request);
        let mut sse = tokio::select! {
            biased;
            _ = abort.cancelled() => {
                return Err(AgentError::Aborted("request cancelled".into()));
            }
            r = create => {
                r.map_err(|e| AgentError::provider("openai-compat", error_message(&e)))?
            }
        };

        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();

        tokio::spawn(async move {
            let mut tool_call_acc: HashMap<u32, ToolCallAccumulator> = HashMap::new();
            let mut model: Option<String> = None;
            let mut stop_reason: Option<String> = None;
            let mut usage: Option<(u32, u32, u32)> = None; // (input, output, cache_read)

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
                                // Final chunk (choices empty) carries usage when
                                // include_usage is set. cached_tokens is the
                                // server-side prefix-cache hit (DeepSeek et al.).
                                if let Some(u) = &response.usage {
                                    let cached = u
                                        .prompt_tokens_details
                                        .as_ref()
                                        .and_then(|d| d.cached_tokens)
                                        .unwrap_or(0);
                                    // Report input_tokens as the NON-cached
                                    // (full-rate) prompt tokens, matching the
                                    // Anthropic provider's convention where
                                    // cache_read is separate. DeepSeek's
                                    // prompt_tokens is the TOTAL, so subtract
                                    // the cached portion — otherwise downstream
                                    // cache% (cache/total) is inconsistent
                                    // across providers.
                                    let non_cached = u.prompt_tokens.saturating_sub(cached);
                                    usage = Some((non_cached, u.completion_tokens, cached));
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
                                    error_message(&e),
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

            // Emit token usage (with prompt-cache hits) ahead of the result.
            if let Some((input_tokens, output_tokens, cache_read)) = usage {
                let _ = tx.unbounded_send(Ok(Event::Usage {
                    input_tokens,
                    output_tokens,
                    cache_read,
                    cache_create: 0,
                }));
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

/// Error text with a leading `HTTP <code>` when the underlying transport
/// exposes a status — the driver's retry classifier works from a real
/// code instead of substring guessing (async-openai's `ApiError` drops
/// the HTTP status entirely; only the Reqwest variant still carries it).
fn error_message(e: &async_openai::error::OpenAIError) -> String {
    if let async_openai::error::OpenAIError::Reqwest(re) = e {
        if let Some(status) = re.status() {
            return format!("HTTP {status}: {re}");
        }
    }
    e.to_string()
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

fn render_tools(tools: &[ToolDefinition]) -> Vec<ChatCompletionTools> {
    tools
        .iter()
        .map(|t| {
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: t.name.clone(),
                    description: if t.description.is_empty() {
                        None
                    } else {
                        Some(t.description.clone())
                    },
                    parameters: Some(t.input_schema.clone()),
                    strict: None,
                },
            })
        })
        .collect()
}

fn render_tool_choice(choice: &ToolChoice) -> ChatCompletionToolChoiceOption {
    match choice {
        ToolChoice::Auto => ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto),
        ToolChoice::Required => ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Required),
        ToolChoice::None => ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::None),
        ToolChoice::Tool(name) => {
            ChatCompletionToolChoiceOption::Function(ChatCompletionNamedToolChoice {
                function: FunctionName { name: name.clone() },
            })
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
                let any_non_tool_result = content
                    .iter()
                    .any(|b| !matches!(b, ContentBlock::ToolResult { .. }));
                if any_non_tool_result {
                    let Some(user_content) = render_user_content(content) else {
                        continue;
                    };
                    let m = ChatCompletionRequestUserMessageArgs::default()
                        .content(user_content)
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

fn render_user_content(blocks: &[ContentBlock]) -> Option<ChatCompletionRequestUserMessageContent> {
    if !blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }))
    {
        let text = content_to_plain_text(blocks);
        return if text.is_empty() {
            None
        } else {
            Some(ChatCompletionRequestUserMessageContent::Text(text))
        };
    }

    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => {
                if !text.is_empty() {
                    parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText { text: text.clone() },
                    ));
                }
            }
            ContentBlock::Thinking { thinking, .. } => {
                if !thinking.is_empty() {
                    parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                        ChatCompletionRequestMessageContentPartText {
                            text: thinking.clone(),
                        },
                    ));
                }
            }
            ContentBlock::Image { source } => {
                if let Some(url) = image_source_to_openai_url(source) {
                    parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                        ChatCompletionRequestMessageContentPartImage {
                            image_url: ImageUrl { url, detail: None },
                        },
                    ));
                }
            }
            ContentBlock::Document { .. } => {
                parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: "[document attachment]".into(),
                    },
                ));
            }
            ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => {}
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(ChatCompletionRequestUserMessageContent::Array(parts))
    }
}

fn image_source_to_openai_url(source: &crate::message::ImageSource) -> Option<String> {
    match source {
        crate::message::ImageSource::Base64 { media_type, data } => {
            Some(format!("data:{media_type};base64,{data}"))
        }
        crate::message::ImageSource::Url { url } => Some(url.clone()),
        crate::message::ImageSource::File { .. } => None,
    }
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
            // Make the presence of a non-text block visible so the
            // model isn't confused by an empty user message when
            // the host attached an image / document. OpenAI's
            // chat-completions API does support multimodal content
            // arrays, but our renderer falls back to a plain-text
            // join for now; surface a marker rather than silently
            // dropping.
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
    fn capabilities_advertise_tool_use() {
        // tools are wired into the request builder via render_tools,
        // so supports_tool_use is on. supports_thinking stays off
        // because async-openai 0.36 doesn't surface DeepSeek's
        // `reasoning_content` field — flipping that on before the code
        // path exists would mislead capability-gated callers.
        let p = OpenAiCompatProvider::new(OpenAiCompatConfig::openai("k"));
        assert!(p.capabilities().supports_tool_use);
        assert!(!p.capabilities().supports_thinking);
        assert!(!p.capabilities().supports_images);
        let p =
            OpenAiCompatProvider::new(OpenAiCompatConfig::openai("k").with_supports_images(true));
        assert!(p.capabilities().supports_images);
        let p = OpenAiCompatProvider::new(OpenAiCompatConfig::deepseek("k"));
        assert!(p.capabilities().supports_tool_use);
        assert!(!p.capabilities().supports_thinking);
    }

    #[test]
    fn render_tools_serializes_to_function_shape() {
        let defs = vec![ToolDefinition::new(
            "calc",
            "perform arithmetic",
            serde_json::json!({"type": "object", "properties": {"a": {"type": "number"}}}),
        )];
        let rendered = render_tools(&defs);
        assert_eq!(rendered.len(), 1);
        let json = serde_json::to_value(&rendered[0]).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "calc");
        assert_eq!(json["function"]["description"], "perform arithmetic");
        assert_eq!(json["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn render_tools_omits_empty_description() {
        // OpenAI's spec allows description to be absent — sending an
        // empty string would be valid but noisy. Confirm we drop it.
        let defs = vec![ToolDefinition::new(
            "noop",
            "",
            serde_json::json!({"type": "object"}),
        )];
        let rendered = render_tools(&defs);
        let json = serde_json::to_value(&rendered[0]).unwrap();
        assert!(json["function"].get("description").is_none());
    }

    #[test]
    fn render_tool_choice_modes() {
        let auto = serde_json::to_value(render_tool_choice(&ToolChoice::Auto)).unwrap();
        assert_eq!(auto, serde_json::json!("auto"));
        let required = serde_json::to_value(render_tool_choice(&ToolChoice::Required)).unwrap();
        assert_eq!(required, serde_json::json!("required"));
        let none = serde_json::to_value(render_tool_choice(&ToolChoice::None)).unwrap();
        assert_eq!(none, serde_json::json!("none"));
    }

    #[test]
    fn render_tool_choice_named_tool() {
        let v = serde_json::to_value(render_tool_choice(&ToolChoice::Tool("calc".into()))).unwrap();
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], "calc");
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
    fn render_messages_encodes_user_image_parts() {
        let messages = vec![Message::User {
            header: Header::new(),
            content: vec![
                ContentBlock::Text {
                    text: "describe".into(),
                },
                ContentBlock::Image {
                    source: crate::message::ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: "abc123".into(),
                    },
                },
            ],
        }];
        let rendered = render_messages(&None, &messages).unwrap();
        let json = serde_json::to_value(&rendered[0]).unwrap();
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "describe");
        assert_eq!(json["content"][1]["type"], "image_url");
        assert_eq!(
            json["content"][1]["image_url"]["url"],
            "data:image/png;base64,abc123"
        );
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
    fn content_to_plain_text_surfaces_image_and_document_placeholders() {
        // Codex round-2 finding (f): silent dropping was the wrong
        // failure mode. Multimodal blocks now show up as visible
        // markers in the plain-text fallback so the model isn't
        // surprised by a "missing" attachment.
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
