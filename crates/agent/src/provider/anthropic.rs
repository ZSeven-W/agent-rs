//! Hand-rolled Anthropic Messages API provider (Phase 2 / Task 2.1).
//!
//! Per `notes/2026-05-01-sdk-maturity-research.md`, no Rust crate covers
//! `prompt-caching` + `extended-thinking` + `messages-batches` together,
//! so we build directly on `reqwest` + `eventsource-stream`. ~400-600
//! LOC of focused work that we own.
//!
//! Surface: [`AnthropicProvider`] implements [`crate::provider::Provider`].
//! Feature-gated behind `anthropic`.

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::channel::mpsc;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, ImageSource, Message, ToolResultContent};
use crate::provider::{Provider, ProviderCapabilities, StreamRequest, ToolChoice, ToolDefinition};
use crate::stream::{Event, EventStream, ResultData};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const PROMPT_CACHING_BETA: &str = "prompt-caching-2024-07-31";
const EXTENDED_THINKING_BETA: &str = "extended-thinking-2025-05-01";

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    capabilities: ProviderCapabilities,
}

impl AnthropicProvider {
    /// Construct with the given API key. Uses the default `api.anthropic.com`
    /// base URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_client(api_key, reqwest::Client::new())
    }

    pub fn with_client(api_key: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.into(),
            client,
            capabilities: ProviderCapabilities {
                supports_tool_use: true,
                supports_prompt_caching: true,
                supports_thinking: true,
                max_context_tokens: 200_000,
                needs_placeholder_text_before_tool_use: false,
            },
        }
    }

    /// Override the base URL — useful for mock servers in tests or for
    /// proxying through a corporate gateway.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn id(&self) -> &str {
        "anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }

    async fn stream(
        &self,
        req: StreamRequest,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        let body = build_request_body(&req);
        let url = format!("{}/v1/messages", self.base_url);

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).map_err(|e| {
                AgentError::provider("anthropic", format!("invalid x-api-key header: {e}"))
            })?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        let mut betas: Vec<&'static str> = Vec::new();
        if req.use_prompt_cache {
            betas.push(PROMPT_CACHING_BETA);
        }
        if req.thinking.is_some() {
            betas.push(EXTENDED_THINKING_BETA);
        }
        if !betas.is_empty() {
            headers.insert(
                "anthropic-beta",
                HeaderValue::from_str(&betas.join(",")).map_err(|e| {
                    AgentError::provider("anthropic", format!("invalid anthropic-beta header: {e}"))
                })?,
            );
        }

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::provider("anthropic", e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::provider(
                "anthropic",
                format!("HTTP {status}: {body}"),
            ));
        }

        let bytes = response.bytes_stream();
        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();

        tokio::spawn(parse_sse_into_events(bytes, tx, abort));

        Ok(Box::new(rx))
    }
}

/// Build the JSON body for `POST /v1/messages`.
fn build_request_body(req: &StreamRequest) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": req.model,
        "max_tokens": req.max_output_tokens,
        "stream": true,
        "messages": render_messages(&req.messages, req.use_prompt_cache),
    });

    if let Some(system) = &req.system {
        body["system"] = render_system(system, req.use_prompt_cache);
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if !req.stop_sequences.is_empty() {
        body["stop_sequences"] = serde_json::json!(req.stop_sequences);
    }
    if let Some(thinking) = req.thinking {
        body["thinking"] = serde_json::json!({
            "type": "enabled",
            "budget_tokens": thinking.max_tokens,
        });
    }
    if !req.tools.is_empty() {
        body["tools"] = render_tools(&req.tools);
    }
    if let Some(choice) = &req.tool_choice {
        // Anthropic's API only accepts `tool_choice` when at least one
        // tool is supplied. Sending it standalone is a 400 — silently
        // omit instead.
        if !req.tools.is_empty() {
            body["tool_choice"] = render_tool_choice(choice);
        }
    }

    body
}

fn render_tools(tools: &[ToolDefinition]) -> serde_json::Value {
    serde_json::Value::Array(
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect(),
    )
}

fn render_tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Tool(name) => serde_json::json!({"type": "tool", "name": name}),
    }
}

fn render_system(system: &str, cache: bool) -> serde_json::Value {
    if cache {
        // Array form supports cache_control. Mark the system prompt as
        // ephemeral so the provider caches it across turns.
        serde_json::json!([{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }])
    } else {
        serde_json::json!(system)
    }
}

fn render_messages(messages: &[Message], cache: bool) -> Vec<serde_json::Value> {
    let len = messages.len();
    messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let role = match m {
                Message::User { .. } => "user",
                Message::Assistant { .. } => "assistant",
                // System / Progress / Tombstone are agent-internal — skip.
                _ => return None,
            };
            // Mark only the last user message with cache_control if caching.
            let mark_cache = cache && i + 1 == len && matches!(m, Message::User { .. });
            let content = match m {
                Message::User { content, .. } | Message::Assistant { content, .. } => {
                    render_content(content, mark_cache)
                }
                _ => return None,
            };
            Some(serde_json::json!({"role": role, "content": content}))
        })
        .collect()
}

fn render_content(blocks: &[ContentBlock], cache_last: bool) -> Vec<serde_json::Value> {
    let len = blocks.len();
    blocks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut v = render_block(b);
            if cache_last && i + 1 == len {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert(
                        "cache_control".into(),
                        serde_json::json!({"type": "ephemeral"}),
                    );
                }
            }
            v
        })
        .collect()
}

fn render_block(b: &ContentBlock) -> serde_json::Value {
    match b {
        ContentBlock::Text { text } => serde_json::json!({"type": "text", "text": text}),
        ContentBlock::Image { source } => match source {
            ImageSource::Base64 { media_type, data } => serde_json::json!({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }),
            ImageSource::Url { url } => serde_json::json!({
                "type": "image",
                "source": {"type": "url", "url": url},
            }),
        },
        ContentBlock::ToolUse { id, name, input } => serde_json::json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
        }),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let inner = match content {
                ToolResultContent::Text(t) => serde_json::json!(t),
                ToolResultContent::Blocks(bs) => {
                    // Anthropic's tool_result content array only accepts
                    // `text` and `image` blocks. Filter anything else
                    // (tool_use / tool_result / thinking would be rejected
                    // by the API) so we degrade gracefully instead of
                    // sending an invalid payload.
                    let filtered: Vec<serde_json::Value> = bs
                        .iter()
                        .filter(|b| {
                            matches!(b, ContentBlock::Text { .. } | ContentBlock::Image { .. })
                        })
                        .map(render_block)
                        .collect();
                    serde_json::json!(filtered)
                }
            };
            let mut obj = serde_json::json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": inner,
            });
            if *is_error {
                obj["is_error"] = serde_json::json!(true);
            }
            obj
        }
        ContentBlock::Thinking {
            thinking,
            signature,
        } => {
            let mut obj = serde_json::json!({"type": "thinking", "thinking": thinking});
            if let Some(sig) = signature {
                obj["signature"] = serde_json::json!(sig);
            }
            obj
        }
    }
}

#[derive(Debug, Clone)]
enum BlockState {
    Text,
    Thinking,
    ToolUse {
        id: String,
        name: String,
        partial_json: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStart },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: ContentBlockStart,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: BlockDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        #[serde(default)]
        usage: Option<UsageDelta>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: AnthropicError },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct MessageStart {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<UsageDelta>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)] // serde reads these for forward-compat; runtime uses some.
enum ContentBlockStart {
    #[serde(rename = "text")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "thinking")]
    Thinking,
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        /// Initial input — usually empty `{}`; the real input is streamed
        /// via `input_json_delta` events. Captured for forward-compat /
        /// debugging.
        #[serde(default)]
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum BlockDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct UsageDelta {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

async fn parse_sse_into_events<S>(
    byte_stream: S,
    tx: mpsc::UnboundedSender<Result<Event, AgentError>>,
    abort: AbortController,
) where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
{
    let mut sse = byte_stream.eventsource();
    let mut blocks: HashMap<u32, BlockState> = HashMap::new();
    let mut model: Option<String> = None;
    let mut stop_reason: Option<String> = None;
    let mut total_usage = AccumulatedUsage::default();

    loop {
        let item = tokio::select! {
            _ = abort.cancelled() => {
                let _ = tx.unbounded_send(Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "aborted".into()),
                )));
                return;
            }
            item = sse.next() => item,
        };

        let Some(item) = item else { break };
        let raw = match item {
            Ok(e) => e,
            Err(err) => {
                let _ = tx.unbounded_send(Err(AgentError::provider("anthropic", err.to_string())));
                return;
            }
        };

        let data = raw.data;
        if data.is_empty() {
            continue;
        }

        let parsed: SseEvent = match serde_json::from_str(&data) {
            Ok(p) => p,
            Err(err) => {
                let _ = tx.unbounded_send(Err(AgentError::provider(
                    "anthropic",
                    format!("SSE parse error: {err} (data: {data})"),
                )));
                return;
            }
        };

        match parsed {
            SseEvent::MessageStart { message } => {
                model = message.model.or(model);
                if let Some(u) = message.usage {
                    total_usage.merge(&u);
                }
            }
            SseEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ContentBlockStart::Text { .. } => {
                    blocks.insert(index, BlockState::Text);
                }
                ContentBlockStart::Thinking => {
                    blocks.insert(index, BlockState::Thinking);
                }
                ContentBlockStart::ToolUse { id, name, .. } => {
                    blocks.insert(
                        index,
                        BlockState::ToolUse {
                            id,
                            name,
                            partial_json: String::new(),
                        },
                    );
                }
                ContentBlockStart::Other => {}
            },
            SseEvent::ContentBlockDelta { index, delta } => match (blocks.get_mut(&index), delta) {
                (_, BlockDelta::Text { text }) => {
                    let _ = tx.unbounded_send(Ok(Event::TextDelta { delta: text }));
                }
                (_, BlockDelta::Thinking { thinking }) => {
                    let _ = tx.unbounded_send(Ok(Event::Thinking { delta: thinking }));
                }
                (
                    Some(BlockState::ToolUse { partial_json, .. }),
                    BlockDelta::InputJson {
                        partial_json: chunk,
                    },
                ) => {
                    partial_json.push_str(&chunk);
                }
                _ => {}
            },
            SseEvent::ContentBlockStop { index } => {
                if let Some(BlockState::ToolUse {
                    id,
                    name,
                    partial_json,
                }) = blocks.remove(&index)
                {
                    let input: serde_json::Value = if partial_json.is_empty() {
                        serde_json::Value::Object(Default::default())
                    } else {
                        match serde_json::from_str(&partial_json) {
                            Ok(v) => v,
                            Err(err) => {
                                // Tool-call input is malformed — the
                                // turn cannot complete soundly. Surface
                                // the error and terminate the stream
                                // so the consumer doesn't see a
                                // synthetic Result for a turn whose
                                // tool call was silently dropped.
                                let _ = tx.unbounded_send(Err(AgentError::provider(
                                        "anthropic",
                                        format!(
                                            "tool_use input JSON parse error (id={id}, name={name}): {err}"
                                        ),
                                    )));
                                return;
                            }
                        }
                    };
                    let _ = tx.unbounded_send(Ok(Event::ToolUse { id, name, input }));
                }
            }
            SseEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason {
                    stop_reason = Some(reason);
                }
                if let Some(u) = usage {
                    total_usage.merge(&u);
                    let _ = tx.unbounded_send(Ok(Event::Usage {
                        input_tokens: total_usage.input_tokens,
                        output_tokens: total_usage.output_tokens,
                        cache_read: total_usage.cache_read,
                        cache_create: total_usage.cache_create,
                    }));
                }
            }
            SseEvent::MessageStop => {
                let _ = tx.unbounded_send(Ok(Event::Result {
                    data: ResultData {
                        stop_reason: stop_reason.clone(),
                        model: model.clone(),
                        metadata: Default::default(),
                    },
                }));
            }
            SseEvent::Ping | SseEvent::Other => {}
            SseEvent::Error { error } => {
                let _ = tx.unbounded_send(Ok(Event::Error {
                    code: error.kind,
                    message: error.message,
                }));
            }
        }
    }
}

#[derive(Debug, Default)]
struct AccumulatedUsage {
    input_tokens: u32,
    output_tokens: u32,
    cache_read: u32,
    cache_create: u32,
}

impl AccumulatedUsage {
    fn merge(&mut self, u: &UsageDelta) {
        if let Some(v) = u.input_tokens {
            self.input_tokens = self.input_tokens.max(v);
        }
        if let Some(v) = u.output_tokens {
            self.output_tokens = self.output_tokens.max(v);
        }
        if let Some(v) = u.cache_read_input_tokens {
            self.cache_read = self.cache_read.max(v);
        }
        if let Some(v) = u.cache_creation_input_tokens {
            self.cache_create = self.cache_create.max(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Header, Message};
    use crate::provider::ThinkingConfig;

    #[test]
    fn build_body_minimal() {
        let req = StreamRequest::new(
            "claude-opus-4-7",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
            }],
        );
        let body = build_request_body(&req);
        assert_eq!(body["model"], "claude-opus-4-7");
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["stream"], true);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn build_body_with_system_and_thinking() {
        let req = StreamRequest::new("m", vec![])
            .with_system("you are concise")
            .with_thinking(ThinkingConfig::new(2000));
        let body = build_request_body(&req);
        assert_eq!(body["system"], serde_json::json!("you are concise"));
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 2000);
    }

    #[test]
    fn build_body_with_prompt_cache_marks_system() {
        let req = StreamRequest::new(
            "m",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
            }],
        )
        .with_system("sys")
        .with_prompt_cache(true);
        let body = build_request_body(&req);
        // System rendered as array form when caching.
        assert!(body["system"].is_array());
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        // Last user message's last content block also marked ephemeral.
        let last_user = &body["messages"][0]["content"].as_array().unwrap()[0];
        assert_eq!(last_user["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn build_body_with_tools_and_choice() {
        let req = StreamRequest::new("m", vec![])
            .with_tools(vec![ToolDefinition::new(
                "calc",
                "perform arithmetic",
                serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
            )])
            .with_tool_choice(ToolChoice::Tool("calc".into()));
        let body = build_request_body(&req);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "calc");
        assert_eq!(tools[0]["description"], "perform arithmetic");
        assert_eq!(tools[0]["input_schema"]["type"], "object");
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "calc");
    }

    #[test]
    fn build_body_tool_choice_variants() {
        let mk = |c: ToolChoice| {
            StreamRequest::new("m", vec![])
                .with_tools(vec![ToolDefinition::new(
                    "t",
                    "d",
                    serde_json::json!({"type": "object"}),
                )])
                .with_tool_choice(c)
        };
        assert_eq!(
            build_request_body(&mk(ToolChoice::Auto))["tool_choice"]["type"],
            "auto"
        );
        assert_eq!(
            build_request_body(&mk(ToolChoice::Required))["tool_choice"]["type"],
            "any"
        );
        assert_eq!(
            build_request_body(&mk(ToolChoice::None))["tool_choice"]["type"],
            "none"
        );
    }

    #[test]
    fn build_body_omits_tool_choice_when_no_tools() {
        // Anthropic 400s if tool_choice is sent without tools — make sure
        // we drop it instead.
        let req = StreamRequest::new("m", vec![]).with_tool_choice(ToolChoice::Auto);
        let body = build_request_body(&req);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn build_body_omits_tools_when_empty() {
        let req = StreamRequest::new("m", vec![]);
        let body = build_request_body(&req);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn render_block_tool_result_with_blocks() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "tu_1".into(),
            content: ToolResultContent::Blocks(vec![ContentBlock::Text { text: "ok".into() }]),
            is_error: false,
        };
        let v = render_block(&block);
        assert_eq!(v["type"], "tool_result");
        assert_eq!(v["tool_use_id"], "tu_1");
        assert_eq!(v["content"][0]["type"], "text");
        assert!(v.get("is_error").is_none());
    }

    #[test]
    fn render_block_tool_result_with_error() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "tu_1".into(),
            content: ToolResultContent::Text("boom".into()),
            is_error: true,
        };
        let v = render_block(&block);
        assert_eq!(v["is_error"], true);
        assert_eq!(v["content"], "boom");
    }

    #[test]
    fn provider_id_and_capabilities() {
        let p = AnthropicProvider::new("test-key");
        assert_eq!(p.id(), "anthropic");
        let caps = p.capabilities();
        assert!(caps.supports_tool_use);
        assert!(caps.supports_prompt_caching);
        assert!(caps.supports_thinking);
        assert_eq!(caps.max_context_tokens, 200_000);
    }

    #[test]
    fn sse_parse_message_stop() {
        let raw = r#"{"type":"message_stop"}"#;
        let p: SseEvent = serde_json::from_str(raw).unwrap();
        assert!(matches!(p, SseEvent::MessageStop));
    }

    #[test]
    fn sse_parse_text_delta() {
        let raw =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
        let p: SseEvent = serde_json::from_str(raw).unwrap();
        match p {
            SseEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert!(matches!(delta, BlockDelta::Text { ref text } if text == "hi"));
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn sse_parse_unknown_event_falls_back_to_other() {
        let raw = r#"{"type":"future_event_type_v3"}"#;
        let p: SseEvent = serde_json::from_str(raw).unwrap();
        assert!(matches!(p, SseEvent::Other));
    }

    #[test]
    fn sse_parse_tool_use_block_start() {
        let raw = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"calc","input":{}}}"#;
        let p: SseEvent = serde_json::from_str(raw).unwrap();
        match p {
            SseEvent::ContentBlockStart {
                index,
                content_block: ContentBlockStart::ToolUse { id, name, .. },
            } => {
                assert_eq!(index, 1);
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "calc");
            }
            _ => panic!("expected ToolUse content block"),
        }
    }

    #[test]
    fn render_block_filters_non_text_image_inside_tool_result() {
        // Anthropic only accepts text/image inside tool_result content;
        // anything else (tool_use, tool_result, thinking) must be dropped
        // — otherwise the API rejects the request.
        let block = ContentBlock::ToolResult {
            tool_use_id: "tu_1".into(),
            content: ToolResultContent::Blocks(vec![
                ContentBlock::Text { text: "ok".into() },
                ContentBlock::ToolUse {
                    id: "tu_inner".into(),
                    name: "nested".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Thinking {
                    thinking: "hmm".into(),
                    signature: None,
                },
                ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/x.png".into(),
                    },
                },
            ]),
            is_error: false,
        };
        let v = render_block(&block);
        let arr = v["content"].as_array().unwrap();
        // Only the Text and Image blocks survive.
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "image");
    }

    /// Real-API integration test, gated by ANTHROPIC_API_KEY env var.
    /// Run manually with `cargo test --features anthropic anthropic_real_api -- --ignored --nocapture`.
    /// Skipped in normal `cargo test` runs and CI without an explicit key.
    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY env var; hits real Anthropic API"]
    async fn anthropic_real_api_hello_world() {
        let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
            eprintln!("ANTHROPIC_API_KEY not set; skipping real-API test");
            return;
        };
        let provider = AnthropicProvider::new(api_key);
        let req = StreamRequest::new(
            "claude-haiku-4-5-20251001",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text {
                    text: "Reply with exactly the single word: ok".into(),
                }],
            }],
        )
        .with_max_output_tokens(64);

        let abort = AbortController::new();
        let mut stream = provider.stream(req, abort).await.expect("provider stream");

        use futures::StreamExt;
        let mut got_text = String::new();
        let mut got_result = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                Event::TextDelta { delta } => got_text.push_str(&delta),
                Event::Result { .. } => got_result = true,
                _ => {}
            }
        }
        assert!(got_result, "stream should end with Result event");
        assert!(
            !got_text.is_empty(),
            "expected non-empty assistant text, got nothing"
        );
    }
}
