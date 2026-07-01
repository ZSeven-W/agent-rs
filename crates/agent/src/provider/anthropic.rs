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
use crate::message::{ContentBlock, DocumentSource, ImageSource, Message, ToolResultContent};
use crate::provider::{Provider, ProviderCapabilities, StreamRequest, ToolChoice, ToolDefinition};
use crate::stream::{Event, EventStream, ResultData};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const PROMPT_CACHING_BETA: &str = "prompt-caching-2024-07-31";
const EXTENDED_THINKING_BETA: &str = "extended-thinking-2025-05-01";
/// Required to reference Files API uploads (`file_id` sources) from
/// Messages requests. We send it whenever any block in the request
/// uses a `File` source; sending it spuriously is harmless on
/// Anthropic's side but adds a header byte.
const FILES_API_BETA: &str = "files-api-2025-04-14";

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
                supports_images: true,
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

    pub fn with_supports_images(mut self, supports_images: bool) -> Self {
        self.capabilities.supports_images = supports_images;
        self
    }

    /// Insert the authentication headers. The official Anthropic API reads
    /// `x-api-key`; third-party Anthropic-compatible gateways (DeepSeek,
    /// LongCat, …) instead read `Authorization: Bearer <key>`. For a custom
    /// base URL we send BOTH, so whichever the endpoint expects is present. The
    /// official endpoint (default base URL) keeps receiving only `x-api-key` —
    /// sending it a bearer token would be read as an OAuth credential.
    fn insert_auth_headers(&self, headers: &mut HeaderMap) -> Result<(), AgentError> {
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).map_err(|e| {
                AgentError::provider("anthropic", format!("invalid x-api-key header: {e}"))
            })?,
        );
        if self.base_url != DEFAULT_BASE_URL {
            headers.insert(
                "authorization",
                HeaderValue::from_str(&format!("Bearer {}", self.api_key)).map_err(|e| {
                    AgentError::provider("anthropic", format!("invalid authorization header: {e}"))
                })?,
            );
        }
        Ok(())
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
        self.insert_auth_headers(&mut headers)?;
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
        if request_uses_file_sources(&req) {
            betas.push(FILES_API_BETA);
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

/// `true` if any content block in the request references a Files
/// API upload (`ImageSource::File` or `DocumentSource::File`),
/// including blocks nested inside `ToolResultContent::Blocks`.
/// Triggers the `anthropic-beta: files-api-...` header — without
/// it Anthropic 400s on `file_id` sources, and we'd silently miss
/// the requirement for tool_results that carry an image file ref.
fn request_uses_file_sources(req: &StreamRequest) -> bool {
    fn block_uses_file(block: &ContentBlock) -> bool {
        match block {
            ContentBlock::Image {
                source: ImageSource::File { .. },
            } => true,
            ContentBlock::Document {
                source: DocumentSource::File { .. },
            } => true,
            ContentBlock::ToolResult { content, .. } => match content {
                ToolResultContent::Blocks(bs) => bs.iter().any(block_uses_file),
                ToolResultContent::Text(_) => false,
            },
            _ => false,
        }
    }
    for m in &req.messages {
        let content = match m {
            Message::User { content, .. } | Message::Assistant { content, .. } => content,
            _ => continue,
        };
        if content.iter().any(block_uses_file) {
            return true;
        }
    }
    false
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

/// Merge consecutive same-role (user/user, assistant/assistant) turns into one.
/// The query loop guarantees a `user(tool_result)` message is pushed after
/// every `assistant(tool_use)`; when a turn is interrupted, the user's next
/// prompt lands as a SECOND consecutive user message. Anthropic requires strict
/// user/assistant alternation, so adjacent same-role turns are folded here.
/// System/Progress/Tombstone are agent-internal (not rendered) and transparent
/// to adjacency — they neither emit a role nor break a same-role run, so an
/// interrupted `user(tool_result)` / `user(prompt)` pair still coalesces across
/// an interleaved tombstone.
fn coalesce_messages(messages: &[Message]) -> Vec<(&'static str, Vec<ContentBlock>)> {
    let mut out: Vec<(&'static str, Vec<ContentBlock>)> = Vec::with_capacity(messages.len());
    for m in messages {
        let (role, content) = match m {
            Message::User { content, .. } => ("user", content),
            Message::Assistant { content, .. } => ("assistant", content),
            _ => continue,
        };
        match out.last_mut() {
            Some((prev_role, prev_content)) if *prev_role == role => {
                prev_content.extend(content.iter().cloned());
            }
            _ => out.push((role, content.clone())),
        }
    }
    // Anthropic requires the first message to be `user`. A leading assistant
    // turn can't be answered by anything before it and would be rejected. It
    // never arises in normal flow (the store always opens with the user's
    // prompt), but guard defensively so an odd/partial history still yields a
    // valid request.
    while matches!(out.first(), Some((role, _)) if *role == "assistant") {
        out.remove(0);
    }
    out
}

fn render_messages(messages: &[Message], cache: bool) -> Vec<serde_json::Value> {
    let coalesced = coalesce_messages(messages);
    let len = coalesced.len();
    coalesced
        .iter()
        .enumerate()
        .map(|(i, (role, content))| {
            // Mark only the final user message's final block with cache_control.
            let mark_cache = cache && i + 1 == len && *role == "user";
            serde_json::json!({"role": role, "content": render_content(content, mark_cache)})
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
            ImageSource::File { file_id } => serde_json::json!({
                "type": "image",
                "source": {"type": "file", "file_id": file_id},
            }),
        },
        ContentBlock::Document { source } => match source {
            DocumentSource::Base64 { media_type, data } => serde_json::json!({
                "type": "document",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }),
            DocumentSource::Url { url } => serde_json::json!({
                "type": "document",
                "source": {"type": "url", "url": url},
            }),
            DocumentSource::File { file_id } => serde_json::json!({
                "type": "document",
                "source": {"type": "file", "file_id": file_id},
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
                    // Anthropic's tool_result content array only
                    // accepts `text` and `image` blocks. Other blocks
                    // (tool_use / tool_result / thinking / document)
                    // would be wire-rejected. Render text/image
                    // blocks as-is; surface other variants as a
                    // visible text placeholder so the model still
                    // sees that something was returned (silent
                    // dropping was the wrong failure mode — the
                    // model would refer to a missing attachment).
                    let mapped: Vec<serde_json::Value> = bs
                        .iter()
                        .map(|b| match b {
                            ContentBlock::Text { .. } | ContentBlock::Image { .. } => {
                                render_block(b)
                            }
                            ContentBlock::Document { .. } => serde_json::json!({
                                "type": "text",
                                "text": "[document attachment elided — Anthropic tool_result blocks accept only text/image]",
                            }),
                            ContentBlock::ToolUse { .. }
                            | ContentBlock::ToolResult { .. }
                            | ContentBlock::Thinking { .. } => serde_json::json!({
                                "type": "text",
                                "text": "[non-text/image block elided from tool_result]",
                            }),
                        })
                        .collect();
                    serde_json::json!(mapped)
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
                                //
                                // The common cause is the model hitting
                                // the output-token limit MID tool call,
                                // which leaves the accumulated JSON
                                // truncated (serde reports an EOF). Give
                                // an actionable message instead of a raw
                                // parser error so the user knows to raise
                                // `max_output_tokens` (or ask for a
                                // smaller change) rather than retry blind.
                                let msg = if err.is_eof() {
                                    format!(
                                        "tool_use `{name}` input was cut off at {} bytes — the model hit \
                                         the output-token limit mid-call. Raise `max_output_tokens` in \
                                         config or ask for a smaller change. (id={id}: {err})",
                                        partial_json.len()
                                    )
                                } else {
                                    format!(
                                        "tool_use input JSON parse error (id={id}, name={name}): {err}"
                                    )
                                };
                                let _ =
                                    tx.unbounded_send(Err(AgentError::provider("anthropic", msg)));
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
    use crate::message::{Header, Message, ToolResultContent};
    use crate::provider::ThinkingConfig;

    #[test]
    fn custom_base_url_sends_bearer_and_x_api_key() {
        // A third-party Anthropic-compatible gateway (LongCat/DeepSeek) reads
        // `Authorization: Bearer`; we send both it AND `x-api-key`.
        let p =
            AnthropicProvider::new("ak_test").with_base_url("https://api.longcat.chat/anthropic");
        let mut h = HeaderMap::new();
        p.insert_auth_headers(&mut h).unwrap();
        assert_eq!(h.get("x-api-key").unwrap(), "ak_test");
        assert_eq!(h.get("authorization").unwrap(), "Bearer ak_test");
    }

    #[test]
    fn official_base_url_sends_only_x_api_key() {
        // The official Anthropic API uses `x-api-key`; a bearer token there would
        // be read as an OAuth credential, so we must NOT send one.
        let p = AnthropicProvider::new("sk-ant-test");
        let mut h = HeaderMap::new();
        p.insert_auth_headers(&mut h).unwrap();
        assert_eq!(h.get("x-api-key").unwrap(), "sk-ant-test");
        assert!(h.get("authorization").is_none());
    }

    #[tokio::test]
    async fn truncated_tool_use_input_reports_token_limit() {
        use futures::StreamExt;
        // A tool_use whose input JSON is cut off mid-string (the model hit the
        // output-token limit) must surface an actionable error pointing at
        // `max_output_tokens`, not a raw serde EOF parse error.
        let start = serde_json::json!({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "tool_use", "id": "t1", "name": "Bash"}
        });
        let delta = serde_json::json!({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"echo "}
        });
        let stop = serde_json::json!({"type": "content_block_stop", "index": 0});
        let sse = format!("data: {start}\n\ndata: {delta}\n\ndata: {stop}\n\n");
        let stream = futures::stream::iter(vec![Ok::<_, reqwest::Error>(bytes::Bytes::from(sse))]);

        let (tx, mut rx) = mpsc::unbounded::<Result<Event, AgentError>>();
        parse_sse_into_events(stream, tx, AbortController::default()).await;

        let mut err_msg = None;
        while let Some(item) = rx.next().await {
            if let Err(e) = item {
                err_msg = Some(e.to_string());
            }
        }
        let msg = err_msg.expect("expected a truncation error event");
        assert!(
            msg.contains("cut off") && msg.contains("max_output_tokens"),
            "error should be actionable about the token limit; got: {msg}"
        );
    }

    #[test]
    fn render_coalesces_consecutive_same_role_messages() {
        // The interrupt shape: assistant(tool_use), then TWO user messages —
        // the synthetic tool_result and the user's next prompt. These must
        // render as one alternating sequence (user / assistant / user), with
        // the final user message carrying both blocks (tool_result first),
        // or Anthropic rejects the request for non-alternating roles.
        let msgs = vec![
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "go".into() }],
            },
            Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "Bash".into(),
                    input: serde_json::json!({}),
                }],
            },
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: ToolResultContent::Text("[interrupted]".into()),
                    is_error: true,
                }],
            },
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "why".into() }],
            },
        ];
        let rendered = render_messages(&msgs, false);
        assert_eq!(rendered.len(), 3, "the two trailing user turns coalesce");
        assert_eq!(rendered[0]["role"], "user");
        assert_eq!(rendered[1]["role"], "assistant");
        assert_eq!(rendered[2]["role"], "user");
        let last = rendered[2]["content"].as_array().unwrap();
        assert_eq!(last.len(), 2, "merged user turn holds tool_result + text");
        assert_eq!(last[0]["type"], "tool_result");
        assert_eq!(last[1]["type"], "text");
    }

    #[test]
    fn render_drops_leading_assistant_to_keep_first_message_user() {
        // Anthropic rejects a request whose first message is `assistant`. A
        // leading assistant turn can't occur in normal flow, but if it does,
        // coalescing drops it so the request stays valid.
        let msgs = vec![
            Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::Text {
                    text: "orphan".into(),
                }],
            },
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
            },
        ];
        let rendered = render_messages(&msgs, false);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0]["role"], "user");
    }

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
        assert!(caps.supports_images);
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
    fn request_uses_file_sources_detects_image_and_document_refs() {
        use crate::message::{ContentBlock, DocumentSource, Header, ImageSource, Message};
        // No file refs.
        let plain = StreamRequest::new(
            "m",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Text { text: "hi".into() }],
            }],
        );
        assert!(!request_uses_file_sources(&plain));
        // Image file ref.
        let img_ref = StreamRequest::new(
            "m",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Image {
                    source: ImageSource::File {
                        file_id: "file_a".into(),
                    },
                }],
            }],
        );
        assert!(request_uses_file_sources(&img_ref));
        // Document file ref.
        let doc_ref = StreamRequest::new(
            "m",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Document {
                    source: DocumentSource::File {
                        file_id: "file_b".into(),
                    },
                }],
            }],
        );
        assert!(request_uses_file_sources(&doc_ref));
        // URL-source images should NOT trigger the beta.
        let url_img = StreamRequest::new(
            "m",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/x.png".into(),
                    },
                }],
            }],
        );
        assert!(!request_uses_file_sources(&url_img));
    }

    #[test]
    fn request_uses_file_sources_recurses_into_tool_result_blocks() {
        // Codex round-2 finding (k): a tool_result returning an
        // image-file-ref must still trigger the Files beta header
        // on the outer Messages request.
        use crate::message::{ContentBlock, Header, ImageSource, Message, ToolResultContent};
        let req = StreamRequest::new(
            "m",
            vec![Message::User {
                header: Header::new(),
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: ToolResultContent::Blocks(vec![ContentBlock::Image {
                        source: ImageSource::File {
                            file_id: "file_inside_tool_result".into(),
                        },
                    }]),
                    is_error: false,
                }],
            }],
        );
        assert!(request_uses_file_sources(&req));
    }

    #[test]
    fn render_block_image_file_ref() {
        let block = ContentBlock::Image {
            source: ImageSource::File {
                file_id: "file_abc".into(),
            },
        };
        let v = render_block(&block);
        assert_eq!(v["type"], "image");
        assert_eq!(v["source"]["type"], "file");
        assert_eq!(v["source"]["file_id"], "file_abc");
    }

    #[test]
    fn render_block_document_variants() {
        // Base64 PDF
        let inline = ContentBlock::Document {
            source: DocumentSource::Base64 {
                media_type: "application/pdf".into(),
                data: "JVBERi0xLjc=".into(),
            },
        };
        let v = render_block(&inline);
        assert_eq!(v["type"], "document");
        assert_eq!(v["source"]["type"], "base64");
        assert_eq!(v["source"]["media_type"], "application/pdf");

        // File-id reference
        let by_ref = ContentBlock::Document {
            source: DocumentSource::File {
                file_id: "file_doc".into(),
            },
        };
        let v = render_block(&by_ref);
        assert_eq!(v["type"], "document");
        assert_eq!(v["source"]["type"], "file");
        assert_eq!(v["source"]["file_id"], "file_doc");

        // URL
        let url = ContentBlock::Document {
            source: DocumentSource::Url {
                url: "https://example.com/x.pdf".into(),
            },
        };
        let v = render_block(&url);
        assert_eq!(v["source"]["type"], "url");
        assert_eq!(v["source"]["url"], "https://example.com/x.pdf");
    }

    #[test]
    fn tool_result_document_block_surfaces_placeholder() {
        // Anthropic wire-rejects non-text/image inside tool_result.
        // Round-1 codex flagged silently dropping Document blocks
        // here — they now produce a visible text placeholder so the
        // model still sees that something was returned.
        let block = ContentBlock::ToolResult {
            tool_use_id: "tu_1".into(),
            content: ToolResultContent::Blocks(vec![
                ContentBlock::Text { text: "ok".into() },
                ContentBlock::Document {
                    source: DocumentSource::File {
                        file_id: "file_doc".into(),
                    },
                },
            ]),
            is_error: false,
        };
        let v = render_block(&block);
        let arr = v["content"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "ok");
        assert_eq!(arr[1]["type"], "text");
        assert!(
            arr[1]["text"]
                .as_str()
                .unwrap()
                .contains("document attachment elided"),
            "got {}",
            arr[1]["text"]
        );
    }

    #[test]
    fn render_block_replaces_unsupported_inside_tool_result_with_placeholder() {
        // Anthropic only accepts text/image inside tool_result content;
        // anything else (tool_use, tool_result, thinking, document)
        // is replaced with a visible text placeholder so the model
        // sees that something existed there. Silent dropping was the
        // wrong failure mode (codex round-1 finding).
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
        // Text + placeholder(ToolUse) + placeholder(Thinking) + Image
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "ok");
        assert_eq!(arr[1]["type"], "text");
        assert!(arr[1]["text"].as_str().unwrap().contains("elided"));
        assert_eq!(arr[2]["type"], "text");
        assert!(arr[2]["text"].as_str().unwrap().contains("elided"));
        assert_eq!(arr[3]["type"], "image");
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
