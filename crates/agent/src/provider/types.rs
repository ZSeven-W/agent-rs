use async_trait::async_trait;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::Message;
use crate::stream::EventStream;

/// Pluggable LLM provider.
///
/// Implementations live in `crate::provider::anthropic`, `::openai_compat`,
/// `::ollama` (Phase 2 / 5). Consumers usually hold `Box<dyn Provider>` so
/// the same QueryEngine can swap between providers at runtime.
///
/// `Debug` is required because the crate-wide `missing_debug_implementations`
/// lint enforces it for tracing / panic diagnostics. Providers holding
/// opaque internals (closures, raw HTTP client futures) should derive a
/// manual `Debug` that redacts the sensitive parts (e.g., API keys).
#[async_trait]
pub trait Provider: Send + Sync + std::fmt::Debug {
    /// Stable id (e.g., `"anthropic"`, `"openai"`, `"ollama"`,
    /// `"openai-compat:deepseek"`).
    fn id(&self) -> &str;

    /// What this provider can do. Stable across the lifetime of one
    /// `Provider` instance — the QueryEngine reads this once at session
    /// start.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Open a streaming request. Returns a boxed [`EventStream`] that yields
    /// `Event` items until the turn completes (or `abort` fires).
    ///
    /// Implementors should `tokio::select!` on `abort.cancelled()` and the
    /// underlying network read so cancellation is prompt.
    async fn stream(
        &self,
        req: StreamRequest,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError>;
}

/// Static capability flags. New flags are added with `#[non_exhaustive]` so
/// consumers must use the constructor (or `..Default::default()`) and won't
/// break when a future flag is introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProviderCapabilities {
    pub supports_tool_use: bool,
    pub supports_prompt_caching: bool,
    pub supports_thinking: bool,
    pub supports_images: bool,
    pub max_context_tokens: u32,
    /// Some providers (older Anthropic models) require a non-empty text
    /// block immediately before a `tool_use` block in the assistant
    /// response. The QueryEngine inserts a zero-width-space placeholder
    /// when this is set.
    pub needs_placeholder_text_before_tool_use: bool,
    /// Whether this provider's tool_result rendering preserves image
    /// blocks (rather than flattening ToolResultContent::Blocks to
    /// plain text). Gates rich tool-result emission in the query loop.
    pub tool_result_images: bool,
}

impl Default for ProviderCapabilities {
    /// Conservative defaults: every optional capability is off; context
    /// window is `128_000` (the Anthropic Claude 3.5 / 4 baseline,
    /// chosen so the QueryEngine has a reasonable lower bound when a
    /// provider impl forgets to override). Provider impls SHOULD return
    /// their actual capability set; the default exists so test mocks
    /// can construct a `ProviderCapabilities` ergonomically.
    fn default() -> Self {
        Self {
            supports_tool_use: false,
            supports_prompt_caching: false,
            supports_thinking: false,
            supports_images: false,
            max_context_tokens: 128_000,
            needs_placeholder_text_before_tool_use: false,
            tool_result_images: false,
        }
    }
}

/// One LLM call.
///
/// `#[non_exhaustive]` keeps the door open for cache_control config,
/// stop tokens, top_p, top_k, etc. without breaking impls.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub system: Option<String>,
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    pub stop_sequences: Vec<String>,
    pub thinking: Option<ThinkingConfig>,
    /// Hint to attach `cache_control` markers (Anthropic-specific; other
    /// providers ignore). The actual placement is provider-implementation
    /// detail (typically last system prompt + last user message).
    pub use_prompt_cache: bool,
    /// Tool definitions exposed to the model. Empty means the model
    /// cannot call tools on this turn. Provider impls translate these
    /// into their wire-level shape (Anthropic `tools`, OpenAI
    /// `ChatCompletionTools::Function`, Ollama `ToolInfo`).
    pub tools: Vec<ToolDefinition>,
    /// Steers tool calling behavior. `None` → provider default (which is
    /// usually "auto when tools are present, none otherwise").
    pub tool_choice: Option<ToolChoice>,
}

impl StreamRequest {
    /// Convenience constructor — the absolute minimum required fields.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            system: None,
            max_output_tokens: 4096,
            temperature: None,
            stop_sequences: Vec::new(),
            thinking: None,
            use_prompt_cache: false,
            tools: Vec::new(),
            tool_choice: None,
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn with_thinking(mut self, cfg: ThinkingConfig) -> Self {
        self.thinking = Some(cfg);
        self
    }

    pub fn with_prompt_cache(mut self, enabled: bool) -> Self {
        self.use_prompt_cache = enabled;
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }
}

/// Provider-neutral tool definition.
///
/// `name` and `description` are forwarded verbatim. `input_schema` is a
/// JSON Schema (draft 2020-12 recommended) — the same shape returned by
/// [`crate::tool::Tool::input_schema`]. Providers that demand draft-07
/// (e.g., Ollama via schemars 1.x) accept the same JSON object form.
///
/// Stable serialization is important: provider impls hash this struct
/// into [`crate::api::PromptCacheState::tool_schema_hash`] so any change
/// breaks the cache deterministically.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Construct a tool definition. **Does not validate** that
    /// `input_schema` is a JSON object — providers vary in strictness
    /// (Ollama rejects non-object schemas locally; Anthropic and OpenAI
    /// only fail at the wire level). Use [`Self::try_new`] for
    /// up-front validation.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }

    /// Validating constructor. Returns `Err` if `input_schema` is not a
    /// JSON object. Anthropic and OpenAI's function-calling protocols
    /// both require an object schema; Ollama enforces it at the
    /// schemars layer. Catching the mismatch up-front gives a clearer
    /// error than a provider HTTP 400.
    pub fn try_new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Result<Self, ToolDefinitionError> {
        if !input_schema.is_object() {
            return Err(ToolDefinitionError::SchemaMustBeObject {
                got: schema_kind(&input_schema),
            });
        }
        Ok(Self::new(name, description, input_schema))
    }
}

fn schema_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Errors from [`ToolDefinition::try_new`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolDefinitionError {
    #[error("tool input_schema must be a JSON object, got {got}")]
    SchemaMustBeObject { got: &'static str },
}

/// Steers the model's tool-calling decision for this turn.
///
/// Each provider maps these onto its own surface:
///
/// | Variant   | Anthropic         | OpenAI/compat | Ollama |
/// |-----------|-------------------|---------------|--------|
/// | `Auto`    | `{type:"auto"}`   | `auto`        | default (no tool_choice) |
/// | `Required`| `{type:"any"}`    | `required`    | n/a (silently dropped) |
/// | `None`    | `{type:"none"}`   | `none`        | drop `tools` from request |
/// | `Tool(n)` | `{type:"tool",name:n}` | `{type:"function",function:{name:n}}` | n/a (logs warning) |
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolChoice {
    /// Model decides whether to call a tool.
    Auto,
    /// Model MUST call exactly one tool (provider chooses which).
    Required,
    /// Model must NOT call any tool.
    None,
    /// Model must call this specific tool by name.
    Tool(String),
}

/// Configuration for Anthropic-style "extended thinking". Other providers
/// ignore this field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ThinkingConfig {
    /// Maximum tokens to spend on thinking before producing the user-visible
    /// response. Provider may impose its own ceiling.
    pub max_tokens: u32,
}

impl ThinkingConfig {
    pub fn new(max_tokens: u32) -> Self {
        Self { max_tokens }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Header, Message};
    use crate::stream::Event;
    use futures::stream;

    #[derive(Debug)]
    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        fn id(&self) -> &str {
            "mock"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_tool_use: true,
                supports_prompt_caching: true,
                supports_thinking: false,
                supports_images: false,
                max_context_tokens: 200_000,
                needs_placeholder_text_before_tool_use: false,
                tool_result_images: false,
            }
        }

        async fn stream(
            &self,
            _req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            Ok(Box::new(stream::iter(vec![Ok(Event::TextDelta {
                delta: "hi".into(),
            })])))
        }
    }

    #[test]
    fn capabilities_default_is_conservative() {
        let c = ProviderCapabilities::default();
        assert!(!c.supports_tool_use);
        assert!(!c.supports_prompt_caching);
        assert!(!c.supports_thinking);
        assert!(!c.supports_images);
        assert!(!c.needs_placeholder_text_before_tool_use);
        assert_eq!(c.max_context_tokens, 128_000);
    }

    #[test]
    fn tool_result_images_defaults_false() {
        assert!(!ProviderCapabilities::default().tool_result_images);
    }

    #[test]
    fn stream_request_builder() {
        let user = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: "hi".into() }],
        };
        let req = StreamRequest::new("claude-opus-4-7", vec![user])
            .with_system("be concise")
            .with_max_output_tokens(1024)
            .with_temperature(0.7)
            .with_thinking(ThinkingConfig::new(2000))
            .with_prompt_cache(true);
        assert_eq!(req.model, "claude-opus-4-7");
        assert_eq!(req.system.as_deref(), Some("be concise"));
        assert_eq!(req.max_output_tokens, 1024);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.thinking.map(|t| t.max_tokens), Some(2000));
        assert!(req.use_prompt_cache);
        assert!(req.tools.is_empty());
        assert!(req.tool_choice.is_none());
    }

    #[test]
    fn tool_definition_try_new_rejects_non_object() {
        for v in [
            serde_json::Value::Null,
            serde_json::json!(true),
            serde_json::json!(42),
            serde_json::json!("string-schema"),
            serde_json::json!([1, 2, 3]),
        ] {
            let err = ToolDefinition::try_new("t", "d", v).expect_err("should reject");
            assert!(matches!(
                err,
                ToolDefinitionError::SchemaMustBeObject { .. }
            ));
        }
    }

    #[test]
    fn tool_definition_try_new_accepts_object() {
        let ok = ToolDefinition::try_new("t", "d", serde_json::json!({"type": "object"})).unwrap();
        assert_eq!(ok.name, "t");
    }

    #[test]
    fn stream_request_builder_with_tools_and_choice() {
        let req = StreamRequest::new("m", vec![])
            .with_tools(vec![ToolDefinition::new(
                "calc",
                "do math",
                serde_json::json!({"type": "object"}),
            )])
            .with_tool_choice(ToolChoice::Tool("calc".into()));
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "calc");
        assert_eq!(req.tool_choice, Some(ToolChoice::Tool("calc".into())));
    }

    #[tokio::test]
    async fn mock_provider_stream() {
        use futures::StreamExt;
        let p: Box<dyn Provider> = Box::new(MockProvider);
        let req = StreamRequest::new("any", vec![]);
        let abort = AbortController::new();
        let mut stream = p.stream(req, abort).await.unwrap();
        let mut count = 0;
        while let Some(item) = stream.next().await {
            assert!(item.is_ok());
            count += 1;
        }
        assert_eq!(count, 1);
        assert_eq!(p.id(), "mock");
        assert!(p.capabilities().supports_tool_use);
    }
}
