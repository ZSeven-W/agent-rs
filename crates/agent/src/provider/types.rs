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
    pub max_context_tokens: u32,
    /// Some providers (older Anthropic models) require a non-empty text
    /// block immediately before a `tool_use` block in the assistant
    /// response. The QueryEngine inserts a zero-width-space placeholder
    /// when this is set.
    pub needs_placeholder_text_before_tool_use: bool,
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
            max_context_tokens: 128_000,
            needs_placeholder_text_before_tool_use: false,
        }
    }
}

/// One LLM call. Tool definitions are deliberately absent in Phase 1;
/// they're added when [`crate::tool`] lands in Phase 2.
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
                max_context_tokens: 200_000,
                needs_placeholder_text_before_tool_use: false,
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
        assert!(!c.needs_placeholder_text_before_tool_use);
        assert_eq!(c.max_context_tokens, 128_000);
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
