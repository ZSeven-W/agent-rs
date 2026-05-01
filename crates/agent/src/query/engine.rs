use std::sync::{Arc, Mutex};

use futures::channel::mpsc;
use futures::StreamExt;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, Header, Message, MessageStore};
use crate::provider::{Provider, StreamRequest};
use crate::stream::{Event, EventStream, ResultData};
use crate::tool::ToolRegistry;

/// Glue layer that drives one turn of an LLM conversation.
///
/// **Phase 2 minimal**: pushes the user message, calls the provider,
/// forwards events as-is, and on stream end emits a `Result` event
/// (if the provider didn't already) plus pushes a single Assistant
/// message containing the concatenated text deltas. Tool dispatch is
/// **not** performed in this batch — `Event::ToolUse` is forwarded
/// untouched so callers can see it but no `Event::ToolResult` follows
/// unless the provider produces one.
///
/// Phase 3+ adds:
/// - Tool dispatch via [`ToolRegistry::get`] + receipt-order yielding
/// - 7-step permission chain + external_tool_queue
/// - Hook events around tool execution
/// - Multi-turn loop until `stop_reason` indicates done
#[derive(Debug)]
pub struct QueryEngine {
    store: Arc<Mutex<MessageStore>>,
    provider: Arc<dyn Provider>,
    tools: Arc<ToolRegistry>,
    /// Model identifier passed to the provider for each turn. Provider
    /// impls are responsible for resolving it (e.g., Anthropic accepts
    /// `claude-opus-4-7`, OpenAI accepts `gpt-5.4-codex`, etc.).
    pub model: String,
    /// Optional system prompt prepended to every turn.
    pub system: Option<String>,
    /// Cap on tokens emitted per turn.
    pub max_output_tokens: u32,
}

impl QueryEngine {
    /// Construct an engine with default config (no system prompt, 4096
    /// max output tokens, empty tool registry).
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            store: Arc::new(Mutex::new(MessageStore::new())),
            provider,
            tools: Arc::new(ToolRegistry::new()),
            model: model.into(),
            system: None,
            max_output_tokens: 4096,
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Arc::new(tools);
        self
    }

    pub fn with_max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = n;
        self
    }

    /// Borrow the registry — handy for tests + hot-loading additional tools.
    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    /// Borrow the message store handle.
    pub fn message_store(&self) -> &Arc<Mutex<MessageStore>> {
        &self.store
    }

    /// Clone the full message history out from under the lock. Returns an
    /// independent `Vec<Message>` the caller can iterate without holding
    /// the store mutex.
    ///
    /// Production callers can use this to render a transcript, persist a
    /// session, or feed history into a different engine. Tests use it to
    /// assert on the post-run state. The clone is O(N) in message count.
    pub fn snapshot(&self) -> Result<Vec<Message>, AgentError> {
        let store = self
            .store
            .lock()
            .map_err(|_| AgentError::other("message store lock poisoned"))?;
        Ok(store.iter().cloned().collect())
    }

    /// Run a single turn. Returns a stream of `Event`s; the stream
    /// completes after the provider's stream ends and the engine has
    /// pushed the Assistant message + emitted `Event::Result`.
    pub async fn run(
        &self,
        user_msg: impl Into<String>,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        let user_message = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text {
                text: user_msg.into(),
            }],
        };
        let user_uuid = user_message.uuid();

        let messages_for_request = {
            let mut store = self
                .store
                .lock()
                .map_err(|_| AgentError::other("message store lock poisoned"))?;
            store.push(user_message)?;
            store.iter().cloned().collect::<Vec<_>>()
        };

        let mut req = StreamRequest::new(self.model.clone(), messages_for_request)
            .with_max_output_tokens(self.max_output_tokens);
        if let Some(system) = &self.system {
            req = req.with_system(system.clone());
        }

        let upstream = self.provider.stream(req, abort.clone()).await?;

        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();
        let store = self.store.clone();

        tokio::spawn(forward_turn(upstream, tx, store, user_uuid));

        Ok(Box::new(rx))
    }
}

/// Spawned background task: relay provider events, accumulate text deltas,
/// finalize the turn with assistant message + Result event.
async fn forward_turn(
    mut upstream: Box<dyn EventStream>,
    tx: mpsc::UnboundedSender<Result<Event, AgentError>>,
    store: Arc<Mutex<MessageStore>>,
    parent_uuid: uuid::Uuid,
) {
    let mut accumulated_text = String::new();
    let mut provider_emitted_result = false;

    while let Some(item) = upstream.next().await {
        if let Ok(event) = &item {
            match event {
                Event::TextDelta { delta } => accumulated_text.push_str(delta),
                Event::Result { .. } => provider_emitted_result = true,
                _ => {}
            }
        }
        if tx.unbounded_send(item).is_err() {
            // Receiver dropped — caller stopped polling. Bail out.
            return;
        }
    }

    // Push assistant message reflecting the concatenated text deltas.
    let assistant = Message::Assistant {
        header: Header::child_of(parent_uuid),
        content: vec![ContentBlock::Text {
            text: accumulated_text,
        }],
    };
    if let Ok(mut store_guard) = store.lock() {
        let _ = store_guard.push(assistant);
    }

    // Emit our own Result if the provider didn't already.
    if !provider_emitted_result {
        let _ = tx.unbounded_send(Ok(Event::Result {
            data: ResultData::default(),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::testing::MockProvider;
    use futures::StreamExt;

    fn run_blocking(events: Vec<Event>) -> (Vec<Event>, Vec<Message>) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let provider = Arc::new(MockProvider::new(events));
            let engine = QueryEngine::new(provider, "mock-model");
            let abort = AbortController::new();
            let mut stream = engine.run("hello", abort).await.unwrap();
            let mut emitted = Vec::new();
            while let Some(item) = stream.next().await {
                emitted.push(item.unwrap());
            }
            let snap = engine.snapshot().unwrap();
            (emitted, snap)
        })
    }

    #[test]
    fn happy_path_text_only() {
        let (emitted, snap) = run_blocking(vec![
            Event::TextDelta { delta: "hi ".into() },
            Event::TextDelta { delta: "there".into() },
        ]);

        // Engine emits the 2 forwarded TextDeltas + 1 synthesized Result.
        assert_eq!(emitted.len(), 3);
        assert!(matches!(emitted[0], Event::TextDelta { ref delta, .. } if delta == "hi "));
        assert!(matches!(emitted[1], Event::TextDelta { ref delta, .. } if delta == "there"));
        assert!(matches!(emitted[2], Event::Result { .. }));

        // Store should have user + assistant.
        assert_eq!(snap.len(), 2);
        assert!(matches!(snap[0], Message::User { .. }));
        assert!(matches!(snap[1], Message::Assistant { .. }));
        if let Message::Assistant { content, .. } = &snap[1] {
            if let ContentBlock::Text { text } = &content[0] {
                assert_eq!(text, "hi there");
            } else {
                panic!("expected Text content");
            }
        } else {
            panic!("expected Assistant variant");
        }
    }

    #[test]
    fn provider_emitted_result_not_duplicated() {
        let (emitted, _) = run_blocking(vec![
            Event::TextDelta { delta: "x".into() },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]);

        // Should be exactly 2 events forwarded; no synthesized Result.
        assert_eq!(emitted.len(), 2);
        assert!(matches!(emitted[1], Event::Result { ref data, .. }
            if data.stop_reason.as_deref() == Some("end_turn")));
    }

    #[test]
    fn empty_provider_stream_still_emits_result() {
        let (emitted, snap) = run_blocking(vec![]);
        assert_eq!(emitted.len(), 1);
        assert!(matches!(emitted[0], Event::Result { .. }));
        // Store has user + (empty-content) assistant.
        assert_eq!(snap.len(), 2);
        if let Message::Assistant { content, .. } = &snap[1] {
            if let ContentBlock::Text { text } = &content[0] {
                assert_eq!(text, "");
            }
        }
    }

    #[test]
    fn assistant_parent_links_to_user() {
        let (_, snap) = run_blocking(vec![Event::TextDelta { delta: "a".into() }]);
        let user_uuid = snap[0].uuid();
        let assistant_parent = snap[1].parent_uuid();
        assert_eq!(assistant_parent, Some(user_uuid));
    }

    #[test]
    fn engine_propagates_provider_errors() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            #[derive(Debug)]
            struct ErrorProvider;
            #[async_trait::async_trait]
            impl Provider for ErrorProvider {
                fn id(&self) -> &str {
                    "err"
                }
                fn capabilities(&self) -> crate::provider::ProviderCapabilities {
                    crate::provider::ProviderCapabilities::default()
                }
                async fn stream(
                    &self,
                    _req: StreamRequest,
                    _abort: AbortController,
                ) -> Result<Box<dyn EventStream>, AgentError> {
                    Err(AgentError::provider("err", "boom"))
                }
            }
            let engine = QueryEngine::new(Arc::new(ErrorProvider), "any");
            let res = engine.run("hi", AbortController::new()).await;
            assert!(matches!(res, Err(AgentError::Provider { .. })));
        });
    }

    #[test]
    fn builder_sets_system_and_tokens() {
        let provider = Arc::new(MockProvider::new(vec![]));
        let engine = QueryEngine::new(provider, "m")
            .with_system("be brief")
            .with_max_output_tokens(1024);
        assert_eq!(engine.system.as_deref(), Some("be brief"));
        assert_eq!(engine.max_output_tokens, 1024);
    }
}
