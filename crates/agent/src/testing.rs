//! Test doubles: scripted [`MockProvider`] + recording [`FakeTool`].
//!
//! Phase 2 / Task 2.4. Used by Phase 2 batch E (QueryEngine tests) and any
//! consumer that wants to integration-test against agent-rs without
//! network access.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::provider::{Provider, ProviderCapabilities, StreamRequest};
use crate::stream::{Event, EventStream};
use crate::tool::{Tool, ToolUseContext};

/// A provider whose `stream()` returns a pre-canned sequence of [`Event`]s.
///
/// Two construction modes:
/// - [`MockProvider::new`] takes a single Vec; the first `stream()` call
///   drains it, subsequent calls return empty (matches the Phase 2
///   single-turn QueryEngine test pattern).
/// - [`MockProvider::with_turns`] takes a `Vec<Vec<Event>>` — one Vec per
///   turn. Each `stream()` call pops the front Vec, useful for the
///   multi-turn QueryLoop in Phase 3.
#[derive(Debug)]
pub struct MockProvider {
    id: String,
    capabilities: ProviderCapabilities,
    turns: Mutex<Vec<Vec<Event>>>,
}

impl MockProvider {
    pub fn new(scripted: Vec<Event>) -> Self {
        Self::with_turns(vec![scripted])
    }

    pub fn with_turns(turns: Vec<Vec<Event>>) -> Self {
        Self {
            id: "mock".into(),
            capabilities: ProviderCapabilities {
                supports_tool_use: true,
                ..Default::default()
            },
            turns: Mutex::new(turns),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_capabilities(mut self, capabilities: ProviderCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Number of turns remaining to be drained.
    pub fn remaining_turns(&self) -> usize {
        self.turns.lock().map(|t| t.len()).unwrap_or(0)
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }

    async fn stream(
        &self,
        _req: StreamRequest,
        _abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        // Pop the front turn. Subsequent calls beyond the scripted turns
        // see an empty stream — keeps tests honest about expected call
        // count.
        let events = {
            let mut turns = self
                .turns
                .lock()
                .map_err(|_| AgentError::Other("MockProvider lock poisoned".into()))?;
            if turns.is_empty() {
                Vec::new()
            } else {
                turns.remove(0)
            }
        };
        let iter = events.into_iter().map(Ok);
        Ok(Box::new(stream::iter(iter)))
    }
}

/// A tool that records every invocation and returns a fixed result.
#[derive(Debug)]
pub struct FakeTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    /// Captured invocations in call order — `(input_json,)` per call.
    pub calls: Arc<Mutex<Vec<serde_json::Value>>>,
    pub result: Result<serde_json::Value, AgentError>,
}

impl FakeTool {
    pub fn new(name: impl Into<String>, result: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: "fake tool for testing".into(),
            input_schema: serde_json::json!({"type": "object"}),
            calls: Arc::new(Mutex::new(Vec::new())),
            result: Ok(result),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_input_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }

    pub fn with_error(mut self, err: AgentError) -> Self {
        self.result = Err(err);
        self
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().map(|c| c.len()).unwrap_or(0)
    }
}

#[async_trait]
impl Tool for FakeTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> serde_json::Value {
        self.input_schema.clone()
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        if let Ok(mut g) = self.calls.lock() {
            g.push(input);
        }
        // Cheap clone; both Ok and Err are clonable enough via to_string for
        // AgentError but we re-construct to keep semantics stable.
        match &self.result {
            Ok(v) => Ok(v.clone()),
            Err(e) => Err(AgentError::Other(format!("{e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolUseContext;
    use futures::StreamExt;

    #[tokio::test]
    async fn mock_provider_returns_scripted_events() {
        let p = MockProvider::new(vec![
            Event::TextDelta { delta: "a".into() },
            Event::TextDelta { delta: "b".into() },
        ]);
        let req = StreamRequest::new("any", vec![]);
        let mut stream = p.stream(req, AbortController::new()).await.unwrap();
        let mut deltas = Vec::new();
        while let Some(item) = stream.next().await {
            if let Ok(Event::TextDelta { delta }) = item {
                deltas.push(delta);
            }
        }
        assert_eq!(deltas, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn mock_provider_drains_after_first_call() {
        let p = MockProvider::new(vec![Event::TextDelta {
            delta: "once".into(),
        }]);
        let req = StreamRequest::new("any", vec![]);
        let _first = p.stream(req.clone(), AbortController::new()).await.unwrap();
        let mut second = p.stream(req, AbortController::new()).await.unwrap();
        assert!(second.next().await.is_none());
    }

    #[tokio::test]
    async fn fake_tool_records_calls() {
        let t = FakeTool::new("echo", serde_json::json!({"ok": true}));
        let ctx = ToolUseContext::new("/tmp");
        t.call(&ctx, serde_json::json!({"x": 1})).await.unwrap();
        t.call(&ctx, serde_json::json!({"x": 2})).await.unwrap();
        assert_eq!(t.call_count(), 2);
        let calls = t.calls.lock().unwrap();
        assert_eq!(calls[0], serde_json::json!({"x": 1}));
        assert_eq!(calls[1], serde_json::json!({"x": 2}));
    }

    #[tokio::test]
    async fn fake_tool_returns_error_when_configured() {
        let t = FakeTool::new("err", serde_json::json!({})).with_error(AgentError::other("boom"));
        let ctx = ToolUseContext::new("/tmp");
        let res = t.call(&ctx, serde_json::json!({})).await;
        assert!(res.is_err());
        assert_eq!(t.call_count(), 1);
    }
}
