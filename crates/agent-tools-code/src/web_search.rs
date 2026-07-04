//! `WebSearch` — model-driven web search via a pluggable backend.
//!
//! The tool itself is provider-agnostic: it holds an
//! `Arc<dyn WebSearchBackend>` and dispatches `query` /
//! `max_results` to it. The crate ships a [`TavilyBackend`] impl
//! out of the box (Tavily Search API — `POST
//! https://api.tavily.com/search`); hosts can swap in Brave, Bing,
//! Kagi, SerpAPI, or a private corpus by implementing the trait.
//!
//! Why not bake one provider in?
//!
//! - Search APIs all want their own authentication (API key in
//!   header / body / OAuth) and rate-limit envelopes. A trait keeps
//!   the tool's input/output surface stable while hosts pick the
//!   backend they're already paying for.
//! - The `agent-tools-code` crate stays free of any one search
//!   vendor's `Cargo.toml` ceremony — `web` feature already pulls
//!   `reqwest`, which is enough to ship Tavily without extra deps.
//!
//! Output shape is `[{title, url, snippet}]` plus an optional
//! aggregated `answer` field when the backend supports synthesized
//! answers (Tavily's `include_answer: true`). Models can feed each
//! result URL straight into [`crate::WebFetchTool`] for full-page
//! reading, so `WebSearch` → `WebFetch` is the canonical pair.

use std::sync::Arc;
use std::time::Duration;

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time::timeout;

const DEFAULT_MAX_RESULTS: usize = 5;
const HARD_MAX_RESULTS: usize = 20;
const DEFAULT_TIMEOUT_SECS: u64 = 20;
const MAX_TIMEOUT_SECS: u64 = 60;

/// One search result row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Aggregated response: results + optional synthesized `answer` and
/// echo of the `query` (handy for downstream cite-back). Backends
/// that don't synthesize an answer leave it `None`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SearchResponse {
    pub query: String,
    pub answer: Option<String>,
    pub results: Vec<SearchResult>,
}

/// Pluggable backend. Implement this for your search vendor and pass
/// it to [`WebSearchTool::new`].
#[async_trait]
pub trait WebSearchBackend: Send + Sync + std::fmt::Debug {
    /// Run a search. The tool will clamp `max_results` and apply a
    /// timeout before calling — the backend just needs to perform
    /// the HTTP call and shape the response.
    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<SearchResponse, WebSearchError>;
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WebSearchError {
    #[error("backend network error: {0}")]
    Network(String),
    #[error("backend rejected query: {0}")]
    Rejected(String),
    #[error("backend response parse failed: {0}")]
    Parse(String),
    #[error("missing api key for {backend}")]
    MissingApiKey { backend: &'static str },
    #[error("other: {0}")]
    Other(String),
}

impl From<WebSearchError> for AgentError {
    fn from(value: WebSearchError) -> Self {
        AgentError::other(format!("WebSearch: {value}"))
    }
}

/// `WebSearch` tool. Construct with a [`WebSearchBackend`] (e.g.
/// [`TavilyBackend::new`]).
#[derive(Debug)]
pub struct WebSearchTool {
    backend: Arc<dyn WebSearchBackend>,
}

impl WebSearchTool {
    pub fn new(backend: Arc<dyn WebSearchBackend>) -> Self {
        Self { backend }
    }
}

#[derive(Debug, Deserialize)]
struct WebSearchInput {
    query: String,
    /// Cap on returned results. Default 5, hard ceiling 20.
    #[serde(default)]
    max_results: Option<usize>,
    /// Per-call timeout. Default 20s, hard ceiling 60s.
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "WebSearch"
    }
    fn description(&self) -> &str {
        "Run a web search via the configured backend. Returns ranked results with title / url / snippet plus an optional synthesized answer. Default 5 results, max 20."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": HARD_MAX_RESULTS},
                "timeout_secs": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECS}
            },
            "required": ["query"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    async fn call(&self, ctx: &ToolUseContext, input: Value) -> Result<Value, AgentError> {
        let parsed: WebSearchInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("WebSearch invalid input: {e}")))?;
        let query = parsed.query.trim().to_string();
        if query.is_empty() {
            return Err(AgentError::other("WebSearch query must be non-empty"));
        }
        let max_results = parsed
            .max_results
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(1, HARD_MAX_RESULTS);
        let timeout_secs = parsed
            .timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);
        let abort = ctx.abort.clone();
        let backend = self.backend.clone();
        let q = query.clone();
        let response = tokio::select! {
            biased;
            _ = abort.cancelled() => {
                return Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "aborted".into()),
                ));
            }
            r = timeout(Duration::from_secs(timeout_secs), async move {
                backend.search(&q, max_results).await
            }) => {
                match r {
                    Ok(Ok(resp)) => resp,
                    Ok(Err(e)) => return Err(e.into()),
                    Err(_) => return Err(AgentError::other(format!(
                        "WebSearch '{query}' timed out after {timeout_secs}s"
                    ))),
                }
            }
        };
        Ok(serde_json::to_value(&response).unwrap_or_else(|_| {
            json!({
                "query": response.query,
                "results": response.results,
                "answer": response.answer,
            })
        }))
    }
}

// =============================================================
// Tavily backend
// =============================================================

/// Default backend for [`WebSearchTool`]. Hits Tavily's
/// `POST /search` with the API key in the body. Why Tavily? It's
/// the simplest of the model-friendly search APIs (no header dance,
/// JSON in / JSON out, free tier).
#[derive(Debug, Clone)]
pub struct TavilyBackend {
    api_key: String,
    client: reqwest::Client,
    endpoint: String,
}

const TAVILY_DEFAULT_ENDPOINT: &str = "https://api.tavily.com/search";

impl TavilyBackend {
    /// Construct with an API key. Hosts that already have a
    /// `reqwest::Client` (custom proxy, root CA, etc.) can use
    /// [`TavilyBackend::with_client`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::builder()
                .user_agent(format!("agent-tools-code/{}", env!("CARGO_PKG_VERSION")))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            endpoint: TAVILY_DEFAULT_ENDPOINT.to_string(),
        }
    }

    pub fn with_client(api_key: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            api_key: api_key.into(),
            client,
            endpoint: TAVILY_DEFAULT_ENDPOINT.to_string(),
        }
    }

    /// Override the API endpoint. Useful for tests or for proxying
    /// through an internal gateway.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Read the API key from `TAVILY_API_KEY`. Returns
    /// `MissingApiKey` if the env var is unset or empty.
    pub fn from_env() -> Result<Self, WebSearchError> {
        let key = std::env::var("TAVILY_API_KEY")
            .map_err(|_| WebSearchError::MissingApiKey { backend: "tavily" })?;
        if key.trim().is_empty() {
            return Err(WebSearchError::MissingApiKey { backend: "tavily" });
        }
        Ok(Self::new(key))
    }
}

#[async_trait]
impl WebSearchBackend for TavilyBackend {
    async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<SearchResponse, WebSearchError> {
        if self.api_key.trim().is_empty() {
            return Err(WebSearchError::MissingApiKey { backend: "tavily" });
        }
        let body = json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": true,
            "include_raw_content": false,
        });
        let resp = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| WebSearchError::Network(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| WebSearchError::Network(e.to_string()))?;
        if !status.is_success() {
            return Err(WebSearchError::Rejected(format!("HTTP {status}: {text}")));
        }
        let parsed: TavilyResponse = serde_json::from_str(&text)
            .map_err(|e| WebSearchError::Parse(format!("{e} (body: {})", clip(&text, 200))))?;
        let results = parsed
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title.unwrap_or_default(),
                url: r.url.unwrap_or_default(),
                snippet: r.content.unwrap_or_default(),
            })
            .collect();
        Ok(SearchResponse {
            query: query.to_string(),
            answer: parsed.answer,
            results,
        })
    }
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    fn ctx() -> ToolUseContext {
        ToolUseContext {
            cwd: std::env::current_dir().unwrap(),
            abort: agent::abort::AbortController::new(),
            file_cache: Arc::new(agent::file_cache::FileStateCache::new(
                NonZeroUsize::new(8).unwrap(),
                1024 * 1024,
            )),
            permissions: Arc::new(agent::permission::PermissionManager::new()),
            hooks: Arc::new(agent::hook::HookRunner::new()),
            task_depth: 0,
        }
    }

    #[derive(Debug)]
    struct StubBackend {
        response: SearchResponse,
        observed_max: std::sync::Mutex<Option<usize>>,
    }

    #[async_trait]
    impl WebSearchBackend for StubBackend {
        async fn search(
            &self,
            _query: &str,
            max_results: usize,
        ) -> Result<SearchResponse, WebSearchError> {
            *self.observed_max.lock().unwrap() = Some(max_results);
            Ok(self.response.clone())
        }
    }

    fn stub(resp: SearchResponse) -> Arc<StubBackend> {
        Arc::new(StubBackend {
            response: resp,
            observed_max: std::sync::Mutex::new(None),
        })
    }

    #[tokio::test]
    async fn returns_results_from_backend() {
        let backend = stub(SearchResponse {
            query: "rust".into(),
            answer: Some("Rust is a systems language.".into()),
            results: vec![SearchResult {
                title: "Rust homepage".into(),
                url: "https://www.rust-lang.org/".into(),
                snippet: "A language empowering everyone …".into(),
            }],
        });
        let tool = WebSearchTool::new(backend.clone());
        let out = tool.call(&ctx(), json!({"query": "rust"})).await.unwrap();
        assert_eq!(out["query"], "rust");
        assert_eq!(out["answer"], "Rust is a systems language.");
        assert_eq!(out["results"][0]["url"], "https://www.rust-lang.org/");
    }

    #[tokio::test]
    async fn rejects_empty_query() {
        let tool = WebSearchTool::new(stub(SearchResponse {
            query: "".into(),
            answer: None,
            results: vec![],
        }));
        let err = tool
            .call(&ctx(), json!({"query": "  "}))
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("non-empty"));
    }

    #[tokio::test]
    async fn clamps_max_results_to_hard_ceiling() {
        let backend = stub(SearchResponse {
            query: "x".into(),
            answer: None,
            results: vec![],
        });
        let tool = WebSearchTool::new(backend.clone());
        tool.call(&ctx(), json!({"query": "x", "max_results": 9999}))
            .await
            .unwrap();
        assert_eq!(
            *backend.observed_max.lock().unwrap(),
            Some(HARD_MAX_RESULTS)
        );
    }

    #[tokio::test]
    async fn aborts_on_ctx_abort_before_send() {
        let tool = WebSearchTool::new(stub(SearchResponse {
            query: "x".into(),
            answer: None,
            results: vec![],
        }));
        let c = ctx();
        c.abort.abort_with_reason("user cancelled");
        let err = tool
            .call(&c, json!({"query": "x"}))
            .await
            .expect_err("aborted");
        assert!(matches!(err, AgentError::Aborted(_)));
    }

    #[tokio::test]
    async fn classified_read_only() {
        let tool = WebSearchTool::new(stub(SearchResponse {
            query: "".into(),
            answer: None,
            results: vec![],
        }));
        assert_eq!(tool.safety_class(), SafetyClass::ReadOnly);
    }

    #[derive(Debug)]
    struct FailingBackend;

    #[async_trait]
    impl WebSearchBackend for FailingBackend {
        async fn search(
            &self,
            _query: &str,
            _max_results: usize,
        ) -> Result<SearchResponse, WebSearchError> {
            Err(WebSearchError::Rejected("rate limited".into()))
        }
    }

    #[tokio::test]
    async fn surfaces_backend_errors_as_agent_error() {
        let tool = WebSearchTool::new(Arc::new(FailingBackend));
        let err = tool
            .call(&ctx(), json!({"query": "x"}))
            .await
            .expect_err("rate limited");
        assert!(err.to_string().contains("rate limited"), "got {err}");
    }

    #[derive(Debug)]
    struct SlowBackend;

    #[async_trait]
    impl WebSearchBackend for SlowBackend {
        async fn search(
            &self,
            _query: &str,
            _max_results: usize,
        ) -> Result<SearchResponse, WebSearchError> {
            tokio::time::sleep(Duration::from_secs(60)).await;
            unreachable!()
        }
    }

    #[tokio::test]
    async fn caller_supplied_timeout_kicks_in() {
        let tool = WebSearchTool::new(Arc::new(SlowBackend));
        let err = tool
            .call(&ctx(), json!({"query": "x", "timeout_secs": 1}))
            .await
            .expect_err("timeout");
        assert!(err.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn tavily_from_env_missing_returns_error() {
        // Best-effort: if a CI runner happens to have TAVILY_API_KEY
        // set we just skip (don't override env in tests, that's a
        // sharp edge with parallel test threads).
        if std::env::var("TAVILY_API_KEY").is_ok() {
            return;
        }
        let err = TavilyBackend::from_env().expect_err("missing");
        assert!(matches!(err, WebSearchError::MissingApiKey { .. }));
    }

    #[tokio::test]
    async fn tavily_empty_api_key_errors_at_call_time() {
        let backend = TavilyBackend::new("");
        let err = backend.search("x", 5).await.expect_err("empty key");
        assert!(matches!(err, WebSearchError::MissingApiKey { .. }));
    }
}
