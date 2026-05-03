//! Standalone example: hit Tavily Search via `WebSearchTool`.
//!
//! Run:
//!     TAVILY_API_KEY=tv-... cargo run \
//!         --example web_search_tavily \
//!         --features web-search \
//!         -p agent-tools-code

use agent::abort::AbortController;
use agent::file_cache::FileStateCache;
use agent::hook::HookRunner;
use agent::permission::PermissionManager;
use agent::tool::{Tool, ToolUseContext};
use agent_tools_code::{TavilyBackend, WebSearchTool};
use serde_json::json;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `from_env()` reads `TAVILY_API_KEY`. To use Brave / Bing /
    // your own backend, implement `WebSearchBackend` and swap it
    // in here.
    let backend = TavilyBackend::from_env().map_err(|e| format!("set TAVILY_API_KEY: {e}"))?;
    let tool = WebSearchTool::new(Arc::new(backend));

    let ctx = ToolUseContext {
        cwd: std::env::current_dir()?,
        abort: AbortController::new(),
        file_cache: Arc::new(FileStateCache::new(
            NonZeroUsize::new(8).unwrap(),
            1024 * 1024,
        )),
        permissions: Arc::new(PermissionManager::new()),
        hooks: Arc::new(HookRunner::new()),
    };

    let result = tool
        .call(
            &ctx,
            json!({
                "query": "rust async ecosystem 2026",
                "max_results": 3,
            }),
        )
        .await?;

    println!("{result:#}");
    Ok(())
}
