//! End-to-end example wiring the bundled coding tool pack
//! (`agent-tools-code`) into a QueryLoop. The model can read /
//! grep / glob the workspace, then summarize.
//!
//! Run:
//!     ANTHROPIC_API_KEY=sk-... cargo run \
//!         --example with_tools \
//!         --features anthropic \
//!         -p agent
//!
//! NOTE: this example depends on `agent-tools-code` being a dev-
//! dependency of the `agent` crate. We add that wiring inline via
//! `[[example]]` in `crates/agent/Cargo.toml` rather than dragging
//! it into the prod dep tree.

use agent::prelude::*;
use agent_tools_code::{register_default, WorkspacePolicy};
use futures::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "set ANTHROPIC_API_KEY in the environment")?;

    // Workspace policy = current directory, default 8 MiB file cap,
    // symlinks resolved. Hosts that want sibling-dir reads call
    // `.with_allowed_root(...)`.
    let cwd = std::env::current_dir()?;
    let policy = WorkspacePolicy::new(&cwd)?.into_arc();

    // Register every default tool (FileRead/Write/Edit, ListDir,
    // Mkdir, Move, Remove, Grep, Glob). Since this crate doesn't
    // pull `shell` / `web` / `todo` / `notebook` features, those
    // tools aren't included unless you opt in via
    // `agent-tools-code` features.
    let mut tools = ToolRegistry::new();
    register_default(&mut tools, policy);

    let provider = Arc::new(AnthropicProvider::new(api_key));
    let qloop = QueryLoop::builder(provider, "claude-opus-4-7")
        .tools(Arc::new(tools))
        .system(
            "You are a concise code reader. Use Grep + FileRead to \
             answer questions about the workspace. Stop after the \
             second tool call.",
        )
        .build();

    let mut stream = qloop
        .run(
            "List the .rs files at the workspace root and tell me \
             how many lines `lib.rs` has, in one sentence.",
            AbortController::new(),
        )
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            Event::TextDelta { delta } => {
                use std::io::Write;
                print!("{delta}");
                std::io::stdout().flush().ok();
            }
            Event::ToolUse { name, .. } => {
                eprintln!("\n[tool] {name}");
            }
            Event::Result { data } => {
                eprintln!("\n--- done (stop_reason={:?}) ---", data.stop_reason);
            }
            _ => {}
        }
    }
    Ok(())
}
