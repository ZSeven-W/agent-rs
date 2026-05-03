//! Minimal end-to-end example: Anthropic provider + QueryLoop.
//!
//! Streams a single-turn answer to stdout. Requires
//! `ANTHROPIC_API_KEY` in the environment.
//!
//! Run:
//!     ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic
//!
//! Counts as the "complete agent" claim from the README — a
//! provider, a model id, an abort controller, and a stream loop is
//! all you need to talk to Claude. No tool registry, no permission
//! manager, no compact state — defaults handle every other knob.

use agent::prelude::*;
use futures::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "set ANTHROPIC_API_KEY in the environment")?;

    let provider = Arc::new(AnthropicProvider::new(api_key));
    let qloop = QueryLoop::builder(provider, "claude-opus-4-7")
        .system("Be concise — two sentences max.")
        .build();

    let mut stream = qloop
        .run(
            "In two sentences: what is the borrow checker?",
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
            Event::Result { data } => {
                eprintln!("\n--- done (stop_reason={:?}) ---", data.stop_reason);
            }
            _ => {}
        }
    }
    Ok(())
}
