# `agent-rs`

**भाषाएं:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` LLM agents को product में जोड़ने के लिए pure Rust async runtime है। यह multi-provider support, end-to-end tool calls, structured permissions, MCP, file attachments, cost tracking और automatic compaction को एक ही event loop में जोड़ता है। हर crate में `unsafe` forbidden है।

> यह localized README quick start है। पूरी module surface और API details के लिए [English README](../../README.md) canonical है।

## ZSeven-W उत्पाद

`agent-rs` ZSeven-W के AI-native product family का हिस्सा है:

- [zode](https://github.com/ZSeven-W/zode) - terminal workflows के लिए AI-native coding CLI, जो Rust microkernel, plugins, multi-provider models और full-screen TUI पर आधारित है।
- [jian](https://github.com/ZSeven-W/jian) - Rust-native cross-platform UI framework, जहां एक `.op` file सीधे app बन सकती है।
- [noema](https://github.com/ZSeven-W/noema) - coding agents के लिए local-first, non-vector memory, जिसमें review queues, lexical recall, MCP, S3 offload और enterprise policy controls शामिल हैं।
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - design-as-code workflows और concurrent Agent Teams के लिए open-source AI-native vector design tool।

## TL;DR

```rust
use agent::prelude::*;
use std::sync::Arc;

let provider = Arc::new(AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY")?));
let engine = QueryEngine::new(provider, "claude-opus-4-7").with_system("Be concise.");

let mut stream = engine.run("Summarize Rust's borrow checker in two lines.", AbortController::new()).await?;
while let Some(event) = futures::StreamExt::next(&mut stream).await {
    if let Event::TextDelta { delta } = event? { print!("{delta}") }
}
```

यह code पहले से एक complete agent है: provider streaming, tool dispatch, hooks, permissions, auto-compaction और USD cost tracking wired हैं। Files API, MCP servers या bundled coding tool pack को extra registration से जोड़ा जा सकता है।

## agent-rs क्यों?

- **Rust-native library.** यह `tokio::main` hijack नहीं करता, global state नहीं रखता और bad input पर `panic!` नहीं करता।
- **कई providers, एक event vocabulary.** Anthropic Messages, OpenAI-compatible APIs और Ollama समान `Event` shape देते हैं।
- **End-to-end tools.** Runtime JSON Schema, execution, result feedback, multi-turn loop और concurrency संभालता है।
- **Fail-safe permissions.** 7-step decision chain, JSON input matchers और `SafetyClass` unknown tools को high-risk मानते हैं।
- **Real MCP integration.** stdio child processes, streamable HTTP, OAuth 2.0 + PKCE, elicitation, channel permissions और reconnect repair।
- **Cost और context management.** Nanodollar integer accounting, token estimation, auto-compaction, microcompact और session memory।
- **Optional tool pack.** `agent-tools-code` में FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit और ToolSearch हैं।

## Install

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Common `agent` features: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Common `agent-tools-code` features: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Examples

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## Tests

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## License

MIT. See [LICENSE](../../LICENSE).
