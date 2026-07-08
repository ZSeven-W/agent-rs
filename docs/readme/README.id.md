# `agent-rs`

**Bahasa:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` adalah runtime async murni Rust untuk membawa LLM agents ke produk nyata. Ia menyatukan multi-provider, tool calls end-to-end, structured permissions, MCP, file attachments, cost tracking, dan automatic compaction dalam satu event loop. Semua crate melarang `unsafe`.

> README lokal ini adalah panduan cepat. Permukaan module lengkap dan detail API tetap mengacu ke [README bahasa Inggris](../../README.md).

## Produk ZSeven-W

`agent-rs` adalah bagian dari keluarga produk AI-native ZSeven-W:

- [zode](https://github.com/ZSeven-W/zode) - AI-native coding CLI untuk terminal workflows, dibangun dengan Rust microkernel, plugins, multi-provider models, dan full-screen TUI.
- [jian](https://github.com/ZSeven-W/jian) - Rust-native cross-platform UI framework tempat file `.op` bisa menjadi aplikasi.
- [noema](https://github.com/ZSeven-W/noema) - local-first, non-vector memory untuk coding agents, mencakup review queues, lexical recall, MCP, S3 offload, dan enterprise policy controls.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - open-source AI-native vector design tool untuk design-as-code workflows dan concurrent Agent Teams.

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

Itu sudah menjadi agent lengkap: provider streaming, tool dispatch, hooks, permissions, auto-compaction, dan USD cost tracking sudah tersambung. Files API, MCP servers, atau bundled coding tool pack bisa ditambahkan lewat registration.

## Mengapa agent-rs?

- **Rust-native dan library-only.** Tidak mengambil alih `tokio::main`, tidak memakai global state, dan tidak `panic!` pada input buruk.
- **Banyak provider, satu event vocabulary.** Anthropic Messages, OpenAI-compatible APIs, dan Ollama mengeluarkan bentuk `Event` yang sama.
- **Tools end-to-end.** Runtime menangani JSON Schema, execution, result feedback, multi-turn loop, dan concurrency.
- **Permissions fail-safe.** Decision chain 7 langkah, JSON matchers, dan `SafetyClass` menganggap unknown tools sebagai high-risk.
- **MCP nyata.** stdio child processes, streamable HTTP, OAuth 2.0 + PKCE, elicitation, channel permissions, dan reconnect repair.
- **Cost dan context management.** Nanodollar integer accounting, token estimation, auto-compaction, microcompact, dan session memory.
- **Tool pack opsional.** `agent-tools-code` menyediakan FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit, dan ToolSearch.

## Instalasi

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

`agent` features yang umum: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

`agent-tools-code` features yang umum: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Contoh

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## Pengujian

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## Lisensi

MIT. Lihat [LICENSE](../../LICENSE).
