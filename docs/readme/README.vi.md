# `agent-rs`

**Ngôn ngữ:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` là runtime async thuần Rust để đưa LLM agents vào sản phẩm thật. Nó gom nhiều provider, tool calls end-to-end, structured permissions, MCP, file attachments, cost tracking và automatic compaction vào cùng một event loop. Mọi crate đều cấm `unsafe`.

> README bản địa hóa này là hướng dẫn nhanh. Danh sách module đầy đủ và chi tiết API nằm trong [README tiếng Anh](../../README.md).

## Sản phẩm của ZSeven-W

`agent-rs` là một phần trong họ sản phẩm AI-native của ZSeven-W:

- [zode](https://github.com/ZSeven-W/zode) - AI-native coding CLI cho terminal workflows, được xây quanh Rust microkernel, plugins, nhiều model providers và full-screen TUI.
- [jian](https://github.com/ZSeven-W/jian) - Rust-native cross-platform UI framework, nơi một file `.op` có thể trở thành app.
- [noema](https://github.com/ZSeven-W/noema) - local-first, non-vector memory cho coding agents, gồm review queues, lexical recall, MCP, S3 offload và enterprise policy controls.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - open-source AI-native vector design tool cho design-as-code workflows và concurrent Agent Teams.

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

Đây đã là một agent hoàn chỉnh: provider streaming, tool dispatch, hooks, permissions, auto-compaction và USD cost tracking đã được nối sẵn. Files API, MCP servers hoặc bundled coding tool pack có thể được thêm bằng registration.

## Vì sao chọn agent-rs?

- **Rust-native, dạng library.** Không chiếm `tokio::main`, không có global state và không `panic!` khi input xấu.
- **Nhiều provider, một event vocabulary.** Anthropic Messages, OpenAI-compatible APIs và Ollama phát cùng một dạng `Event`.
- **Tools end-to-end.** Runtime xử lý JSON Schema, execution, result feedback, multi-turn loop và concurrency.
- **Permissions fail-safe.** Decision chain 7 bước, JSON matchers và `SafetyClass` xem unknown tools là high-risk.
- **MCP thực dụng.** stdio child processes, streamable HTTP, OAuth 2.0 + PKCE, elicitation, channel permissions và reconnect repair.
- **Cost và context management.** Nanodollar integer accounting, token estimation, auto-compaction, microcompact và session memory.
- **Tool pack tùy chọn.** `agent-tools-code` có FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit và ToolSearch.

## Cài đặt

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Các `agent` features thường dùng: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Các `agent-tools-code` features thường dùng: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Ví dụ

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## Kiểm thử

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## Giấy phép

MIT. Xem [LICENSE](../../LICENSE).
