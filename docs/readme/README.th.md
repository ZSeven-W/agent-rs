# `agent-rs`

**ภาษา:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` คือ runtime แบบ async ที่เขียนด้วย Rust ล้วนสำหรับนำ LLM agents ไปใช้ในผลิตภัณฑ์จริง รองรับหลาย provider, การเรียก tools แบบ end-to-end, permissions แบบมีโครงสร้าง, MCP, file attachments, cost tracking และ automatic compaction ใน event loop เดียว ทุก crate ห้ามใช้ `unsafe`

> README ภาษานี้เป็นคู่มือเริ่มต้นแบบย่อ รายละเอียด module และ API ฉบับเต็มให้ยึด [English README](../../README.md)

## ผลิตภัณฑ์ของ ZSeven-W

`agent-rs` เป็นส่วนหนึ่งของตระกูลผลิตภัณฑ์ AI-native ของ ZSeven-W:

- [zode](https://github.com/ZSeven-W/zode) - AI-native coding CLI สำหรับ terminal workflows สร้างบน Rust microkernel, plugins, multi-provider models และ full-screen TUI
- [jian](https://github.com/ZSeven-W/jian) - Rust-native cross-platform UI framework ที่ทำให้ไฟล์ `.op` เป็นแอปได้
- [noema](https://github.com/ZSeven-W/noema) - local-first, non-vector memory สำหรับ coding agents พร้อม review queues, lexical recall, MCP, S3 offload และ enterprise policy controls
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - open-source AI-native vector design tool สำหรับ design-as-code workflows และ concurrent Agent Teams

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

โค้ดนี้เป็น agent ที่สมบูรณ์แล้ว: provider streaming, tool dispatch, hooks, permissions, auto-compaction และ USD cost tracking ถูกเชื่อมไว้ครบ หากต้องการ Files API, MCP servers หรือ bundled coding tool pack ก็เพิ่ม registration ได้

## ทำไมต้อง agent-rs?

- **Rust-native และเป็น library ล้วน.** ไม่ยึด `tokio::main`, ไม่มี global state และไม่ `panic!` เมื่อ input ไม่ถูกต้อง
- **หลาย provider แต่ใช้ event vocabulary เดียว.** Anthropic Messages, OpenAI-compatible APIs และ Ollama ส่ง `Event` รูปแบบเดียวกัน
- **รองรับ tools end-to-end.** Runtime จัดการ JSON Schema, execution, result feedback, multi-turn loop และ concurrency
- **Permissions ที่ fail-safe.** Decision chain 7 ขั้น, JSON matchers และ `SafetyClass` ทำให้ unknown tools ถูกจัดเป็น high-risk
- **MCP ใช้งานจริง.** stdio child processes, streamable HTTP, OAuth 2.0 + PKCE, elicitation, channel permissions และ reconnect repair
- **Cost และ context management.** Nanodollar integer accounting, token estimation, auto-compaction, microcompact และ session memory
- **Tool pack แบบเลือกใช้.** `agent-tools-code` มี FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit และ ToolSearch

## ติดตั้ง

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

`agent` features ที่ใช้บ่อย: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`

`agent-tools-code` features ที่ใช้บ่อย: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`

## ตัวอย่าง

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## ทดสอบ

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## License

MIT ดู [LICENSE](../../LICENSE)
