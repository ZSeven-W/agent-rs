# `agent-rs`

**語言:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` 是純 Rust 的非同步 LLM Agent 執行環境。它針對實際產品整合而設計：多模型提供商、端到端工具呼叫、結構化權限、MCP、檔案附件、成本追蹤與自動壓縮，都整合在同一個事件迴圈中。每個 crate 都禁止 `unsafe`。

> 這份本地化 README 是快速入口。完整模組表與更詳細的 API 說明以 [英文 README](../../README.md) 為準。

## ZSeven-W 生態產品

`agent-rs` 是 ZSeven-W AI-native 產品家族的一部分：

- [zode](https://github.com/ZSeven-W/zode) - 面向終端工作流的 AI-native 程式設計 CLI，採用 Rust 微核心、外掛架構、多模型提供商和全螢幕 TUI。
- [jian](https://github.com/ZSeven-W/jian) - Rust 原生跨平台 UI 框架，讓 `.op` 檔案可以直接成為應用程式。
- [noema](https://github.com/ZSeven-W/noema) - 面向 coding agents 的本地優先、非向量記憶系統，支援 review queues、詞法召回、MCP、S3 offload 和企業策略控制。
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - 開源 AI-native 向量設計工具，面向 design-as-code 工作流和並發 Agent Teams。

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

這已經是一個完整 agent：模型串流、工具派發、hook、權限、成本追蹤與自動壓縮都已串好。若需要檔案附件、MCP server 或內建程式碼工具包，只要再註冊對應能力。

## 為什麼選 agent-rs？

- **Rust 原生、純函式庫模式。** 不接管 `tokio::main`，沒有全域狀態，錯誤輸入不會 `panic!`。
- **多提供商共用同一事件模型。** Anthropic Messages、OpenAI 相容介面與 Ollama 都輸出相同的 `Event` 類型。
- **工具呼叫全鏈路。** 註冊工具後，執行環境會處理 JSON Schema、工具派發、結果回灌、多輪迴圈與並行執行。
- **權限預設安全。** 7 步決策鏈、JSON 輸入匹配器與 `SafetyClass` 讓未知工具以高風險處理。
- **真正可用的 MCP。** 支援 stdio 子程序、streamable HTTP、OAuth 2.0 + PKCE、elicitation、通道權限與重連修復。
- **成本與上下文管理。** 納美元整數成本追蹤、token 估算、自動壓縮、microcompact 與 session memory。
- **可選工具包。** `agent-tools-code` 提供 FileRead/Write/Edit、Grep/Glob、Bash、WebFetch、TodoWrite、NotebookEdit 與 ToolSearch。

## 安裝

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

常用 `agent` features：`anthropic`、`openai`、`ollama`、`mcp`、`session-jsonl`、`swarm`、`tiktoken`、`full`。

常用 `agent-tools-code` features：`fs`、`search`、`shell`、`bash-async`、`web`、`web-search`、`task`、`todo`、`notebook`、`all`。

## 範例

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## 測試

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## 授權

MIT。詳見 [LICENSE](../../LICENSE)。
