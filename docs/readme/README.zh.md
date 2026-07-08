# `agent-rs`

**语言:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` 是一个纯 Rust 的异步 LLM Agent 运行时。它面向真实产品集成：多模型提供商、端到端工具调用、结构化权限、MCP、文件附件、成本追踪和自动压缩都内置在同一个事件循环里。每个 crate 都禁止 `unsafe`。

> 这份本地化 README 是快速入口。完整模块表和更细的 API 说明以 [英文 README](../../README.md) 为准。

## ZSeven-W 生态产品

`agent-rs` 是 ZSeven-W AI-native 产品家族的一部分：

- [zode](https://github.com/ZSeven-W/zode) - 面向终端工作流的 AI-native 编程 CLI，采用 Rust 微内核、插件架构、多模型提供商和全屏 TUI。
- [jian](https://github.com/ZSeven-W/jian) - Rust 原生跨平台 UI 框架，让 `.op` 文件可以直接成为应用。
- [noema](https://github.com/ZSeven-W/noema) - 面向 coding agents 的本地优先、非向量记忆系统，支持 review queues、词法召回、MCP、S3 offload 和企业策略控制。
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - 开源 AI-native 矢量设计工具，面向 design-as-code 工作流和并发 Agent Teams。

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

这已经是一个完整 agent：模型流式输出、工具分发、hook、权限、自动压缩和美元成本追踪都已经接好。需要文件附件、MCP server 或内置代码工具包时，只需再注册对应能力。

## 为什么选 agent-rs？

- **Rust 原生、纯库模式。** 不接管 `tokio::main`，没有全局状态，坏输入不会 `panic!`。适合 CLI、IDE 插件、桌面应用和服务端。
- **一个事件模型连接多个提供商。** Anthropic Messages、OpenAI 兼容接口和 Ollama 都输出同一套 `Event`：`TextDelta`、`ToolUse`、`Usage`、`Result` 等。
- **工具调用全链路。** 注册工具后，运行时负责 JSON Schema、工具调度、结果回灌、多轮循环、并发执行和 receipt-order 回收。
- **权限默认安全。** 7 步决策链、JSON 输入匹配器和 `SafetyClass` 安全等级让未知工具按高风险处理。
- **真实 MCP 集成。** 支持 stdio 子进程、streamable HTTP、OAuth 2.0 + PKCE、elicitation、通道权限和断线修复。
- **成本与上下文管理。** 纳美元级整数成本追踪、token 估算、自动压缩、microcompact 和 session memory 帮长会话保持可控。
- **可选工具包。** `agent-tools-code` 提供 FileRead/Write/Edit、Grep/Glob、Bash、WebFetch、TodoWrite、NotebookEdit 和 ToolSearch。

## 安装

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

常用 `agent` features：`anthropic`、`openai`、`ollama`、`mcp`、`session-jsonl`、`swarm`、`tiktoken`、`full`。

常用 `agent-tools-code` features：`fs`、`search`、`shell`、`bash-async`、`web`、`web-search`、`task`、`todo`、`notebook`、`all`。

## 示例

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## 测试

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## 许可证

MIT。详见 [LICENSE](../../LICENSE)。
