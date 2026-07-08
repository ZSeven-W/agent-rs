# `agent-rs`

**Языки:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` — это чистый Rust async runtime для встраивания LLM-агентов в реальные продукты. Он объединяет несколько провайдеров, end-to-end вызовы инструментов, структурированные разрешения, MCP, файловые вложения, учет стоимости и автоматическую компактизацию в одном event loop. Во всех crate запрещен `unsafe`.

> Эта локализованная README — быстрый вход. Полный список модулей и подробности API находятся в [английской README](../../README.md).

## Продукты ZSeven-W

`agent-rs` входит в AI-native семейство продуктов ZSeven-W:

- [zode](https://github.com/ZSeven-W/zode) - AI-native CLI для программирования в терминале, построенный вокруг Rust-микроядра, плагинов, нескольких модельных провайдеров и полноэкранного TUI.
- [jian](https://github.com/ZSeven-W/jian) - Rust-native кроссплатформенный UI-фреймворк, где файл `.op` может быть приложением.
- [noema](https://github.com/ZSeven-W/noema) - local-first невекторная память для coding agents с review queues, lexical recall, MCP, S3 offload и корпоративными политиками.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - open source AI-native инструмент векторного дизайна для design-as-code workflows и параллельных Agent Teams.

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

Это уже полноценный агент: provider streaming, dispatch инструментов, hooks, permissions, auto-compaction и учет стоимости в USD подключены. Files API, MCP servers и пакет coding tools добавляются регистрацией.

## Почему agent-rs?

- **Rust-native библиотека.** Не перехватывает `tokio::main`, не использует глобальное состояние и не делает `panic!` на плохом вводе.
- **Несколько провайдеров, один словарь событий.** Anthropic Messages, OpenAI-compatible APIs и Ollama возвращают одни и те же `Event`.
- **Инструменты end-to-end.** Runtime управляет JSON Schema, выполнением, возвратом результатов, multi-turn loop и параллельностью.
- **Безопасные разрешения по умолчанию.** Цепочка из 7 шагов, JSON matchers и `SafetyClass` считают неизвестные инструменты высокорисковыми.
- **Реальный MCP.** stdio child processes, streamable HTTP, OAuth 2.0 + PKCE, elicitation, channel permissions и reconnect repair.
- **Стоимость и контекст.** Учет в целых nanodollars, оценка tokens, auto-compaction, microcompact и session memory.
- **Опциональный набор инструментов.** `agent-tools-code` включает FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit и ToolSearch.

## Установка

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Основные `agent` features: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Основные `agent-tools-code` features: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Примеры

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## Тесты

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## Лицензия

MIT. См. [LICENSE](../../LICENSE).
