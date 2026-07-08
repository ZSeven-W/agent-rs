# `agent-rs`

**Sprachen:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` ist eine reine Rust-Async-Runtime zum Ausliefern von LLM-Agenten in echten Produkten. Mehrere Provider, Tool-Aufrufe Ende zu Ende, strukturierte Berechtigungen, MCP, Datei-Anhänge, Kostenerfassung und automatische Kompaktierung laufen über dieselbe Event-Schleife. Alle Crates verbieten `unsafe`.

> Diese lokalisierte README ist ein schneller Einstieg. Die vollständige Moduloberfläche und detaillierte API-Hinweise stehen in der [englischen README](../../README.md).

## ZSeven-W-Produkte

`agent-rs` ist Teil der AI-native Produktfamilie von ZSeven-W:

- [zode](https://github.com/ZSeven-W/zode) - eine AI-native Coding-CLI für Terminal-Workflows, gebaut um einen Rust-Mikrokernel, Plugins, mehrere Modellanbieter und eine Vollbild-TUI.
- [jian](https://github.com/ZSeven-W/jian) - ein Rust-natives Cross-Platform-UI-Framework, bei dem eine `.op`-Datei eine App sein kann.
- [noema](https://github.com/ZSeven-W/noema) - local-first, nicht-vektorbasierter Speicher für coding agents, mit review queues, lexical recall, MCP, S3 offload und Enterprise-Policy-Kontrollen.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - ein Open-Source AI-native Vektordesign-Tool für design-as-code-Workflows und parallele Agent Teams.

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

Das ist bereits ein vollständiger Agent: Provider-Streaming, Tool-Dispatch, Hooks, Berechtigungen, automatische Kompaktierung und USD-Kostentracking sind verdrahtet. Files API, MCP-Server oder das Code-Toolpaket werden zusätzlich registriert.

## Warum agent-rs?

- **Rust-nativ und nur Bibliothek.** Kein Übernehmen von `tokio::main`, kein globaler Zustand, kein `panic!` bei ungültigen Eingaben.
- **Mehrere Provider, ein Event-Vokabular.** Anthropic Messages, OpenAI-kompatible APIs und Ollama liefern dieselben `Event`-Typen.
- **Tool-Aufrufe Ende zu Ende.** Die Runtime behandelt JSON Schema, Ausführung, Ergebnisrückführung, Multi-Turn-Loops und Parallelität.
- **Berechtigungen fallen sicher aus.** 7-stufige Entscheidungskette, JSON-Matcher und `SafetyClass` behandeln unbekannte Tools als hohes Risiko.
- **Praktisches MCP.** stdio-Prozesse, streamable HTTP, OAuth 2.0 + PKCE, Elicitation, Kanalberechtigungen und Reconnect-Reparatur.
- **Kosten und Kontext.** Nanodollar-genaue Integer-Kosten, Token-Schätzung, Auto-Compaction, Microcompact und Session Memory.
- **Optionale Tools.** `agent-tools-code` enthält FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit und ToolSearch.

## Installation

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Häufige `agent`-Features: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Häufige `agent-tools-code`-Features: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Beispiele

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

## Lizenz

MIT. Siehe [LICENSE](../../LICENSE).
