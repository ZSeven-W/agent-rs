# `agent-rs`

**Idiomas:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` es un runtime asíncrono en Rust puro para integrar agentes LLM en productos reales. Incluye múltiples proveedores, llamadas a herramientas de extremo a extremo, permisos estructurados, MCP, adjuntos, seguimiento de costes y compactación automática en un mismo bucle de eventos. Todos los crates prohíben `unsafe`.

> Este README localizado es una guía rápida. La superficie completa de módulos y los detalles de API están en el [README en inglés](../../README.md).

## Productos de ZSeven-W

`agent-rs` forma parte de la familia de productos AI-native de ZSeven-W:

- [zode](https://github.com/ZSeven-W/zode) - un CLI de programación AI-native para flujos de trabajo en terminal, construido sobre un micronúcleo Rust, plugins, múltiples proveedores de modelos y una TUI de pantalla completa.
- [jian](https://github.com/ZSeven-W/jian) - un framework UI cross-platform nativo de Rust donde un archivo `.op` puede convertirse en una aplicación.
- [noema](https://github.com/ZSeven-W/noema) - memoria local-first no vectorial para coding agents, con review queues, recuperación léxica, MCP, offload a S3 y controles de políticas empresariales.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - una herramienta open source de diseño vectorial AI-native para flujos design-as-code y Agent Teams concurrentes.

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

Esto ya es un agente completo: streaming del proveedor, dispatch de herramientas, hooks, permisos, compactación automática y seguimiento de costes en USD. Files API, servidores MCP y el paquete de herramientas de código se conectan con registros adicionales.

## ¿Por qué agent-rs?

- **Nativo de Rust y solo biblioteca.** No toma control de `tokio::main`, no usa estado global y no hace `panic!` ante entradas inválidas.
- **Varios proveedores, un vocabulario de eventos.** Anthropic Messages, APIs compatibles con OpenAI y Ollama emiten los mismos `Event`.
- **Herramientas de extremo a extremo.** El runtime maneja JSON Schema, ejecución, reinyección de resultados, bucles multi-turno y concurrencia.
- **Permisos seguros por defecto.** Cadena de 7 pasos, matchers JSON y `SafetyClass` tratan las herramientas desconocidas como de alto riesgo.
- **MCP real.** Procesos stdio, HTTP streamable, OAuth 2.0 + PKCE, elicitation, permisos por canal y reparación de reconexión.
- **Costes y contexto.** Contabilidad entera en nanodólares, estimación de tokens, compactación automática, microcompact y session memory.
- **Herramientas opcionales.** `agent-tools-code` incluye FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit y ToolSearch.

## Instalación

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Features habituales de `agent`: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Features habituales de `agent-tools-code`: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Ejemplos

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## Pruebas

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## Licencia

MIT. Ver [LICENSE](../../LICENSE).
