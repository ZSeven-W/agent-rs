# `agent-rs`

**Idiomas:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` é um runtime assíncrono em Rust puro para entregar agentes LLM em produtos reais. Ele reúne múltiplos provedores, chamadas de ferramentas de ponta a ponta, permissões estruturadas, MCP, anexos, rastreamento de custos e compactação automática em um único loop de eventos. Todos os crates proíbem `unsafe`.

> Este README localizado é uma entrada rápida. A superfície completa de módulos e os detalhes de API ficam no [README em inglês](../../README.md).

## Produtos da ZSeven-W

`agent-rs` faz parte da família de produtos AI-native da ZSeven-W:

- [zode](https://github.com/ZSeven-W/zode) - uma CLI de programação AI-native para fluxos de terminal, construída com um microkernel Rust, plugins, múltiplos provedores de modelos e TUI em tela cheia.
- [jian](https://github.com/ZSeven-W/jian) - um framework de UI cross-platform nativo em Rust no qual um arquivo `.op` pode ser um aplicativo.
- [noema](https://github.com/ZSeven-W/noema) - memória local-first não vetorial para coding agents, com review queues, recuperação lexical, MCP, offload para S3 e controles de políticas empresariais.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - uma ferramenta open source de design vetorial AI-native para fluxos design-as-code e Agent Teams concorrentes.

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

Isso já é um agente completo: streaming do provedor, dispatch de ferramentas, hooks, permissões, compactação automática e rastreamento de custo em USD. Files API, servidores MCP e o pacote de ferramentas de código entram com registro adicional.

## Por que agent-rs?

- **Nativo em Rust e apenas biblioteca.** Não toma `tokio::main`, não usa estado global e não faz `panic!` com entrada ruim.
- **Vários provedores, um vocabulário de eventos.** Anthropic Messages, APIs compatíveis com OpenAI e Ollama emitem os mesmos `Event`.
- **Ferramentas de ponta a ponta.** O runtime cuida de JSON Schema, execução, reinjeção de resultados, loop multi-turno e concorrência.
- **Permissões seguras por padrão.** Cadeia de 7 passos, matchers JSON e `SafetyClass` tratam ferramentas desconhecidas como alto risco.
- **MCP de verdade.** Processos stdio, HTTP streamable, OAuth 2.0 + PKCE, elicitation, permissões por canal e reparo de reconexão.
- **Custos e contexto.** Contabilidade inteira em nanodólares, estimativa de tokens, compactação automática, microcompact e session memory.
- **Ferramentas opcionais.** `agent-tools-code` traz FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit e ToolSearch.

## Instalação

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Features comuns de `agent`: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Features comuns de `agent-tools-code`: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Exemplos

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## Testes

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## Licença

MIT. Veja [LICENSE](../../LICENSE).
