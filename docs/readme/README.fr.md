# `agent-rs`

**Langues :** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` est un runtime asynchrone pur Rust pour intégrer des agents LLM dans des produits. Il réunit plusieurs fournisseurs, les appels d'outils de bout en bout, les permissions structurées, MCP, les pièces jointes, le suivi des coûts et la compaction automatique dans une même boucle d'événements. Tous les crates interdisent `unsafe`.

> Cette README localisée est un point d'entrée rapide. La liste complète des modules et les détails d'API restent dans la [README anglaise](../../README.md).

## Produits ZSeven-W

`agent-rs` fait partie de la famille de produits AI-native de ZSeven-W :

- [zode](https://github.com/ZSeven-W/zode) - un CLI de codage AI-native pour les workflows terminal, construit autour d'un micro-noyau Rust, de plugins, de plusieurs fournisseurs de modèles et d'une TUI plein écran.
- [jian](https://github.com/ZSeven-W/jian) - un framework UI cross-platform natif Rust où un fichier `.op` peut devenir une application.
- [noema](https://github.com/ZSeven-W/noema) - une mémoire local-first non vectorielle pour coding agents, avec review queues, rappel lexical, MCP, offload S3 et contrôles de politiques d'entreprise.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - un outil open source de design vectoriel AI-native pour les workflows design-as-code et les Agent Teams concurrents.

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

Ce code forme déjà un agent complet : streaming provider, dispatch d'outils, hooks, permissions, compaction automatique et suivi des coûts en USD. Les Files API, serveurs MCP et le pack d'outils de code s'ajoutent par simple enregistrement.

## Pourquoi agent-rs ?

- **Natif Rust, en mode bibliothèque.** Pas de prise de contrôle de `tokio::main`, pas d'état global, pas de `panic!` sur une mauvaise entrée.
- **Plusieurs fournisseurs, un vocabulaire d'événements.** Anthropic Messages, APIs compatibles OpenAI et Ollama émettent les mêmes `Event`.
- **Outils de bout en bout.** Le runtime gère JSON Schema, exécution, retour des résultats, boucle multi-tour et exécution concurrente.
- **Permissions sûres par défaut.** Chaîne de décision en 7 étapes, matchers JSON et `SafetyClass` traitent les outils inconnus comme risqués.
- **MCP utilisable en production.** Processus stdio, HTTP streamable, OAuth 2.0 + PKCE, elicitation, permissions de canal et réparation de reconnexion.
- **Coûts et contexte.** Suivi en nanodollars entiers, estimation de tokens, compaction automatique, microcompact et session memory.
- **Pack d'outils optionnel.** `agent-tools-code` fournit FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit et ToolSearch.

## Installation

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Features `agent` courantes : `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Features `agent-tools-code` courantes : `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Exemples

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

## Licence

MIT. Voir [LICENSE](../../LICENSE).
