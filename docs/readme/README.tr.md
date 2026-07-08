# `agent-rs`

**Diller:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs`, LLM agent'larını ürünlere yerleştirmek için saf Rust ile yazılmış async bir runtime'dır. Çoklu provider desteği, uçtan uca tool çağrıları, yapılandırılmış izinler, MCP, dosya ekleri, maliyet takibi ve otomatik compaction aynı event loop içinde birleşir. Tüm crate'lerde `unsafe` yasaktır.

> Bu yerelleştirilmiş README hızlı başlangıç içindir. Tam modül yüzeyi ve ayrıntılı API açıklamaları için [İngilizce README](../../README.md) esas alınır.

## ZSeven-W Ürünleri

`agent-rs`, ZSeven-W'nin AI-native ürün ailesinin bir parçasıdır:

- [zode](https://github.com/ZSeven-W/zode) - terminal iş akışları için AI-native kodlama CLI'si; Rust microkernel, plugin mimarisi, çoklu model sağlayıcıları ve tam ekran TUI üzerine kuruludur.
- [jian](https://github.com/ZSeven-W/jian) - bir `.op` dosyasını uygulamaya dönüştürebilen Rust-native cross-platform UI framework.
- [noema](https://github.com/ZSeven-W/noema) - coding agents için local-first, vektörsüz hafıza; review queues, lexical recall, MCP, S3 offload ve enterprise policy kontrolleri içerir.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - design-as-code iş akışları ve eşzamanlı Agent Teams için open-source AI-native vector design tool.

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

Bu kod başlı başına tam bir agent'tır: provider streaming, tool dispatch, hook'lar, izinler, auto-compaction ve USD maliyet takibi hazırdır. Files API, MCP server'lar veya dahili coding tool pack ek kayıtla bağlanır.

## Neden agent-rs?

- **Rust-native ve sadece kütüphane.** `tokio::main` devralınmaz, global state yoktur, kötü input `panic!` üretmez.
- **Birden çok provider, tek event sözlüğü.** Anthropic Messages, OpenAI-compatible API'ler ve Ollama aynı `Event` tiplerini üretir.
- **Uçtan uca tool desteği.** Runtime JSON Schema, çalıştırma, sonuç besleme, multi-turn loop ve concurrency işlerini üstlenir.
- **Fail-safe izinler.** 7 adımlı karar zinciri, JSON matcher'lar ve `SafetyClass` bilinmeyen tool'ları yüksek riskli kabul eder.
- **Gerçek MCP entegrasyonu.** stdio child process, streamable HTTP, OAuth 2.0 + PKCE, elicitation, channel permission ve reconnect repair.
- **Maliyet ve context yönetimi.** Nanodollar integer accounting, token estimation, auto-compaction, microcompact ve session memory.
- **Opsiyonel tool pack.** `agent-tools-code` FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit ve ToolSearch sağlar.

## Kurulum

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Yaygın `agent` features: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

Yaygın `agent-tools-code` features: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## Örnekler

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## Testler

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## Lisans

MIT. Bkz. [LICENSE](../../LICENSE).
