# `agent-rs`

**言語:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs` は、LLM エージェントを製品に組み込むための純 Rust 非同期ランタイムです。複数プロバイダー、エンドツーエンドのツール呼び出し、構造化権限、MCP、ファイル添付、コスト追跡、自動コンパクションを同じイベントループで扱います。すべての crate で `unsafe` を禁止しています。

> このローカライズ版 README はクイックスタートです。完全なモジュール一覧と詳細な API 説明は [英語 README](../../README.md) を参照してください。

## ZSeven-W のプロダクト

`agent-rs` は ZSeven-W の AI-native プロダクト群の一部です：

- [zode](https://github.com/ZSeven-W/zode) - ターミナルワークフロー向けの AI-native コーディング CLI。Rust マイクロカーネル、プラグイン構成、複数モデルプロバイダー、フルスクリーン TUI を備えています。
- [jian](https://github.com/ZSeven-W/jian) - `.op` ファイルをアプリとして扱える Rust-native のクロスプラットフォーム UI フレームワーク。
- [noema](https://github.com/ZSeven-W/noema) - coding agents 向けの local-first な非ベクトルメモリ。review queues、字句リコール、MCP、S3 offload、エンタープライズポリシー制御を備えています。
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - design-as-code ワークフローと並行 Agent Teams のためのオープンソース AI-native ベクターデザインツール。

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

これだけで、ストリーミング、ツールディスパッチ、hook、権限、自動コンパクション、USD コスト追跡を備えたエージェントになります。Files API、MCP server、同梱のコードツールパックも追加登録だけで利用できます。

## なぜ agent-rs か

- **Rust ネイティブでライブラリ専用。** `tokio::main` を奪わず、グローバル状態を持たず、不正入力で `panic!` しません。
- **複数プロバイダーを同じイベント語彙で扱う。** Anthropic Messages、OpenAI 互換 API、Ollama が同じ `Event` を返します。
- **ツール呼び出しを端から端までサポート。** JSON Schema、ツール実行、結果の再投入、複数ターン、並行実行をランタイムが処理します。
- **安全側に倒す権限モデル。** 7 ステップの判定チェーン、JSON 入力マッチャー、`SafetyClass` により未知のツールを高リスクとして扱います。
- **実用的な MCP。** stdio 子プロセス、streamable HTTP、OAuth 2.0 + PKCE、elicitation、チャンネル権限、再接続修復に対応します。
- **コストとコンテキスト管理。** nanodollar 精度の整数コスト追跡、token 推定、自動コンパクション、microcompact、session memory。
- **任意のツールパック。** `agent-tools-code` は FileRead/Write/Edit、Grep/Glob、Bash、WebFetch、TodoWrite、NotebookEdit、ToolSearch を提供します。

## インストール

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

主な `agent` features: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`。

主な `agent-tools-code` features: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`。

## 例

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## テスト

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## ライセンス

MIT。詳細は [LICENSE](../../LICENSE) を参照してください。
