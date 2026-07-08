# `agent-rs`

**언어:** [English](../../README.md) · [简体中文](./README.zh.md) · [繁體中文](./README.zh-TW.md) · [日本語](./README.ja.md) · [한국어](./README.ko.md) · [Français](./README.fr.md) · [Español](./README.es.md) · [Deutsch](./README.de.md) · [Português](./README.pt.md) · [Русский](./README.ru.md) · [हिन्दी](./README.hi.md) · [Türkçe](./README.tr.md) · [ไทย](./README.th.md) · [Tiếng Việt](./README.vi.md) · [Bahasa Indonesia](./README.id.md)

`agent-rs`는 LLM 에이전트를 제품에 넣기 위한 순수 Rust 비동기 런타임입니다. 여러 provider, 종단 간 도구 호출, 구조화된 권한, MCP, 파일 첨부, 비용 추적, 자동 압축을 하나의 이벤트 루프로 다룹니다. 모든 crate에서 `unsafe`를 금지합니다.

> 이 현지화 README는 빠른 안내입니다. 전체 모듈 표와 자세한 API 설명은 [영문 README](../../README.md)를 기준으로 합니다.

## ZSeven-W 제품군

`agent-rs`는 ZSeven-W의 AI-native 제품군에 속합니다:

- [zode](https://github.com/ZSeven-W/zode) - 터미널 워크플로를 위한 AI-native 코딩 CLI입니다. Rust 마이크로커널, 플러그인 아키텍처, 다중 모델 제공자, 전체 화면 TUI를 기반으로 합니다.
- [jian](https://github.com/ZSeven-W/jian) - `.op` 파일을 앱으로 만들 수 있는 Rust-native 크로스 플랫폼 UI 프레임워크입니다.
- [noema](https://github.com/ZSeven-W/noema) - coding agents를 위한 local-first 비벡터 메모리입니다. review queues, lexical recall, MCP, S3 offload, 엔터프라이즈 정책 제어를 포함합니다.
- [OpenPencil](https://github.com/ZSeven-W/openpencil) - design-as-code 워크플로와 동시 Agent Teams를 위한 오픈소스 AI-native 벡터 디자인 도구입니다.

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

이 코드만으로 provider streaming, tool dispatch, hook, permission, auto-compaction, USD cost tracking이 연결된 에이전트가 됩니다. Files API, MCP server, 번들 코드 도구 팩도 추가 등록만으로 붙일 수 있습니다.

## 왜 agent-rs인가?

- **Rust 네이티브 라이브러리.** `tokio::main`을 가로채지 않고, 전역 상태가 없으며, 잘못된 입력으로 `panic!`하지 않습니다.
- **여러 provider, 하나의 이벤트 모델.** Anthropic Messages, OpenAI 호환 API, Ollama가 같은 `Event` 형식을 사용합니다.
- **도구 호출 전체 경로 지원.** JSON Schema, 실행, 결과 회수, 다중 턴, 동시 실행을 런타임이 처리합니다.
- **안전한 권한 모델.** 7단계 결정 체인, JSON 입력 matcher, `SafetyClass`가 알 수 없는 도구를 높은 위험으로 취급합니다.
- **실제 MCP 통합.** stdio child process, streamable HTTP, OAuth 2.0 + PKCE, elicitation, channel permission, reconnect repair 지원.
- **비용과 컨텍스트 관리.** nanodollar 정밀도 비용 추적, token estimation, auto-compaction, microcompact, session memory.
- **선택형 도구 팩.** `agent-tools-code`는 FileRead/Write/Edit, Grep/Glob, Bash, WebFetch, TodoWrite, NotebookEdit, ToolSearch를 제공합니다.

## 설치

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

주요 `agent` features: `anthropic`, `openai`, `ollama`, `mcp`, `session-jsonl`, `swarm`, `tiktoken`, `full`.

주요 `agent-tools-code` features: `fs`, `search`, `shell`, `bash-async`, `web`, `web-search`, `task`, `todo`, `notebook`, `all`.

## 예제

```sh
ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_basic --features anthropic -p agent
ANTHROPIC_API_KEY=sk-... cargo run --example with_tools --features anthropic -p agent
cargo run --example notebook_edit --features notebook -p agent-tools-code
TAVILY_API_KEY=tv-... cargo run --example web_search_tavily --features web-search -p agent-tools-code
```

## 테스트

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny --all-features check
```

## 라이선스

MIT. 자세한 내용은 [LICENSE](../../LICENSE)를 참조하세요.
