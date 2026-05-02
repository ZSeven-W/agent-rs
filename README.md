# agent-rs

[![CI](https://github.com/ZSeven-W/agent-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ZSeven-W/agent-rs/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Rust 1.80+](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)

**A pure-Rust async runtime for building LLM agents.** Multi-provider, tool-capable, and built around a clear streaming `Event` vocabulary so you can drop it into a CLI, an IDE, a desktop app, or a server.

```rust
use agent::prelude::*;
use std::sync::Arc;

let provider = Arc::new(AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY")?));
let engine = QueryEngine::new(provider, "claude-opus-4-7")
    .with_system("Be concise.");

let mut stream = engine.run("Summarize Rust's borrow checker in two lines.", AbortController::new()).await?;
while let Some(event) = futures::StreamExt::next(&mut stream).await {
    match event? {
        Event::TextDelta { delta } => print!("{delta}"),
        Event::Result { .. } => println!(),
        _ => {}
    }
}
```

## Highlights

- **Three providers, one event stream** — Anthropic Messages (hand-rolled SSE, full prompt-cache + extended-thinking betas), OpenAI-compatible (`async-openai` 0.36, covers DeepSeek / Moonshot / Groq / OpenRouter / LM Studio), and local Ollama.
- **Tool-capable end-to-end** — define a tool, register it, the runtime wires the JSON Schema into the request body, dispatches `ToolUse` events to your code, feeds results back. Multi-turn loop with phase machine; receipt-order concurrent execution.
- **Structured permission system** — 7-step decision chain (deny / ask / callback / bypass / allow / default-ask / dont_ask), composable `PermissionMatcher` rules over tool input shapes (JSON-pointer fields, glob/prefix/regex-style patterns, AnyOf / AllOf / Not), and a four-level `SafetyClass` lattice that fails safe for unclassified tools.
- **Real MCP support** — full Model Context Protocol client lifecycle: stdio child processes, streamable HTTP, OAuth 2.0 + PKCE, server-initiated elicitation, channel permissions, stale-handle reconnect repair.
- **Cost accounting in nanodollar precision** — `Event::Usage` flows into a `CostTracker` with a model-price catalog (Claude + GPT defaults, BYO entries trivially). Integer accumulator, no f64 drift.
- **Files API for large attachments** — `FilesClient` trait + `AnthropicFilesClient`. Smart helpers auto-route between inline base64 and uploaded `file_id` references based on size.
- **Reactive auto-compaction** — token estimator + LLM-driven `<analysis>`/`<summary>` summarization, microcompact, session memory, post-cleanup file restoration. Keeps long sessions inside the context window without losing critical state.
- **Optional coding tool pack** — companion crate `agent-tools-code` ships generic FileRead/Write/Edit, Grep/Glob (gitignore-aware), and a `ToolSearch` for deferred-tool discovery. Each tool declares its `SafetyClass`; a `WorkspacePolicy` enforces path containment + size caps + symlink rules.
- **Built for embedding** — library-only, no panics on bad input, no `unsafe`, no `tokio::main`, every async surface respects an `AbortController`. Default features stay slim; opt in to providers / persistence / MCP / swarm via Cargo features.

## Workspace layout

```text
agent-rs/
├── Cargo.toml                       # workspace root
└── crates/
    ├── agent/                       # the runtime library
    └── agent-tools-code/            # optional coding tool pack
```

The two crates are versioned together, but you only need to depend on the ones you use.

## Install

```toml
[dependencies]
agent = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["anthropic", "session-jsonl"] }

# Optional: ready-made coding tool pack (FileRead/Write/Edit, Grep/Glob, ToolSearch)
agent-tools-code = { git = "https://github.com/ZSeven-W/agent-rs", default-features = false, features = ["fs", "search"] }
```

Vendored installs (submodule under `vendor/agent-rs/`) work the same way with `path = "vendor/agent-rs/crates/agent"`.

### Feature flags — `agent`

| Flag | Pulls in | Notes |
|---|---|---|
| `anthropic` *(default)* | `reqwest` + `eventsource-stream` | Hand-rolled Anthropic SSE — no SDK dep. |
| `openai` | `async-openai` 0.36 | OpenAI-compatible providers. |
| `ollama` | `ollama-rs` 0.3 | Local models. |
| `mcp` | `rmcp` 1.5 + `reqwest` + `http` | MCP client lifecycle + production stdio/HTTP connector + OAuth/PKCE. |
| `session-jsonl` | `fs4` | JSONL session persistence with file lock. |
| `swarm` | `fs4` + `notify` | Sub-agents, mailbox, teams. |
| `full` | all of the above | |

### Feature flags — `agent-tools-code`

| Flag | Pulls in | Notes |
|---|---|---|
| `fs` *(default)* | (none — uses `tokio::fs`) | FileRead/Write/Edit, ListDir, Mkdir, Move, Remove. |
| `search` *(default)* | `regex` + `ignore` | Grep + Glob with gitignore-aware traversal. |
| `shell` | `shell-words` | Bash *(planned)*. |
| `web` | `reqwest` | WebFetch *(planned)*. |
| `todo` | (none) | TodoWrite *(planned)*. |
| `all` | all of the above | |

## Module surface — `agent`

### Foundation

| Module | Purpose |
|---|---|
| `provider/` | Multi-provider LLM client. Tool definitions wired into request bodies; capability flags + streaming `Event` vocabulary. |
| `query/` | `QueryLoop` multi-turn phase machine. Reactive auto-compaction wired in. |
| `tool/` | `Tool` trait, `ToolRegistry`, `ToolUseContext`, `SafetyClass` lattice. Receipt-order concurrent execution via `ToolExecutor`. |
| `permission/` | 7-step chain + structured `PermissionMatcher` (Always / Field / ExactJson / AnyOf / AllOf / Not) + `StringPattern`. External-queue async approval flow. |
| `hook/` | 27 typed `HookEvent` variants — Before/AfterToolUse, Pre/PostCompact, OnSession*. |
| `message/` | `Message` enum + DAG-aware `MessageStore`. `ContentBlock::Document` for PDFs / large text; `ImageSource::File` for Files-API references. |
| `stream/` | `Event` taxonomy (TextDelta / Thinking / ToolUse / ToolResult / Result / Usage / Error / Notice). |
| `session/` | JSONL persistence (schema v1) with atomic-rename + `fs4` file lock. |
| `swarm/` | Sub-agents / teams — file-locked mailbox, leader-worker permission sync, in-process / tmux / iTerm2 backends. |
| `compact/` | Token estimator + reactive auto-compaction. LLM-driven summarization, partial directions, microcompact, session-memory store. |
| `context/` | Sliding-window trim. |

### Service layer

| Module | Purpose |
|---|---|
| `api/` | Retry policy with decorrelated jitter, error classification, prompt-cache-break detection (with tool-schema fingerprints), effort/output config, request-fingerprint logging, secret redaction. |
| `cost/` | Model-price-aware USD accounting. `ModelPriceCatalog` (Anthropic + OpenAI defaults), `CostTracker` consumes `Event::Usage`, `CostSnapshot` in `u128` nanodollars. |
| `attachments/` | Image + document helpers. Magic-byte mime sniff, inline base64, URL-source images, `FilesClient` trait + `AnthropicFilesClient`, smart size-aware routing. |
| `tokenizer/` | Pluggable trait + `HeuristicTokenizer` / `WordSplitTokenizer` defaults. Real tiktoken plugs in via the trait. |

### Discovery + extensibility

| Module | Purpose |
|---|---|
| `mcp/` *(feature `mcp`)* | Full MCP client lifecycle + production `RmcpConnector` (stdio + HTTP/SSE), OAuth 2.0 + PKCE, channel permissions, server-initiated elicitation. |
| `memdir/` | `MEMORY.md` directory loader — YAML-subset frontmatter, 4-type taxonomy, age-bucket relevance scoring. |
| `skills/` | Frontmatter-loaded prompt templates + input-schema validation + optional model override / tool allowlist. |
| `plugins/` | Plugin trait + registry. Native plugins are Rust trait objects; third-party plugins run in WASM via the `WasmPluginHost` trait. |
| `state/` | `AppStateStore` — typed transient runtime state with broadcast subscribers + `Selector` projection. |
| `bootstrap/` | Schema-versioned migration runner. |
| `context_analysis/` | Inspect a `MessageStore`: token totals by role, top-N largest messages, tool-call breakdown. |
| `tasks/` | Planning task graph — cycle-checked dependencies, status transitions, ready-task query. |
| `memory_extract/` | Heuristic background extractor for `DECISION:` / `User prefers …` / URL-bearing patterns. |
| `remote/` | JSON-RPC-2.0 line-delimited protocol for external hosts driving the agent in a separate process. |

## Module surface — `agent-tools-code`

Optional companion crate. Every tool implements `agent::tool::Tool` with a proper `SafetyClass`.

| Tool | Class | Feature |
|---|---|---|
| `FileReadTool` | ReadOnly | `fs` |
| `FileWriteTool` | Mutating | `fs` |
| `FileEditTool` | Mutating | `fs` |
| `ListDirTool` | ReadOnly | `fs` |
| `MkdirTool` | Mutating | `fs` |
| `MoveTool` | Mutating | `fs` |
| `RemoveTool` | Destructive | `fs` |
| `GrepTool` | ReadOnly | `search` |
| `GlobTool` | ReadOnly | `search` |
| `ToolSearchTool` | ReadOnly | (always) |

A shared `WorkspacePolicy` enforces path containment, file-size caps, and symlink rules. `register_default(registry, policy)` bulk-registers every enabled tool.

`ToolSearchTool` lets a host expose 50+ candidate tools without flooding the model's tool list — register them on a separate registry and the model uses `select:Name1,Name2` or keyword search to surface the few it needs.

## Design principles

1. **Library-only.** No global state, no `tokio::main`, no `panic!` on bad input — every error path is a typed `AgentError`.
2. **Product-agnostic.** Concrete tools live outside the runtime crate. The `agent` crate defines the trait; `agent-tools-code` ships generic implementations; product-specific tools live in product crates.
3. **Streaming first.** Every provider is a streaming source. Multi-turn / tool dispatch / compaction are coordinated through one `Event` vocabulary, no polling.
4. **Cancellation everywhere.** Every async surface honors an `AbortController` — including the `tokio::task::spawn_blocking` workers used by Grep / Glob.
5. **No `unsafe`.** `#![forbid(unsafe_code)]` in both crates.
6. **Defensive against the model.** Permissions fail safe (Unknown ≡ Destructive for gating). Tool schemas validated before reaching the wire. Path operations canonicalize before any I/O. Idempotent writes detect no-ops.
7. **Cost-aware.** Tool schemas, prompt cache, and token usage feed an integer-precision USD accumulator. Long-running sessions don't drift.

## Testing

```sh
cargo test --workspace --all-features        # 743 + 61 unit + 14 integration + 5 doc tests, 4 ignored (real-API gates)
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

The 4 `#[ignore]`-gated tests hit real APIs (Anthropic / OpenAI / Ollama / Anthropic Files) when their environment variables are set. CI runs the full suite with mocks; real-API runs are manual.

## Contributing

PRs welcome. Two ground rules:

1. **No product-specific imports in the `agent` crate.** Generic concepts only.
2. **Adversarial review every change.** This codebase is reviewed by Codex (or equivalent) on every meaningful diff — round 1 catches bugs, round 2 catches the regressions introduced by round 1's fixes. Roughly half of the commit messages name the bug each round caught.

Open an issue first for anything bigger than a small fix so we can align on direction.

## License

MIT — see [LICENSE](./LICENSE).
