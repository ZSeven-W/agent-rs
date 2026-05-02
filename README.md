# agent-rs

Cross-product Rust agent runtime — embedded by [OpenPencil](https://github.com/ZSeven-W/openpencil) and [Zode](https://github.com/ZSeven-W/zode).

> **Status:** Phases 1–6 + Tier 1–4 of the claude-code completeness audit shipped. Library-only — TUI / IDE bridges live in Zode, canvas chrome lives in OpenPencil. Cross-product cutover (Phase 7, OP+Zode) pending.

## What this is

A pure-Rust async agent runtime that hosts multi-turn LLM conversations end-to-end. The `agent` crate is product-agnostic; product-specific tools (canvas ops, terminal I/O, file edits, etc.) are registered into its `ToolRegistry` by the consumer.

Design parity target: [Anthropic's Claude Code TS source](https://docs.anthropic.com/en/docs/claude-code) — feature mapping in `notes/2026-05-02-claude-code-completeness-audit.md`. As of `e23f767`, every Tier 1–4 item from that audit has shipped.

This is the **Rust rewrite of the Zig codebase at `github.com/ZSeven-W/agent`**. The Zig repo stays as a reference; agent-rs is opt-in via `OPENPENCIL_AGENT_BACKEND=rust` (and Zode equivalent) until Phase 7 cuts over.

## Module surface

### Foundation

| Module | Purpose |
|---|---|
| `provider/` | Multi-provider LLM client. Hand-rolled Anthropic Messages SSE, `async-openai` 0.36, `ollama-rs` 0.3. Capability flags + streaming `Event` vocabulary. |
| `query/` | `QueryLoop` — multi-turn phase machine (Streaming → ToolDispatch → ToolCollecting → YieldingResult → Done). Reactive auto-compaction wired in. |
| `tool/` | `Tool` trait + `ToolRegistry` + `ToolUseContext`. Receipt-order concurrent execution via `ToolExecutor::buffered`. |
| `permission/` | 7-step chain (deny → ask → callback → bypass → allow → default-ask → dont_ask). External-queue async ask/approve flow. |
| `hook/` | 27 typed `HookEvent` variants — Before/AfterToolUse, Pre/PostCompact, OnSession*, etc. Block / Ok outcomes. |
| `message/` | `Message` enum (User/Assistant/System/Progress/Tombstone) + DAG-aware `MessageStore`. |
| `stream/` | `Event` taxonomy (TextDelta / Thinking / ToolUse / ToolResult / Result / Usage / Error / **Notice**) and `EventStream` blanket impl. |
| `session/` | JSONL persistence (schema v1) with atomic-rename + fs4 file lock. |
| `swarm/` | Sub-agents / teams — file-locked mailbox, leader-worker permission sync, in-process / tmux / iTerm2 backends. |
| `compact/` | Token estimator + reactive auto-compaction (Tier 1 parity). LLM-driven `<analysis>`/`<summary>` summarization, partial directions (Earliest/Latest Half), microcompact, session-memory store, post-cleanup file restoration, grouping with safe-split. |
| `context/` | Sliding-window trim. |

### Tier 1 (claude-code parity)

| Module | Purpose |
|---|---|
| `api/` | Cross-cutting API service layer — retry policy with decorrelated jitter, error classification, prompt-cache-break detection, effort/output config, request-fingerprint logging, secret redaction. |
| `memdir/` | MEMORY.md directory loader — YAML-subset frontmatter, 4-type taxonomy (User/Feedback/Project/Reference), age-bucket relevance scoring, deterministic file scan. |
| `mcp/` | Full MCP client lifecycle — server registry with state tracking, OAuth 2.0 + PKCE auth, channel permissions, server-initiated elicitation, async per-server connect locks, stale-handle reconnect repair. (feature `mcp`) |

### Tier 2

| Module | Purpose |
|---|---|
| `state/` | `AppStateStore` — typed transient runtime state (session id, mode, queued messages, running tool, last error) with broadcast subscribers + projected `Selector`. |
| `bootstrap/` | Schema-versioned migration runner. Refuses to downgrade. Idempotent. |
| `context_analysis/` | Inspect a `MessageStore`: token totals by role, top-N largest messages, tool-call breakdown. UI cost-tracking. |
| `skills/` | Skill registry + invocation — frontmatter-loaded prompt templates with input-schema validation, optional model override, optional tool allowlist. |
| `plugins/` | Plugin trait + registry. Native plugins are full Rust trait objects; **third-party plugins run in WASM** via the `WasmPluginHost` trait (host plugs in wasmtime/wasmer in their own crate). |

### Tier 3

| Module | Purpose |
|---|---|
| `remote/` | JSON-RPC-2.0 line-delimited protocol for external hosts driving the agent in a separate process. Codec + dispatcher + stable method namespace. |
| `tasks/` | High-level planning task graph — distinct from `swarm::task` (which is the swarm execution side). Cycle-checked dependencies, status auto-transitions on completion, ready-task query. |
| `memory_extract/` | Heuristic background extractor — pulls `DECISION:`/`User prefers …`/URL-bearing/etc. patterns from text into `MemoryCandidate { kind, body, confidence }` for promotion to `memdir`. |

### Tier 4

| Module | Purpose |
|---|---|
| `tokenizer/` | Pluggable `Tokenizer` trait. `HeuristicTokenizer` (4-ASCII/1-CJK), `WordSplitTokenizer` (closer-to-BPE for English). Real tiktoken plugs in via the trait without dep-tree bloat. |
| `attachments/` | Image attachment helpers — magic-byte mime sniff, inline-base64 image blocks, URL-source images, `[file: <path>]`-prefixed text attachments. |

## Feature flags

```toml
[dependencies]
agent = { path = "vendor/agent/crates/agent", default-features = false, features = ["anthropic", "session-jsonl"] }
```

| Flag | Pulls in | Notes |
|---|---|---|
| `anthropic` (default) | `reqwest` + `eventsource-stream` + `bytes` | Hand-rolled SSE; no SDK dep. |
| `openai` | `async-openai` 0.36 | OpenAI-compatible providers. |
| `ollama` | `ollama-rs` 0.3 | Local models. |
| `mcp` | `rmcp` 1.5 + `reqwest` + `bytes` | Full MCP client lifecycle + OAuth. |
| `session-jsonl` | `fs4` | JSONL session persistence. |
| `swarm` | `fs4` + `notify` | Mailbox + teams. |
| `full` | all of the above | Used by OP today; Zode tomorrow. |

## Status by phase + tier

### Phases (foundation work)

- **Phase 1 (foundation):** message DAG, store, abort, error taxonomy. ✅
- **Phase 2 (providers):** Anthropic + OpenAI-compat + Ollama. ✅
- **Phase 3 (query loop):** multi-turn, tool dispatch, hooks, permissions. ✅
- **Phase 4 (context):** sliding window, token estimator, boundary marker. ✅
- **Phase 5 (session):** JSONL persistence + MCP type wiring. ✅
- **Phase 6 (swarm):** mailbox, sub_agent, team, coordinator, permission sync, backends. ✅

### Tiers (claude-code parity audit — `notes/2026-05-02-claude-code-completeness-audit.md`)

- **Tier 1 — Compaction (Q-α/β/γ/δ):** LLM summarization, auto-trigger, microcompact, session memory, post-cleanup, grouping, QueryLoop integration. ✅
- **Tier 1 — API service layer:** retry, prompt-cache-break, effort, output, errors, logging. ✅ (`f53d5a7`)
- **Tier 1 — MEMORY.md loader:** frontmatter, 4-type taxonomy, paths, scan, age, relevance. ✅ (`9ad5a11`)
- **Tier 1 — MCP client:** config, registry, permissions, elicitation, OAuth, lifecycle. ✅ (`01163c8`)
- **Tier 2 — State store + Bootstrap + Context analysis.** ✅ (`6cb6a35`)
- **Tier 2 — Skills + Plugins (WASM-3rd-party).** ✅ (`07970b1`)
- **Tier 3 — Remote protocol + Tasks + Memory extraction.** ✅ (`17e6acb`)
- **Tier 4 — Tokenizer trait + Attachments.** ✅ (`e23f767`)

### Phase 7 — cross-repo cutover (pending)

OP + Zode swap from Zig agent to agent-rs. Tier 1–4 surface is now feature-complete relative to the audit; Phase 7 work is wiring `agent` crate into the consumer build configs.

## Cross-product API rule

The `agent` crate must NOT import or reference product-specific concepts. No `openpencil_*`, no `zode_*`. Domain tools (canvas ops, terminal I/O, file edits, grep, etc.) are implemented per-product in their own crates and registered into agent's `ToolRegistry`. TUI / terminal control / IDE bridges belong to Zode; canvas chrome belongs to `openpencil-shell`.

## Plugin architecture

agent-rs hosts both flavors of plugin uniformly through the same `Plugin` trait:

- **Native (built-in)** plugins are full Rust trait objects — direct dispatch, no sandboxing overhead. Implement `Plugin` in your own crate.
- **Third-party** plugins run in a WASM sandbox. agent-rs deliberately does NOT pull `wasmtime` / `wasmer` into the dep tree; it ships the `WasmPluginHost` trait + a no-op default. Consumers wire a wasmtime-backed host into the registry.

Per the project decision (memory entry `project_plugins_wasm_third_party.md`, 2026-05-02): WASM only for untrusted third-party code; built-ins stay native.

## Layout

```text
agent-rs/
├── Cargo.toml             # workspace root
├── deny.toml              # cargo-deny advisory + license policy
├── crates/
│   └── agent/             # the library crate
├── notes/                 # research notes, audits, plan deliverables
└── docs/                  # architecture, migration, swarm-format docs
```

The crate name is `agent`. Consumers add it as a `vendor/agent` submodule pointing at this repo, then depend on it via `path = "vendor/agent/crates/agent"`.

## Building + testing

```sh
cargo build --all-features
cargo test --workspace --all-features      # 610 unit + 13 integration + 5 doc tests, 3 ignored (real-API gates)
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo deny --all-features check            # CI gate
```

## License

MIT — see [LICENSE](./LICENSE).
