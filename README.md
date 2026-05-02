# agent-rs

Cross-product Rust agent runtime — embedded by [OpenPencil](https://github.com/ZSeven-W/openpencil) and [Zode](https://github.com/ZSeven-W/zode).

> **Status:** Phases 1–6 shipped. Tier 1 compaction parity (Q-α/β/γ/δ) shipped. Library-only — TUI / IDE bridges live in Zode, canvas chrome lives in OpenPencil. Cross-product cutover (Phase 7, OP+Zode) pending.

## What this is

A pure-Rust async agent runtime that hosts multi-turn LLM conversations end-to-end. The `agent` crate is product-agnostic; product-specific tools (canvas ops, terminal I/O, file edits, etc.) are registered into its `ToolRegistry` by the consumer.

Design parity target: [Anthropic's Claude Code TS source](https://docs.anthropic.com/en/docs/claude-code) — feature mapping in `notes/2026-05-02-claude-code-completeness-audit.md`.

This is the **Rust rewrite of the Zig codebase at `github.com/ZSeven-W/agent`**. The Zig repo stays as a reference; agent-rs is opt-in via `OPENPENCIL_AGENT_BACKEND=rust` (and Zode equivalent) until Phase 7 cuts over.

## Module surface

| Module | Purpose |
|---|---|
| `provider/` | Multi-provider LLM client. Hand-rolled Anthropic Messages SSE, `async-openai` 0.36, `ollama-rs` 0.3. Capability flags + streaming `Event` vocabulary. |
| `query/` | `QueryLoop` — multi-turn phase machine (Streaming → ToolDispatch → ToolCollecting → YieldingResult → Done). Reactive auto-compaction wired in. |
| `tool/` | `Tool` trait + `ToolRegistry` + `ToolUseContext`. Receipt-order concurrent execution via `ToolExecutor::buffered`. |
| `permission/` | 7-step chain (deny → ask → callback → bypass → allow → default-ask → dont_ask). External-queue async ask/approve flow. |
| `hook/` | 27 typed `HookEvent` variants — Before/AfterToolUse, Pre/PostCompact, OnSession*, etc. Block / Ok outcomes. |
| `compact/` | Token estimator + reactive auto-compaction (Tier 1 parity). LLM-driven `<analysis>`/`<summary>` summarization, partial directions (Earliest/Latest Half), microcompact, session-memory store, post-cleanup file restoration, grouping with safe-split. |
| `context/` | Sliding-window trim. |
| `message/` | `Message` enum (User/Assistant/System/Progress/Tombstone) + DAG-aware `MessageStore`. |
| `stream/` | `Event` taxonomy (TextDelta / Thinking / ToolUse / ToolResult / Result / Usage / Error / **Notice**) and `EventStream` blanket impl. |
| `session/` | JSONL persistence (schema v1) with atomic-rename + fs4 file lock. |
| `swarm/` | Sub-agents / teams — file-locked mailbox, leader-worker permission sync, in-process / tmux / iTerm2 backends. |

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
| `mcp` | `rmcp` 1.5 | MCP client surface (rmcp wired, full lifecycle pending). |
| `session-jsonl` | `fs4` | JSONL session persistence. |
| `swarm` | `fs4` + `notify` | Mailbox + teams. |
| `full` | all of the above | Used by OP today; Zode tomorrow. |

## Status by phase

- **Phase 1 (foundation):** message DAG, store, abort, error taxonomy. ✅
- **Phase 2 (providers):** Anthropic + OpenAI-compat + Ollama. ✅
- **Phase 3 (query loop):** multi-turn, tool dispatch, hooks, permissions. ✅
- **Phase 4 (context):** sliding window, token estimator, boundary marker. ✅
- **Phase 5 (session + MCP scaffolding):** JSONL persistence, MCP type wiring. ✅ (full MCP lifecycle still Tier 1 backlog).
- **Phase 6 (swarm):** mailbox, sub_agent, team, coordinator, permission sync, backends. ✅
- **Tier 1 — Compaction parity (Q-α/β/γ/δ):** LLM summarization, auto-trigger + microcompact + session memory, post-cleanup + grouping + hooks, **QueryLoop integration with reactive auto-compaction**. ✅ Full Claude Code parity minus TUI.
- **Tier 1 remaining:** API service layer (~2.5K LOC), MEMORY.md directory loader (~2.5K LOC), full MCP lifecycle (~4.5K LOC).
- **Phase 7 — cross-repo cutover:** OP + Zode swap from Zig agent to agent-rs. Pending.

## Cross-product API rule

The `agent` crate must NOT import or reference product-specific concepts. No `openpencil_*`, no `zode_*`. Domain tools (canvas ops, terminal I/O, file edits, grep, etc.) are implemented per-product in their own crates and registered into agent's `ToolRegistry`. TUI / terminal control / IDE bridges belong to Zode; canvas chrome belongs to `openpencil-shell`.

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
cargo test --workspace --all-features      # 235 unit + 13 integration, 3 ignored (real-API gates)
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo deny --all-features check            # CI gate
```

## License

MIT — see [LICENSE](./LICENSE).
