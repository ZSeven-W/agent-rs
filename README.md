# agent-rs

Cross-product Rust agent runtime — embedded by [OpenPencil](https://github.com/ZSeven-W/openpencil) and [Zode](https://github.com/ZSeven-W/zode).

> **Status:** Phase 0 — pre-plan research and repo bootstrap. Plan: `2026-04-19-agent-crate.md` (8 phases, 132 tasks). Not yet usable.

## What this is

A pure-Rust async agent runtime: multi-provider LLM client (Anthropic / OpenAI-compat / Ollama), tool trait + registry, 7-step permission chain, streaming event stream, session persistence, hooks, and swarm/teams. Designed to be `path =`-deps'd into both OpenPencil (chrome + canvas) and Zode (terminal coding CLI), so the API surface stays product-agnostic.

This is the **Rust rewrite of the Zig codebase at `github.com/ZSeven-W/agent`**. The Zig repo remains the source of truth during the migration window; agent-rs is opt-in via `OPENPENCIL_AGENT_BACKEND=rust` (and equivalent in Zode) until Phase 7 cuts over.

## Layout

```text
agent-rs/
├── Cargo.toml             # workspace root
├── crates/
│   └── agent/             # the crate (single-crate workspace until subcrate split is justified)
├── notes/                 # research notes (Phase 0 deliverables)
└── docs/                  # architecture, migration, swarm-format docs
```

The crate name is `agent`. Consumers add it as `vendor/agent` submodule pointing at this repo, and depend on it via `path = "vendor/agent/crates/agent"`.

## Cross-product API rule

The `agent` crate must NOT import or reference product-specific concepts. No `openpencil_*`, no `zode_*`. Domain tools (file/shell/grep, canvas, etc.) are implemented per-product in their own crates and registered into agent's `ToolRegistry`.

## License

MIT — see [LICENSE](./LICENSE).
