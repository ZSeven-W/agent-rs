# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/).

Nothing has been tagged yet — every change below is part of the
upcoming first release. Sections will be split out into proper
versions when `0.1.0` ships.

## Unreleased

### Added

#### Runtime — `agent` crate

- `provider/` — multi-provider LLM client. Hand-rolled Anthropic
  Messages SSE (full prompt-cache + extended-thinking betas);
  `async-openai` 0.36 covering OpenAI / DeepSeek / Moonshot / Groq /
  OpenRouter / LM Studio; `ollama-rs` 0.3 for local models. Tool
  definitions wired into request bodies for all three providers.
  Tool-schema fingerprints feed prompt-cache-break detection.
- `query/` — `QueryEngine` for single-turn and `QueryLoop` for the
  full multi-turn phase machine (Streaming → ToolDispatch →
  ToolCollecting → YieldingResult → Done). Reactive auto-compaction
  wired in with cross-run `AutoCompactState` (circuit breaker,
  no-progress latch).
- `tool/` — `Tool` trait, `ToolRegistry`, `ToolUseContext`. Four-level
  `SafetyClass` lattice (ReadOnly / Mutating / Destructive / Unknown)
  where Unknown is treated as Destructive for `is_at_least` gating
  so unclassified tools fail safe. `ToolExecutor::buffered` runs tool
  dispatches concurrently and yields results in receipt order.
- `permission/` — 7-step decision chain (deny / ask / callback /
  bypass / allow / default-ask / dont_ask) with an external-queue
  async approval flow for human-in-the-loop UX. Structured input
  matchers: `PermissionMatcher` enum (Always / Field / ExactJson /
  AnyOf / AllOf / Not) over JSON pointers (RFC 6901), plus
  `StringPattern` (Exact / Prefix / Suffix / Contains / Glob).
  Hand-rolled glob, no regex dep.
- `hook/` — 27 typed `HookEvent` variants — Before/AfterToolUse,
  Pre/PostCompact, OnSession*, etc. — with Block / Ok outcomes.
- `message/` — `Message` enum (User / Assistant / System / Progress /
  Tombstone), DAG-aware `MessageStore`, `ContentBlock::Document` for
  PDFs / large text, `ImageSource::File` for Files API references.
- `stream/` — `Event` taxonomy: TextDelta / Thinking / ToolUse /
  ToolResult / Result / Usage / Error / Notice. `EventStream`
  blanket impl over any `Stream<Item = Result<Event, AgentError>>`.
- `session/` — JSONL persistence (schema v1) with atomic-rename plus
  `fs4` file lock; survives process crashes.
- `swarm/` — sub-agents and teams. File-locked mailbox, leader-worker
  permission sync, `SubAgent`/`Task`/`Team`/`Coordinator` types, and
  in-process / tmux / iTerm2 backends. Sync `fs4` flock wrapped in
  `tokio::task::spawn_blocking` so the runtime stays cooperative.
- `compact/` — token estimator + reactive auto-compaction. LLM-driven
  `<analysis>` / `<summary>` summarization, partial directions
  (Earliest / Latest Half), microcompact, session-memory store,
  post-cleanup file-state restoration, group-aware safe-split,
  PreCompact / PostCompact hook events.
- `context/` — sliding-window trim.
- `api/` — cross-cutting service layer. Retry policy with decorrelated
  jitter, error classification, prompt-cache-break detection (with
  tool-schema fingerprints), effort/output config, request-fingerprint
  logging, secret redaction.
- `cost/` — model-price-aware USD accounting.
  `ModelPriceCatalog` ships defaults for Anthropic + OpenAI tiers;
  `CostTracker` consumes `Event::Usage` with cumulative-replace
  semantics so providers that emit Usage multiple times per turn
  don't double-count. Per-model + session totals in `u128`
  nanodollars (no `f64` drift). `clear_event_baseline(model)`
  handles warm-cache turn boundaries.
- `attachments/` — image and document helpers. Magic-byte mime sniff,
  inline-base64 image blocks, URL-source images, text attachments.
  `FilesClient` async trait + `AnthropicFilesClient` (multipart
  `POST /v1/files` with `anthropic-beta: files-api-2025-04-14`).
  Smart helpers `image_smart` / `pdf_attachment` /
  `text_attachment_via_files` auto-route between inline base64 and
  uploaded file refs based on size. The Anthropic provider auto-adds
  the Files beta header when any block (including those nested in
  `ToolResult`) carries a `file_id`.
- `tokenizer/` — pluggable `Tokenizer` trait. `HeuristicTokenizer`
  (4-ASCII / 1-CJK), `WordSplitTokenizer` (closer-to-BPE for
  English). Real tiktoken plugs in via the trait without dep-tree
  bloat.
- `memdir/` — `MEMORY.md` directory loader. YAML-subset frontmatter,
  4-type taxonomy (User / Feedback / Project / Reference),
  age-bucket relevance scoring, deterministic file scan.
- `mcp/` *(feature `mcp`)* — full Model Context Protocol client
  lifecycle. Server registry, OAuth 2.0 + PKCE, channel permissions,
  server-initiated elicitation, async per-server connect locks,
  stale-handle reconnect repair. Production `RmcpConnector` plugs
  `rmcp` 1.5 into `Lifecycle`: stdio via `TokioChildProcess`,
  HTTP/SSE via `StreamableHttpClientTransport<reqwest::Client>`.
  WebSocket returns a clear "not available in rmcp 1.5" error.
  `RmcpConnection` clones rmcp's `Peer` so concurrent tool calls
  don't serialize on a mutex. Authorization headers forwarded
  verbatim (no Bearer rewriting). Tool result projection:
  `is_error → Err(Connector)`, else `structured_content`, else
  `Vec<Content>`.
- `state/` — `AppStateStore` with broadcast subscribers + projected
  `Selector`.
- `bootstrap/` — schema-versioned migration runner. Refuses to
  downgrade. Idempotent.
- `context_analysis/` — `MessageStore` inspector (token totals by
  role, top-N largest messages, tool-call breakdown).
- `skills/` — frontmatter-loaded prompt templates with input-schema
  validation, optional model override, optional tool allowlist.
- `plugins/` — Plugin trait + registry. Native plugins are full Rust
  trait objects (direct dispatch, no sandboxing overhead).
  Third-party plugins run in a WASM sandbox via the `WasmPluginHost`
  trait — the core crate ships a no-op default; consumers wire
  wasmtime/wasmer in their own crate.
- `remote/` — JSON-RPC 2.0 over line-delimited frames for external
  hosts driving the agent in a separate process. Codec + dispatcher
  + stable method namespace.
- `tasks/` — high-level planning task graph. Cycle-checked
  dependencies, status auto-transitions on completion, ready-task
  query.
- `memory_extract/` — heuristic background extractor for
  `DECISION:` / `User prefers …` / URL-bearing patterns into
  `MemoryCandidate { kind, body, confidence }` for promotion to
  `memdir`.

#### Optional companion — `agent-tools-code` crate

- New workspace crate. Generic coding tool pack — every tool
  implements `agent::tool::Tool` and declares its `SafetyClass`.
  Feature-gated: `fs` (default) / `search` (default) / `shell` /
  `web` / `todo` / `all`. Default features pull only `tokio::fs` +
  `regex` + `ignore`.
- FS pack *(feature `fs`)*: `FileReadTool` (cat -n style line
  numbers, offset/limit defaults 1/2000), `FileWriteTool`
  (idempotent on identical content), `FileEditTool` (refuses
  ambiguous matches unless `replace_all`), `ListDirTool`,
  `MkdirTool`, `MoveTool` (refuses to clobber unless `overwrite`),
  `RemoveTool` (`Destructive`).
- Search pack *(feature `search`)*: `GrepTool` — regex over file
  contents, `ignore::WalkBuilder` for gitignore-aware traversal.
  `RegexBuilder::size_limit` cap (10 MiB) rejects pathological
  patterns at compile time. `tokio::task::spawn_blocking` worker
  polls the abort token at every file boundary.
  `GlobTool` — shell glob with `*` / `**` / `?`. Splits on `/` AND
  `\` so Windows paths match. Adjacent `**` segments are collapsed
  to bound otherwise-exponential backtracking.
- Discovery (always-on): `ToolSearchTool` modeled on Claude Code's
  deferred-discovery pattern. Two query forms — `select:Name1,Name2`
  for direct selection, or a bare keyword for scored search across
  name parts + descriptions. CamelCase + snake_case + `mcp__` name
  parsing, with parity weights against the TS reference.
  Case-insensitive bare-name fallback.
- Shared `WorkspacePolicy` (path containment, file-size cap, symlink
  rules). `register_default(registry, policy)` bulk-registers every
  enabled tool. `WorkspacePolicy::resolve(must_exist=false)` walks
  up to the first existing ancestor and reattaches the missing tail
  lexically — supports `mkdir -p`-style writes whose parent doesn't
  exist yet, while keeping the canonicalize-then-containment-check
  guarantee for the existing prefix.

#### Repo + tooling

- Workspace scaffold: `Cargo.toml`, `rust-toolchain.toml`,
  `rustfmt.toml`, `deny.toml`, `.cargo/config.toml`, GitHub Actions CI.
- `#![forbid(unsafe_code)]` in both crates.
- MIT license.
- `cargo-deny` advisory + license + bans gates run in CI.

### Tests

804 unit + 14 integration + 1 prelude smoke + 5 doc tests. 4 are
`#[ignore]`-gated and only run when their corresponding env var is
set (Anthropic, OpenAI, Ollama, Anthropic Files real-API smoke
tests). `cargo clippy --workspace --all-targets --all-features
-D warnings` clean. `cargo fmt --all -- --check` clean.

### Adversarial review process

Every meaningful diff is run through Codex (or equivalent) review,
typically twice — round 1 catches bugs in the new code, round 2
catches the regressions introduced by round 1's fixes. Roughly half
of commit messages name the bug each round caught. Notable catches
across the work so far:

- async permission path silently ignored structured matchers
  (round 1).
- `RmcpConnection::call_tool` held the service mutex across the
  network await, deadlocking `close()` during slow RPCs (round 1).
- `CostTracker::observe_event` double-counted cumulative `Usage`
  reports (round 1); fix introduced a warm-cache same-cumulative
  undercharge (round 2).
- Anthropic `ToolResult` rendering silently dropped `Document` blocks
  (round 1); same path also dropped multimodal blocks in
  OpenAI/Ollama plain-text fallback (round 1, round 2 expanded).
- Glob matcher only split on `/`, breaking Windows path separators
  (round 1).
- Authorization header was unconditionally rewritten as Bearer auth
  in the MCP HTTP transport (round 1).
