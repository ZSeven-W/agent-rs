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
- `tokenizer/` — pluggable `Tokenizer` trait. Three impls ship in
  the crate. `HeuristicTokenizer` (4-ASCII / 1-CJK) — default,
  zero-dep. `WordSplitTokenizer` — closer-to-BPE for English, no
  fixtures. `TiktokenTokenizer` *(feature `tiktoken`)* — real BPE
  via `tiktoken-rs`, supporting `cl100k_base` (GPT-3.5/4 & Claude
  approximation), `o200k_base` (GPT-4o family / o1), `p50k_base`,
  `r50k_base`. `TiktokenTokenizer::for_model(model_id)` picks the
  encoding heuristically. Off by default — pulls a multi-MB BPE
  vocabulary blob — so hosts opt in only when they need exact
  counts (cost projection, sliding-window trim, prompt precheck).
  For Claude in particular, prefer Anthropic's
  `messages/count_tokens` API for production accounting; the
  `cl100k_base` mapping overestimates by ~5–10% but is fine for
  budgeting.
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
  (idempotent on identical content; bounded read on the no-op
  comparison), `FileEditTool` (refuses ambiguous matches unless
  `replace_all`; pre-read stat + post-edit size cap), `ListDirTool`,
  `MkdirTool`, `MoveTool` (refuses to clobber unless `overwrite`),
  `RemoveTool` (`Destructive`). All read paths bounded by
  `read_with_cap` so a TOCTOU file grow can't OOM the process.
- Search pack *(feature `search`)*: `GrepTool` — regex over file
  contents, `ignore::WalkBuilder` for gitignore-aware traversal.
  `RegexBuilder::size_limit` cap (10 MiB) rejects pathological
  patterns at compile time. `tokio::task::spawn_blocking` worker
  polls the abort token at every file boundary. Symlink-escape
  guard via `entry.file_type()` + canonicalize-and-recheck
  containment, plus a sync `read_file_capped` helper so a TOCTOU
  grow doesn't sneak past the policy cap.
  `GlobTool` — shell glob with `*` / `**` / `?`. Splits on `/` AND
  `\` so Windows paths match. Adjacent `**` segments are collapsed
  to bound otherwise-exponential backtracking. Same symlink-escape
  guard as `GrepTool`.
- Shell pack *(feature `shell`)*: `BashTool` runs `/bin/sh -c`
  (Unix) / `cmd /C` (Windows). Per-stream output capture via a
  `VecDeque<u8>` ring buffer (constant memory regardless of output
  size, tail preserved, `*_truncated` flags). Tail trimmed to a
  valid UTF-8 boundary before formatting. Default 60 s timeout, hard
  ceiling 600 s. On Unix, the child is placed in its own process
  group (`Command::process_group(0)`) so a `/bin/kill -9 -<pgid>`
  on timeout / abort kills descendants too — `kill_on_drop(true)`
  alone wouldn't.
- Web pack *(feature `web`)*: `WebFetchTool` HTTP GET → text /
  HTML. SSRF guard with full DNS-rebinding mitigation: pre-flight
  `tokio::net::lookup_host`, every resolved address screened
  (loopback / RFC 1918 / IPv4 link-local incl. cloud metadata
  169.254.169.254 / IPv4 broadcast / multicast / IPv6 fc00::/7
  unique-local / IPv6 fe80::/10 link-local / IPv4-mapped IPv6 of
  any of the above), then **all** validated addresses pinned via
  `Client::builder().resolve_to_addrs()` so reqwest's connect-time
  DNS lookup can't swap to a private IP. Auto-redirects disabled
  (`Policy::none()`); the redirect chain is walked manually with
  per-hop SSRF re-validation, capped at 5 hops, sharing a single
  `Instant`-based deadline so DNS + redirects can't bypass the
  caller's timeout. Default 30 s timeout (max 120 s), 5 MiB body
  cap (hard ceiling 50 MiB). HTML→text strips `<script>` /
  `<style>` / `<noscript>` and decodes named + numeric entities.
  `allow_private_networks: true` opts out of the guard for hosts
  that legitimately need intranet access.
- Task pack *(feature `task`)*: `TaskTool` lets the model spawn a
  child `QueryLoop` for sub-task delegation. Hosts implement
  `TaskAgentFactory` to enumerate which "agent shapes" the model
  may summon (researcher / reviewer / planner — each can have its
  own provider, model, system prompt, tool registry,
  permissions). The child runs to completion and the aggregated
  assistant text is returned as the tool result, matching Claude
  Code's `Task` semantics. Recursion guard via thread-local depth
  counter (default cap 3, override via `with_max_depth`); abort
  forwarded so cancelling the parent loop short-circuits the
  child.
- Web-search pack *(feature `web-search`)*: `WebSearchTool` with a
  pluggable `WebSearchBackend` trait. Ships `TavilyBackend` as the
  default impl (Tavily Search API, `POST
  https://api.tavily.com/search`); hosts can implement Brave / Bing
  / Kagi / SerpAPI / a private corpus by implementing the trait.
  Output `[{title, url, snippet}]` plus optional aggregated `answer`
  field. Default 5 results / 20 s timeout (hard ceilings 20 / 60).
  Honors `ctx.abort` via `tokio::select!`. Models can pipe each
  result URL into `WebFetchTool` for full-page reading — the
  canonical `WebSearch → WebFetch` pair.
- Notebook pack *(feature `notebook`)*: `NotebookEditTool` —
  cell-level Jupyter `.ipynb` edits. Modes: `replace` (default) /
  `insert` / `delete`. Locate by stable `cell_id` (preferred) or
  zero-based `cell_index`. Round-trips through `serde_json::Value`
  so unknown notebook fields (custom tooling metadata, papermill
  params, etc.) are preserved. Code cells reset
  `outputs` / `execution_count` on replace; markdown / raw cells
  must not carry those fields. Same stat-then-bounded-read +
  WorkspacePolicy size cap as `FileEdit`.
- Todo pack *(feature `todo`)*: `TodoWriteTool` — in-memory shared
  planning list with the same surface as Claude Code's TodoWrite.
  Replaces the list wholesale on each call; emits per-status
  counts; synthesizes ids for items missing one; `with_fresh_state()`
  hands the host a clonable `TodoState` handle for UI binding.
  `TodoStatus` is `#[non_exhaustive]` so future variants aren't a
  SemVer break.
- Discovery (always-on): `ToolSearchTool` modeled on Claude Code's
  deferred-discovery pattern. Two query forms — `select:Name1,Name2`
  for direct selection, or a bare keyword for scored search across
  name parts + descriptions. CamelCase + snake_case + `mcp__` name
  parsing, with parity weights against the TS reference.
  Case-insensitive bare-name fallback.
- Shared `WorkspacePolicy` (path containment, file-size cap, symlink
  rules). `register_default(registry, policy)` bulk-registers every
  enabled tool; `register_default_with_todo(..., todo_state)`
  variant lets the host keep the `TodoState` handle (a
  `tracing::warn!` fires when `todo` is enabled but the plain
  `register_default` is used so the handle isn't silently lost).
  `WorkspacePolicy::resolve(must_exist=false)` walks up to the
  first existing ancestor and reattaches the missing tail
  lexically (rejecting any non-`Component::Normal` segment) —
  supports `mkdir -p`-style writes whose parent doesn't exist yet,
  while keeping the canonicalize-then-containment-check guarantee
  for the existing prefix and refusing `..` smuggling through the
  unresolved tail.

#### Repo + tooling

- Workspace scaffold: `Cargo.toml`, `rust-toolchain.toml`,
  `rustfmt.toml`, `deny.toml`, `.cargo/config.toml`, GitHub Actions CI.
- `#![forbid(unsafe_code)]` in both crates.
- MIT license.
- `cargo-deny` advisory + license + bans gates run in CI.

### Tests

743 unit (agent) + 101 unit (agent-tools-code) + 13 integration +
1 prelude smoke + 5 doc tests. 4 are `#[ignore]`-gated and only
run when their corresponding env var is set (Anthropic, OpenAI,
Ollama, Anthropic Files real-API smoke tests). `cargo clippy
--workspace --all-targets --all-features -D warnings` clean.
`cargo fmt --all -- --check` clean. CI green on Linux + macOS +
Windows.
