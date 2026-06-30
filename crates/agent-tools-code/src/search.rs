//! Search tools: Grep (regex over file contents) + Glob (path
//! pattern match).
//!
//! Both are `ReadOnly`. They use the `ignore` crate so gitignore +
//! `.git` exclusions Just Work — agents don't accidentally match
//! against `target/` or `node_modules/`. The host can override
//! gitignore behavior via the `respect_gitignore` flag.
//!
//! Output is intentionally compact:
//!
//! - `Grep` returns `[{path, line_no, match_text}]` so the model can
//!   feed line ranges into [`crate::fs::FileReadTool`].
//! - `Glob` returns `[path]`.
//!
//! Both tools cap their result count at `MAX_RESULTS` (default 1000)
//! to keep the model from drowning in matches; the response carries
//! a `truncated: true` flag when the cap is hit.

use std::sync::Arc;

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::json;

use crate::policy::WorkspacePolicy;

/// Synchronous bounded file read for the `Grep` blocking walker.
/// Reads at most `cap + 1` bytes from `path`. Caller compares the
/// resulting buffer length against `cap` to detect overflow:
/// `out.len() > cap` means the file grew past the cap between the
/// stat and the read (TOCTOU race) and should be skipped.
fn read_file_capped(path: &std::path::Path, cap: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let f = std::fs::File::open(path)?;
    let limit = cap.saturating_add(1);
    let cap_usize = usize::try_from(cap).unwrap_or(usize::MAX);
    let mut buf = Vec::with_capacity(cap_usize.min(64 * 1024));
    f.take(limit).read_to_end(&mut buf)?;
    Ok(buf)
}

const MAX_RESULTS: usize = 1000;

/// Hard cap on the compiled regex size in bytes. Bounds the worst
/// case for hostile patterns (huge alternations, deeply nested
/// captures) before we even run the search. 10 MiB is more than
/// enough for any reasonable user pattern; pathological ones get
/// rejected at compile time.
const REGEX_SIZE_LIMIT: usize = 10 * 1024 * 1024;

fn policy_to_agent_err(e: crate::policy::PolicyError) -> AgentError {
    AgentError::other(format!("policy: {e}"))
}

// =========================================================================
// Grep
// =========================================================================

/// Search file contents for a regex pattern, returning matching
/// `(path, line_no, line_text)` tuples. Walks the directory tree
/// rooted at `path` (default: workspace cwd) using the `ignore`
/// crate, so gitignore + `.git` are skipped by default.
#[derive(Debug)]
pub struct GrepTool {
    policy: Arc<WorkspacePolicy>,
}

impl GrepTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct GrepInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    /// **Substring** filter on file paths — NOT a glob. `*.rs`
    /// would never match anything (paths don't literally contain
    /// `*.rs`); pass `.rs` instead. The flag is named after the
    /// `--include` flag of `ripgrep` / `grep`, which on agent-rs is
    /// kept simple (no globset dep) — for true glob filtering, run
    /// `Glob` first and feed the matches into `FileRead`.
    #[serde(default)]
    include: Option<String>,
    /// Case-insensitive match. Default false.
    #[serde(default)]
    ignore_case: bool,
    /// Honor `.gitignore`. Default true. Set false to walk every
    /// file under the path (including `target/`, `node_modules/`).
    #[serde(default = "default_true")]
    respect_gitignore: bool,
    /// Max matches to return. Default 1000.
    #[serde(default)]
    max_matches: Option<usize>,
}

fn default_true() -> bool {
    true
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "Grep"
    }
    fn description(&self) -> &str {
        "Search file contents for a regex pattern. Returns (path, line_no, line) tuples; walks the directory tree honoring .gitignore by default."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "Search root (default: workspace cwd)."},
                "include": {"type": "string", "description": "Substring filter on file path (NOT a glob — pass '.rs' not '*.rs')."},
                "ignore_case": {"type": "boolean", "default": false},
                "respect_gitignore": {"type": "boolean", "default": true},
                "max_matches": {"type": "integer", "minimum": 1}
            },
            "required": ["pattern"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    async fn call(
        &self,
        ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: GrepInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("Grep invalid input: {e}")))?;
        let root_str = parsed.path.as_deref().unwrap_or(".");
        let root = self
            .policy
            .resolve_read(root_str)
            .map_err(policy_to_agent_err)?;

        // Use RegexBuilder so case-insensitivity is enforced at the
        // engine level — a user pattern starting with `(?-i)` would
        // otherwise cancel an inline `(?i)` prefix. The size_limit
        // cap rejects pathologically large compiled regexes (huge
        // alternations / deeply nested captures) before we run.
        let regex = RegexBuilder::new(&parsed.pattern)
            .case_insensitive(parsed.ignore_case)
            .size_limit(REGEX_SIZE_LIMIT)
            .build()
            .map_err(|e| AgentError::other(format!("Grep invalid pattern: {e}")))?;

        let max_matches = parsed.max_matches.unwrap_or(MAX_RESULTS).min(MAX_RESULTS);
        let policy = self.policy.clone();
        let include_filter = parsed.include.clone();
        let respect_gi = parsed.respect_gitignore;

        // Walk + read on a blocking pool so the async runtime stays
        // free. `ignore::WalkBuilder` is sync; tokio::task::spawn_blocking
        // is the canonical bridge. We poll the abort token at every
        // file boundary so a host-cancelled query stops within one
        // file's worth of work — meaningful for big trees.
        let abort = ctx.abort.clone();
        let abort_for_blocking = abort.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut walker = WalkBuilder::new(&root);
            walker
                .git_ignore(respect_gi)
                .git_global(respect_gi)
                .git_exclude(respect_gi)
                .ignore(respect_gi)
                .hidden(false)
                .follow_links(false);
            let mut matches: Vec<serde_json::Value> = Vec::new();
            let mut truncated = false;
            let mut aborted = false;
            for dent in walker.build() {
                if abort_for_blocking.is_aborted() {
                    aborted = true;
                    break;
                }
                let entry = match dent {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let path = entry.path();
                // Skip non-files AND symlinks. `Path::is_file` follows
                // symlinks via `metadata`, so a symlink in the
                // workspace pointing to `/etc/passwd` would otherwise
                // get read. With `follow_links(false)` on the walker,
                // `entry.file_type()` returns the symlink's own type.
                match entry.file_type() {
                    Some(ft) if ft.is_file() => {}
                    _ => continue,
                }
                if let Some(filter) = &include_filter {
                    if !path.to_string_lossy().contains(filter.as_str()) {
                        continue;
                    }
                }
                // Lexical containment + canonicalize-and-check. The
                // canonicalize step defeats any symlinked file inside
                // the tree that points outside the workspace; the
                // file_type check above already filters most of those,
                // but this is the belt-and-braces guarantee.
                if policy.allowed_roots.iter().all(|r| !path.starts_with(r)) {
                    continue;
                }
                let canon = match std::fs::canonicalize(path) {
                    Ok(canon) if policy.allowed_roots.iter().any(|r| canon.starts_with(r)) => canon,
                    _ => continue,
                };
                // Strict-read: skip files under a read-denied (credential) subtree.
                // Grep only resolve_read's the root, so this per-file check is what
                // stops grep from surfacing ~/.ssh etc. when an allowed root is an
                // ancestor. Check lexical + canonical (a symlink resolves in).
                if policy.is_read_denied(path) || policy.is_read_denied(&canon) {
                    continue;
                }
                let meta = match std::fs::metadata(path) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if policy.check_size(meta.len()).is_err() {
                    continue;
                }
                // Bounded read so a TOCTOU grow between the stat
                // above and this read can't pull a multi-GB file
                // into RAM. Files that grew past the cap are
                // skipped — they would have been skipped anyway.
                let bytes = match read_file_capped(path, policy.max_file_size_bytes) {
                    Ok(b) if (b.len() as u64) <= policy.max_file_size_bytes => b,
                    _ => continue,
                };
                let text = match String::from_utf8(bytes) {
                    Ok(t) => t,
                    Err(_) => continue, // skip non-UTF-8 files silently
                };
                for (i, line) in text.lines().enumerate() {
                    if regex.is_match(line) {
                        if matches.len() >= max_matches {
                            truncated = true;
                            break;
                        }
                        matches.push(json!({
                            "path": path.display().to_string(),
                            "line_no": (i as u64).saturating_add(1),
                            "match_text": line,
                        }));
                    }
                }
                if truncated {
                    break;
                }
            }
            (matches, truncated, aborted)
        })
        .await
        .map_err(|e| AgentError::other(format!("Grep join error: {e}")))?;
        if result.2 {
            return Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            ));
        }

        Ok(json!({
            "pattern": parsed.pattern,
            "matches": result.0,
            "truncated": result.1,
        }))
    }
}

// =========================================================================
// Glob
// =========================================================================

/// Find file paths matching a shell-style glob. Like `Grep`, walks
/// the tree honoring `.gitignore` by default.
#[derive(Debug)]
pub struct GlobTool {
    policy: Arc<WorkspacePolicy>,
}

impl GlobTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct GlobInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default = "default_true")]
    respect_gitignore: bool,
    #[serde(default)]
    max_matches: Option<usize>,
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "Glob"
    }
    fn description(&self) -> &str {
        "Find files whose path matches a shell-style glob. Walks the directory tree honoring .gitignore by default."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Shell glob (e.g., **/*.rs)."},
                "path": {"type": "string"},
                "respect_gitignore": {"type": "boolean", "default": true},
                "max_matches": {"type": "integer", "minimum": 1}
            },
            "required": ["pattern"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    async fn call(
        &self,
        ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: GlobInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("Glob invalid input: {e}")))?;
        let root_str = parsed.path.as_deref().unwrap_or(".");
        let root = self
            .policy
            .resolve_read(root_str)
            .map_err(policy_to_agent_err)?;
        let policy = self.policy.clone();
        let pattern = parsed.pattern.clone();
        let max_matches = parsed.max_matches.unwrap_or(MAX_RESULTS).min(MAX_RESULTS);
        let respect_gi = parsed.respect_gitignore;
        let abort = ctx.abort.clone();
        let abort_for_blocking = abort.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut walker = WalkBuilder::new(&root);
            walker
                .git_ignore(respect_gi)
                .git_global(respect_gi)
                .git_exclude(respect_gi)
                .ignore(respect_gi)
                .hidden(false)
                .follow_links(false);
            let mut matches: Vec<serde_json::Value> = Vec::new();
            let mut truncated = false;
            let mut aborted = false;
            for dent in walker.build() {
                if abort_for_blocking.is_aborted() {
                    aborted = true;
                    break;
                }
                let entry = match dent {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let path = entry.path();
                // Same symlink/escape guards as Grep — see search.rs
                // line ~175 for the rationale.
                match entry.file_type() {
                    Some(ft) if ft.is_file() => {}
                    _ => continue,
                }
                if policy.allowed_roots.iter().all(|r| !path.starts_with(r)) {
                    continue;
                }
                let canon = match std::fs::canonicalize(path) {
                    Ok(canon) if policy.allowed_roots.iter().any(|r| canon.starts_with(r)) => canon,
                    _ => continue,
                };
                // Strict-read: don't surface paths under a read-denied (credential)
                // subtree (Glob only resolve_read's the root). Lexical + canonical.
                if policy.is_read_denied(path) || policy.is_read_denied(&canon) {
                    continue;
                }
                // Match the pattern against the path relative to the
                // search root — that's what shell-style globs assume.
                let relative = path.strip_prefix(&root).unwrap_or(path);
                if glob_match(&pattern, &relative.to_string_lossy()) {
                    if matches.len() >= max_matches {
                        truncated = true;
                        break;
                    }
                    matches.push(json!({"path": path.display().to_string()}));
                }
            }
            (matches, truncated, aborted)
        })
        .await
        .map_err(|e| AgentError::other(format!("Glob join error: {e}")))?;
        if result.2 {
            return Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            ));
        }

        Ok(json!({
            "pattern": parsed.pattern,
            "matches": result.0,
            "truncated": result.1,
        }))
    }
}

/// Hand-rolled shell-glob matcher. Supports `*` (any chars within a
/// segment), `**` (any chars including path separators), `?`
/// (single char). Anchored on both ends. Char-based for UTF-8
/// safety. Splits both pattern and text on `/` AND `\` so Windows
/// paths match correctly. Adjacent `**` segments are collapsed to a
/// single `**` to bound otherwise-exponential backtracking on
/// `a/**/**/**/x`-style patterns.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p_segs: Vec<&str> = pattern.split(['/', '\\']).collect();
    // Collapse runs of "**" to a single "**" — semantically
    // equivalent and avoids combinatorial recursion blowup.
    let mut p_collapsed: Vec<&str> = Vec::with_capacity(p_segs.len());
    for seg in p_segs {
        if seg == "**" && p_collapsed.last().map(|s| *s == "**").unwrap_or(false) {
            continue;
        }
        p_collapsed.push(seg);
    }
    let t_segs: Vec<&str> = text.split(['/', '\\']).collect();
    glob_segments(&p_collapsed, &t_segs)
}

fn glob_segments(p: &[&str], t: &[&str]) -> bool {
    if p.is_empty() {
        return t.is_empty();
    }
    if p[0] == "**" {
        // ** matches zero or more segments.
        for i in 0..=t.len() {
            if glob_segments(&p[1..], &t[i..]) {
                return true;
            }
        }
        return false;
    }
    if t.is_empty() {
        return false;
    }
    if !glob_match_segment(p[0], t[0]) {
        return false;
    }
    glob_segments(&p[1..], &t[1..])
}

/// Single-segment glob match — `*` matches within one path component
/// only, `?` matches exactly one char.
fn glob_match_segment(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star_p, mut star_t): (Option<usize>, usize) = (None, 0);
    while t < txt.len() {
        if p < pat.len() && (pat[p] == '?' || pat[p] == txt[t]) {
            p += 1;
            t += 1;
        } else if p < pat.len() && pat[p] == '*' {
            star_p = Some(p);
            star_t = t;
            p += 1;
        } else if let Some(sp) = star_p {
            p = sp + 1;
            star_t += 1;
            t = star_t;
        } else {
            return false;
        }
    }
    while p < pat.len() && pat[p] == '*' {
        p += 1;
    }
    p == pat.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::abort::AbortController;
    use std::num::NonZeroUsize;
    use std::path::Path;
    use tempfile::TempDir;

    fn ctx() -> ToolUseContext {
        ToolUseContext {
            cwd: std::env::current_dir().unwrap(),
            abort: AbortController::new(),
            file_cache: Arc::new(agent::file_cache::FileStateCache::new(
                NonZeroUsize::new(8).unwrap(),
                1024 * 1024,
            )),
            permissions: Arc::new(agent::permission::PermissionManager::new()),
            hooks: Arc::new(agent::hook::HookRunner::new()),
        }
    }

    fn policy_for(dir: &Path) -> Arc<WorkspacePolicy> {
        WorkspacePolicy::new(dir).unwrap().into_arc()
    }

    fn tree(dir: &TempDir) {
        std::fs::write(dir.path().join("a.txt"), "alpha\nbeta\ngamma\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn main() { println!(\"hi\"); }\n").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/c.rs"), "fn nested() {}\n").unwrap();
    }

    #[test]
    fn glob_segment_basic() {
        assert!(glob_match_segment("*.rs", "main.rs"));
        assert!(glob_match_segment("foo?bar", "fooXbar"));
        assert!(!glob_match_segment("foo?bar", "fooXXbar"));
        assert!(glob_match_segment("a*b*c", "axxxbyyyc"));
        assert!(!glob_match_segment("foo", "FOO"));
    }

    #[test]
    fn glob_double_star_matches_across_segments() {
        assert!(glob_match("**/*.rs", "src/main.rs"));
        assert!(glob_match("**/*.rs", "deep/sub/dir/x.rs"));
        assert!(glob_match("**/*.rs", "x.rs"));
        assert!(!glob_match("**/*.rs", "x.txt"));
        assert!(glob_match("src/**/lib.rs", "src/a/b/lib.rs"));
        assert!(!glob_match("src/**/lib.rs", "other/lib.rs"));
    }

    #[test]
    fn glob_single_star_does_not_cross_segments() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "src/main.rs"));
    }

    #[tokio::test]
    async fn grep_finds_matches_with_line_numbers() {
        let dir = TempDir::new().unwrap();
        tree(&dir);
        let tool = GrepTool::new(policy_for(dir.path()));
        let out = tool.call(&ctx(), json!({"pattern": "fn"})).await.unwrap();
        let matches = out["matches"].as_array().unwrap();
        assert!(!matches.is_empty());
        assert!(matches
            .iter()
            .any(|m| m["match_text"].as_str().unwrap().contains("fn main")));
    }

    #[tokio::test]
    async fn grep_ignore_case_works() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "Hello World\n").unwrap();
        let tool = GrepTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "hello", "ignore_case": true}))
            .await
            .unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn grep_include_filter_substring() {
        let dir = TempDir::new().unwrap();
        tree(&dir);
        let tool = GrepTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "fn", "include": ".rs"}))
            .await
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        assert!(!matches.is_empty());
        assert!(matches
            .iter()
            .all(|m| m["path"].as_str().unwrap().ends_with(".rs")));
    }

    #[tokio::test]
    async fn grep_truncates_at_max_matches() {
        let dir = TempDir::new().unwrap();
        let lines: Vec<String> = (0..20).map(|_| "match-this-line".to_string()).collect();
        std::fs::write(dir.path().join("a.txt"), lines.join("\n")).unwrap();
        let tool = GrepTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "match", "max_matches": 5}))
            .await
            .unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 5);
        assert_eq!(out["truncated"], true);
    }

    #[tokio::test]
    async fn grep_skips_oversized_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("big.txt"),
            "needle\n".repeat(2 * 1024 * 1024),
        )
        .unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_max_file_size(1024)
            .into_arc();
        let tool = GrepTool::new(policy);
        let out = tool
            .call(&ctx(), json!({"pattern": "needle"}))
            .await
            .unwrap();
        assert!(out["matches"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn grep_invalid_regex_errors() {
        let dir = TempDir::new().unwrap();
        let tool = GrepTool::new(policy_for(dir.path()));
        let err = tool
            .call(&ctx(), json!({"pattern": "(unclosed"}))
            .await
            .expect_err("invalid regex");
        assert!(err.to_string().contains("invalid pattern"));
    }

    #[tokio::test]
    async fn grep_skips_non_utf8_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("text.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("bin.dat"), [0xff, 0xfe, 0xff, 0xfe]).unwrap();
        let tool = GrepTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "needle"}))
            .await
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 1);
        assert!(matches[0]["path"].as_str().unwrap().ends_with("text.txt"));
    }

    #[tokio::test]
    async fn glob_finds_rust_files() {
        let dir = TempDir::new().unwrap();
        tree(&dir);
        let tool = GlobTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "**/*.rs"}))
            .await
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches
            .iter()
            .all(|m| m["path"].as_str().unwrap().ends_with(".rs")));
    }

    #[tokio::test]
    async fn glob_truncates_at_max_matches() {
        let dir = TempDir::new().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "x").unwrap();
        }
        let tool = GlobTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "*.txt", "max_matches": 3}))
            .await
            .unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 3);
        assert_eq!(out["truncated"], true);
    }

    #[test]
    fn glob_handles_windows_path_separators() {
        // Codex round-1 confirmed bug n-extra1: paths with `\\`
        // separators must split correctly.
        assert!(glob_match("**/*.rs", "src\\main.rs"));
        assert!(glob_match("src/**/lib.rs", "src\\a\\b\\lib.rs"));
        assert!(!glob_match("*.rs", "src\\main.rs"));
    }

    #[test]
    fn glob_collapses_repeated_double_stars() {
        // Codex round-1 design gap n-extra2: pathological pattern
        // shouldn't combinatorially explode.
        assert!(glob_match("**/**/**/x.rs", "a/b/c/x.rs"));
        assert!(glob_match("a/**/**/**/x", "a/b/c/x"));
    }

    #[tokio::test]
    async fn grep_ignore_case_via_regex_builder() {
        // ignore_case is wired through RegexBuilder.case_insensitive
        // so the engine does the lowercasing, no `(?i)` prefix
        // injection. Per regex-1.x semantics, an inline `(?-i)` in
        // the user's own pattern can still locally cancel the flag —
        // matches `grep --ignore-case` behavior, documented in the
        // tool description. This test pins the common case.
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "Hello\nGOODBYE\n").unwrap();
        let tool = GrepTool::new(policy_for(dir.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "hello", "ignore_case": true}))
            .await
            .unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 1);
        let out = tool
            .call(&ctx(), json!({"pattern": "goodbye", "ignore_case": true}))
            .await
            .unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn grep_skips_read_denied_subtree() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("public.txt"), "NEEDLE here\n").unwrap();
        let secrets = dir.path().join("secrets");
        std::fs::create_dir_all(&secrets).unwrap();
        std::fs::write(secrets.join("key.txt"), "NEEDLE secret\n").unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_read_denied_subpath(&secrets)
            .into_arc();
        let out = GrepTool::new(policy)
            .call(&ctx(), json!({"pattern": "NEEDLE"}))
            .await
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        // The read-denied subtree is skipped even though it's under the root.
        assert_eq!(matches.len(), 1, "got {matches:?}");
        assert!(matches[0]["path"].as_str().unwrap().ends_with("public.txt"));
    }

    #[tokio::test]
    async fn glob_skips_read_denied_subtree() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        let secrets = dir.path().join("secrets");
        std::fs::create_dir_all(&secrets).unwrap();
        std::fs::write(secrets.join("b.rs"), "").unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_read_denied_subpath(&secrets)
            .into_arc();
        let out = GlobTool::new(policy)
            .call(&ctx(), json!({"pattern": "**/*.rs"}))
            .await
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        assert!(
            matches
                .iter()
                .all(|m| !m["path"].as_str().unwrap().contains("secrets")),
            "read-denied subtree must not appear: {matches:?}"
        );
        assert!(matches
            .iter()
            .any(|m| m["path"].as_str().unwrap().ends_with("a.rs")));
    }

    #[tokio::test]
    async fn grep_respects_gitignore_false_walks_ignored_dirs() {
        let dir = TempDir::new().unwrap();
        // `ignore::WalkBuilder` reads `.gitignore` only when a `.git`
        // marker is present somewhere in the ancestry. Plant an empty
        // `.git` directory so the test reflects the real-world repo
        // shape.
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored/\n").unwrap();
        std::fs::create_dir(dir.path().join("ignored")).unwrap();
        std::fs::write(dir.path().join("ignored/a.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "needle\n").unwrap();
        let tool = GrepTool::new(policy_for(dir.path()));
        // Default (respect_gitignore=true) → only visible.txt
        let out = tool
            .call(&ctx(), json!({"pattern": "needle"}))
            .await
            .unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 1);
        // respect_gitignore=false → both
        let out = tool
            .call(
                &ctx(),
                json!({"pattern": "needle", "respect_gitignore": false}),
            )
            .await
            .unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn grep_aborts_when_ctx_abort_fires_before_call() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "needle\n").unwrap();
        let c = ctx();
        c.abort.abort_with_reason("test cancel");
        let tool = GrepTool::new(policy_for(dir.path()));
        let err = tool
            .call(&c, json!({"pattern": "needle"}))
            .await
            .expect_err("aborted");
        assert!(matches!(err, AgentError::Aborted(_)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn grep_skips_symlinks_pointing_outside_workspace() {
        // Plant a symlink inside the workspace whose target is a file
        // OUTSIDE the workspace containing a needle. Grep must NOT
        // surface the needle, since following the symlink would
        // disclose data outside the policy roots.
        let outside = TempDir::new().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "needle-outside\n").unwrap();
        let inside = TempDir::new().unwrap();
        std::fs::write(inside.path().join("regular.txt"), "needle-inside\n").unwrap();
        std::os::unix::fs::symlink(&secret, inside.path().join("link.txt")).unwrap();
        let tool = GrepTool::new(policy_for(inside.path()));
        let out = tool
            .call(&ctx(), json!({"pattern": "needle"}))
            .await
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        // Only the regular file matches, not the symlink-to-secret.
        assert_eq!(matches.len(), 1);
        assert!(matches[0]["match_text"]
            .as_str()
            .unwrap()
            .contains("inside"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn glob_skips_symlinks_pointing_outside_workspace() {
        let outside = TempDir::new().unwrap();
        let secret = outside.path().join("secret.rs");
        std::fs::write(&secret, "fn x(){}").unwrap();
        let inside = TempDir::new().unwrap();
        std::fs::write(inside.path().join("regular.rs"), "fn y(){}").unwrap();
        std::os::unix::fs::symlink(&secret, inside.path().join("escaped.rs")).unwrap();
        let tool = GlobTool::new(policy_for(inside.path()));
        let out = tool.call(&ctx(), json!({"pattern": "*.rs"})).await.unwrap();
        let matches = out["matches"].as_array().unwrap();
        let names: Vec<&str> = matches
            .iter()
            .map(|m| m["path"].as_str().unwrap())
            .collect();
        assert!(names.iter().any(|p| p.ends_with("regular.rs")));
        assert!(
            !names.iter().any(|p| p.ends_with("escaped.rs")),
            "symlink escape leaked: {names:?}"
        );
    }

    #[tokio::test]
    async fn grep_classified_read_only() {
        let dir = TempDir::new().unwrap();
        let tool = GrepTool::new(policy_for(dir.path()));
        assert_eq!(tool.safety_class(), SafetyClass::ReadOnly);
    }
}
