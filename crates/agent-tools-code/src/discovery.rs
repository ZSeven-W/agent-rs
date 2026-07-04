//! `ToolSearch` — keyword/select discovery over a `ToolRegistry`.
//!
//! Modeled on Claude Code's `ToolSearchTool`
//! (`src/tools/ToolSearchTool/ToolSearchTool.ts`). The model uses
//! it to narrow down a large pool of registered tools to a handful
//! relevant to the current task — useful when a host registers
//! 50+ MCP tools and doesn't want every one taking up the model's
//! tool-selection budget.
//!
//! Two query forms:
//!
//! - `select:Name1,Name2,...` — direct selection. Returns each
//!   matching tool name verbatim. Comma-separated lets the model
//!   request a small batch in one shot.
//! - bare keyword — keyword-scored search over tool names +
//!   descriptions. Returns the top `max_results` matches.
//!
//! Scoring (per Claude Code parity):
//!
//! | Signal                       | Weight |
//! |------------------------------|--------|
//! | name part exact match (mcp__) | 12    |
//! | name part exact match         | 10    |
//! | name part substring (mcp__)   |  6    |
//! | name part substring           |  5    |
//! | full-name fallback            |  3    |
//! | description word-boundary     |  2    |
//!
//! Output:
//!
//! ```json
//! { "matches": ["GitCommit", "GitDiff"], "query": "git", "total_candidate_tools": 50 }
//! ```
//!
//! # How a host integrates this
//!
//! Two registries:
//!
//! 1. **Active** — small set always-exposed (FileRead/FileEdit/Bash
//!    plus `ToolSearchTool` itself).
//! 2. **Candidate** — large pool the model can discover via
//!    `ToolSearchTool` and then promote.
//!
//! When `ToolSearchTool` returns a name, the host promotes the
//! matching tool from candidate → active so the next provider
//! request includes it. Promotion is host-side because the
//! `ToolRegistry` itself is owned by the host's session.

use std::sync::Arc;

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolRegistry, ToolUseContext};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

/// Discovery tool. Constructed with an `Arc<ToolRegistry>` of
/// candidate tools. Cheap to clone — the registry pointer is
/// shared.
#[derive(Debug, Clone)]
pub struct ToolSearchTool {
    candidates: Arc<ToolRegistry>,
}

impl ToolSearchTool {
    pub fn new(candidates: Arc<ToolRegistry>) -> Self {
        Self { candidates }
    }
}

#[derive(Debug, Deserialize)]
struct ToolSearchInput {
    query: String,
    #[serde(default)]
    max_results: Option<usize>,
}

const DEFAULT_MAX_RESULTS: usize = 5;

#[async_trait]
impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "ToolSearch"
    }
    fn description(&self) -> &str {
        "Discover tools by keyword or by exact selection. Use 'select:Name1,Name2' to load specific tools, or pass a keyword query to search names + descriptions."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "default": 5}
            },
            "required": ["query"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: ToolSearchInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("ToolSearch invalid input: {e}")))?;
        let max = parsed.max_results.unwrap_or(DEFAULT_MAX_RESULTS).max(1);
        let total = self.candidates.len();

        // select:Name1,Name2 — exact match path.
        if let Some(rest) = parsed.query.strip_prefix("select:") {
            return Ok(self.handle_select(rest, total));
        }
        // ALSO accept a bare exact name (case-insensitive) as a
        // select shortcut. Models sometimes drop the prefix and
        // sometimes lowercase the name.
        let trimmed = parsed.query.trim();
        let exact = self
            .candidates
            .list()
            .into_iter()
            .find(|t| t.name().eq_ignore_ascii_case(trimmed));
        if let Some(tool) = exact {
            return Ok(json!({
                "matches": [tool.name()],
                "query": parsed.query,
                "total_candidate_tools": total,
            }));
        }
        let matches = self.keyword_search(&parsed.query, max);
        Ok(json!({
            "matches": matches,
            "query": parsed.query,
            "total_candidate_tools": total,
        }))
    }
}

impl ToolSearchTool {
    fn handle_select(&self, rest: &str, total: usize) -> serde_json::Value {
        let mut found: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        for raw in rest.split(',') {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            if let Some(tool) = self.candidates.get(name) {
                let n = tool.name().to_string();
                if !found.contains(&n) {
                    found.push(n);
                }
            } else {
                missing.push(name.to_string());
            }
        }
        let mut out = json!({
            "matches": found,
            "query": format!("select:{rest}"),
            "total_candidate_tools": total,
        });
        if !missing.is_empty() {
            out["missing"] = json!(missing);
        }
        out
    }

    fn keyword_search(&self, query: &str, max: usize) -> Vec<String> {
        let query_lower = query.to_ascii_lowercase();
        let terms: Vec<String> = query_lower
            .split_whitespace()
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect();
        if terms.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(String, u32)> = Vec::new();
        for tool in self.candidates.list() {
            let parsed = parse_tool_name(tool.name());
            let desc_lower = tool.description().to_ascii_lowercase();
            let mut score: u32 = 0;
            for term in &terms {
                // Name-part exact match.
                if parsed.parts.iter().any(|p| p == term) {
                    score += if parsed.is_mcp { 12 } else { 10 };
                } else if parsed.parts.iter().any(|p| p.contains(term)) {
                    score += if parsed.is_mcp { 6 } else { 5 };
                }
                // Full-name fallback (only if no part matched).
                if score == 0 && parsed.full.contains(term) {
                    score += 3;
                }
                // Description word-boundary match.
                if word_boundary_contains(&desc_lower, term) {
                    score += 2;
                }
            }
            if score > 0 {
                scored.push((tool.name().to_string(), score));
            }
        }
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.into_iter().take(max).map(|(n, _)| n).collect()
    }
}

#[derive(Debug)]
struct ParsedName {
    parts: Vec<String>,
    full: String,
    is_mcp: bool,
}

/// Split a tool name into searchable lowercase parts.
///
/// - MCP tools (`mcp__server__action`) split on `__` then on `_`.
/// - Regular tools split CamelCase + `_`.
fn parse_tool_name(name: &str) -> ParsedName {
    if let Some(rest) = name.strip_prefix("mcp__") {
        let lower = rest.to_ascii_lowercase();
        let parts: Vec<String> = lower
            .split("__")
            .flat_map(|p| p.split('_').map(|s| s.to_string()))
            .filter(|s| !s.is_empty())
            .collect();
        let full = lower.replace("__", " ").replace('_', " ");
        return ParsedName {
            parts,
            full,
            is_mcp: true,
        };
    }
    // CamelCase + underscore split.
    let mut spaced = String::with_capacity(name.len() + 8);
    let mut prev_lower = false;
    for c in name.chars() {
        if c == '_' {
            spaced.push(' ');
            prev_lower = false;
            continue;
        }
        if prev_lower && c.is_uppercase() {
            spaced.push(' ');
        }
        spaced.push(c);
        prev_lower = c.is_lowercase();
    }
    let lower = spaced.to_ascii_lowercase();
    let parts: Vec<String> = lower.split_whitespace().map(|s| s.to_string()).collect();
    ParsedName {
        parts,
        full: lower,
        is_mcp: false,
    }
}

/// `true` if `term` appears in `haystack` flanked by non-word
/// characters (or the string boundary). Cheap word-boundary check
/// without dragging in `regex` for this single use.
fn word_boundary_contains(haystack: &str, term: &str) -> bool {
    let mut start = 0;
    while let Some(idx) = haystack[start..].find(term) {
        let abs = start + idx;
        let before_ok = abs == 0
            || !haystack
                .as_bytes()
                .get(abs - 1)
                .map(|b| (*b as char).is_alphanumeric() || *b == b'_')
                .unwrap_or(false);
        let after = abs + term.len();
        let after_ok = after == haystack.len()
            || !haystack
                .as_bytes()
                .get(after)
                .map(|b| (*b as char).is_alphanumeric() || *b == b'_')
                .unwrap_or(false);
        if before_ok && after_ok {
            return true;
        }
        start = abs + term.len();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::abort::AbortController;
    use agent::error::AgentError;
    use std::num::NonZeroUsize;

    /// Mock tool — tests only need a name + description.
    #[derive(Debug)]
    struct MockTool {
        name_: &'static str,
        desc: &'static str,
    }
    impl MockTool {
        fn new(name: &'static str, desc: &'static str) -> Self {
            Self { name_: name, desc }
        }
    }
    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            self.name_
        }
        fn description(&self) -> &str {
            self.desc
        }
        fn input_schema(&self) -> serde_json::Value {
            json!({"type": "object"})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            Ok(json!({}))
        }
    }

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
            task_depth: 0,
        }
    }

    fn registry_with(tools: Vec<(&'static str, &'static str)>) -> Arc<ToolRegistry> {
        let mut r = ToolRegistry::new();
        for (name, desc) in tools {
            r.register(Arc::new(MockTool::new(name, desc)) as Arc<dyn Tool>);
        }
        Arc::new(r)
    }

    #[test]
    fn parse_camel_case_tool_name() {
        let p = parse_tool_name("FileEdit");
        assert_eq!(p.parts, vec!["file".to_string(), "edit".to_string()]);
        assert_eq!(p.full, "file edit");
        assert!(!p.is_mcp);
    }

    #[test]
    fn parse_underscore_tool_name() {
        let p = parse_tool_name("git_commit");
        assert_eq!(p.parts, vec!["git".to_string(), "commit".to_string()]);
        assert_eq!(p.full, "git commit");
        assert!(!p.is_mcp);
    }

    #[test]
    fn parse_mcp_tool_name() {
        let p = parse_tool_name("mcp__github__create_issue");
        assert_eq!(
            p.parts,
            vec![
                "github".to_string(),
                "create".to_string(),
                "issue".to_string()
            ]
        );
        assert_eq!(p.full, "github create issue");
        assert!(p.is_mcp);
    }

    #[test]
    fn word_boundary_basic() {
        // Standard `\b` semantics — `_` and alphanumerics are word
        // chars; punctuation / spaces are boundaries. Matches what
        // Claude Code does for description scoring.
        assert!(word_boundary_contains("hello world", "world"));
        assert!(!word_boundary_contains("worldview", "world"));
        assert!(!word_boundary_contains("aworldb", "world"));
        assert!(word_boundary_contains("(world)", "world"));
        assert!(word_boundary_contains("hello, world!", "world"));
        // Underscores ARE word characters, so the_world_is doesn't
        // surface "world" as a word — for snake_case names rely on
        // `parse_tool_name` part splitting instead.
        assert!(!word_boundary_contains("the_world_is", "world"));
    }

    #[tokio::test]
    async fn select_returns_exact_match() {
        let reg = registry_with(vec![
            ("FileRead", "read"),
            ("FileWrite", "write"),
            ("Bash", "shell"),
        ]);
        let tool = ToolSearchTool::new(reg);
        let out = tool
            .call(&ctx(), json!({"query": "select:FileRead,Bash"}))
            .await
            .unwrap();
        let matches = out["matches"].as_array().unwrap();
        let names: Vec<&str> = matches.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(names.contains(&"FileRead"));
        assert!(names.contains(&"Bash"));
        assert_eq!(out["total_candidate_tools"], 3);
    }

    #[tokio::test]
    async fn select_reports_missing_names() {
        let reg = registry_with(vec![("FileRead", "read")]);
        let tool = ToolSearchTool::new(reg);
        let out = tool
            .call(&ctx(), json!({"query": "select:FileRead,DoesNotExist"}))
            .await
            .unwrap();
        let names: Vec<&str> = out["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["FileRead"]);
        let missing: Vec<&str> = out["missing"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(missing, vec!["DoesNotExist"]);
    }

    #[tokio::test]
    async fn bare_exact_name_acts_like_select() {
        let reg = registry_with(vec![("FileRead", "read")]);
        let tool = ToolSearchTool::new(reg);
        let out = tool
            .call(&ctx(), json!({"query": "FileRead"}))
            .await
            .unwrap();
        assert_eq!(
            out["matches"].as_array().unwrap()[0].as_str().unwrap(),
            "FileRead"
        );
    }

    #[tokio::test]
    async fn bare_exact_name_is_case_insensitive() {
        // Codex round-1 design gap n-extra3: lowercase variant
        // should still hit the exact-name shortcut.
        let reg = registry_with(vec![("FileRead", "read")]);
        let tool = ToolSearchTool::new(reg);
        let out = tool
            .call(&ctx(), json!({"query": "fileread"}))
            .await
            .unwrap();
        assert_eq!(
            out["matches"].as_array().unwrap()[0].as_str().unwrap(),
            "FileRead"
        );
    }

    #[tokio::test]
    async fn keyword_scores_name_part_exact_higher_than_substring() {
        let reg = registry_with(vec![
            ("GitCommit", "commit changes"),
            ("GitCommitter", "info about a committer"),
            ("Unrelated", "something else"),
        ]);
        let tool = ToolSearchTool::new(reg);
        let out = tool.call(&ctx(), json!({"query": "commit"})).await.unwrap();
        let names: Vec<&str> = out["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        // Both Git tools match; GitCommit gets exact-part score (10),
        // GitCommitter only gets substring (5).
        assert_eq!(names[0], "GitCommit");
        assert!(names.contains(&"GitCommitter"));
    }

    #[tokio::test]
    async fn keyword_falls_back_to_description_word_boundary() {
        let reg = registry_with(vec![
            ("Hammer", "useful for nails"),
            ("Drill", "for screws"),
        ]);
        let tool = ToolSearchTool::new(reg);
        let out = tool.call(&ctx(), json!({"query": "nails"})).await.unwrap();
        let names: Vec<&str> = out["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["Hammer"]);
    }

    #[tokio::test]
    async fn keyword_respects_max_results() {
        let reg = registry_with(vec![
            ("FileRead", "read"),
            ("FileWrite", "write"),
            ("FileEdit", "edit"),
            ("FileDelete", "delete"),
        ]);
        let tool = ToolSearchTool::new(reg);
        let out = tool
            .call(&ctx(), json!({"query": "file", "max_results": 2}))
            .await
            .unwrap();
        let names = out["matches"].as_array().unwrap();
        assert_eq!(names.len(), 2);
    }

    #[tokio::test]
    async fn keyword_empty_query_returns_no_matches() {
        let reg = registry_with(vec![("FileRead", "read")]);
        let tool = ToolSearchTool::new(reg);
        let out = tool.call(&ctx(), json!({"query": "   "})).await.unwrap();
        assert_eq!(out["matches"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn mcp_tool_name_parts_score() {
        let reg = registry_with(vec![
            ("mcp__github__list_issues", "list github issues"),
            ("mcp__slack__post_message", "post a slack message"),
            ("Unrelated", "n/a"),
        ]);
        let tool = ToolSearchTool::new(reg);
        let out = tool.call(&ctx(), json!({"query": "github"})).await.unwrap();
        let names: Vec<&str> = out["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(names[0], "mcp__github__list_issues");
    }

    #[tokio::test]
    async fn safety_class_is_read_only() {
        let reg = registry_with(vec![]);
        let tool = ToolSearchTool::new(reg);
        assert_eq!(tool.safety_class(), SafetyClass::ReadOnly);
    }

    #[tokio::test]
    async fn input_schema_has_query_required() {
        let reg = registry_with(vec![]);
        let tool = ToolSearchTool::new(reg);
        let schema = tool.input_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));
    }
}
