//! File-system tools: read / write / edit / list / mkdir / move /
//! remove.
//!
//! Each tool is a struct holding `Arc<WorkspacePolicy>` and
//! implementing `agent::tool::Tool`. They are deliberately small —
//! just a JSON-Schema declaration + `call` body — so a host that
//! wants slightly different semantics (e.g., a custom path
//! allowlist evaluator) can copy + tweak in their own crate.
//!
//! Conventions:
//!
//! - All input paths go through `WorkspacePolicy::resolve` before
//!   any I/O happens.
//! - All async I/O uses `tokio::fs` so the runtime stays cooperative.
//! - Read paths cap at `max_file_size_bytes`; oversized files
//!   produce an error rather than truncating.
//! - `FileRead` formats with `cat -n` style line numbers so the
//!   model can quote line ranges back to other tools.
//! - `FileEdit` requires `old_string` to match exactly once unless
//!   `replace_all` is set, mirroring Claude Code's
//!   ambiguity-refusal semantics.

use std::sync::Arc;

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::policy::{PolicyError, WorkspacePolicy};

fn policy_to_agent_err(e: PolicyError) -> AgentError {
    AgentError::other(format!("policy: {e}"))
}

fn io_to_agent_err(action: &str, path: &str, e: std::io::Error) -> AgentError {
    AgentError::other(format!("{action} '{path}' failed: {e}"))
}

// =========================================================================
// FileRead
// =========================================================================

/// Read a UTF-8 text file with optional `offset` (1-based line) and
/// `limit` (max lines). Output is `cat -n` style — one line per
/// source line, numbered from `offset`. The default offset is 1
/// and the default limit is 2000 lines (matches Claude Code).
#[derive(Debug)]
pub struct FileReadTool {
    policy: Arc<WorkspacePolicy>,
}

impl FileReadTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct FileReadInput {
    path: String,
    #[serde(default)]
    offset: Option<u64>,
    #[serde(default)]
    limit: Option<u64>,
}

const DEFAULT_READ_LINE_LIMIT: u64 = 2000;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "FileRead"
    }
    fn description(&self) -> &str {
        "Read a UTF-8 text file. Returns line-numbered content. Use offset/limit for large files."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path; relative paths resolve against the workspace cwd."},
                "offset": {"type": "integer", "minimum": 1, "description": "1-based starting line."},
                "limit": {"type": "integer", "minimum": 1, "description": "Maximum lines to return (default 2000)."},
            },
            "required": ["path"]
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
        let parsed: FileReadInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("FileRead invalid input: {e}")))?;
        let resolved = self
            .policy
            .resolve(&parsed.path, true)
            .map_err(policy_to_agent_err)?;
        let meta = tokio::fs::metadata(&resolved)
            .await
            .map_err(|e| io_to_agent_err("stat", &parsed.path, e))?;
        if !meta.is_file() {
            return Err(AgentError::other(format!(
                "FileRead '{}' is not a regular file",
                parsed.path
            )));
        }
        self.policy
            .check_size(meta.len())
            .map_err(policy_to_agent_err)?;
        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|e| io_to_agent_err("read", &parsed.path, e))?;
        let text = String::from_utf8(bytes).map_err(|e| {
            AgentError::other(format!("FileRead '{}' is not UTF-8: {e}", parsed.path))
        })?;
        let offset = parsed.offset.unwrap_or(1).max(1);
        let limit = parsed.limit.unwrap_or(DEFAULT_READ_LINE_LIMIT);
        let formatted = format_with_line_numbers(&text, offset, limit);
        Ok(json!({
            "path": resolved.display().to_string(),
            "size_bytes": meta.len(),
            "total_lines": text.lines().count(),
            "content": formatted,
        }))
    }
}

fn format_with_line_numbers(text: &str, offset: u64, limit: u64) -> String {
    let mut out = String::new();
    for (i, line) in text.lines().enumerate() {
        let lineno = (i as u64).saturating_add(1);
        if lineno < offset {
            continue;
        }
        if lineno >= offset.saturating_add(limit) {
            break;
        }
        // Right-align the line number in a 6-char column, padded
        // with spaces. Matches `cat -n` cosmetic output that
        // humans + models recognize.
        out.push_str(&format!("{lineno:>6}\t{line}\n"));
    }
    out
}

// =========================================================================
// FileWrite
// =========================================================================

/// Write `content` to `path`. Creates the file if missing,
/// overwrites otherwise. Refuses to write outside the workspace.
/// Refuses identical-content writes (no-op detect) so chatty
/// agents don't bump mtime unnecessarily.
#[derive(Debug)]
pub struct FileWriteTool {
    policy: Arc<WorkspacePolicy>,
}

impl FileWriteTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct FileWriteInput {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "FileWrite"
    }
    fn description(&self) -> &str {
        "Write text content to a file (creates or overwrites). Refuses identical-content writes."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: FileWriteInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("FileWrite invalid input: {e}")))?;
        let resolved = self
            .policy
            .resolve(&parsed.path, false)
            .map_err(policy_to_agent_err)?;
        let bytes = parsed.content.into_bytes();
        self.policy
            .check_size(bytes.len() as u64)
            .map_err(policy_to_agent_err)?;
        // Idempotency: skip the write if the existing file already
        // matches byte-for-byte. Saves churn + lets the model see
        // "no-op" feedback when it re-emits the same content.
        if let Ok(existing) = tokio::fs::read(&resolved).await {
            if existing == bytes {
                return Ok(json!({
                    "path": resolved.display().to_string(),
                    "status": "no_op_identical_content",
                    "size_bytes": bytes.len(),
                }));
            }
        }
        tokio::fs::write(&resolved, &bytes)
            .await
            .map_err(|e| io_to_agent_err("write", &parsed.path, e))?;
        Ok(json!({
            "path": resolved.display().to_string(),
            "status": "ok",
            "size_bytes": bytes.len(),
        }))
    }
}

// =========================================================================
// FileEdit
// =========================================================================

/// In-place exact-string edit. By default `old_string` must match
/// exactly once or the tool errors with `Ambiguous`; pass
/// `replace_all = true` to replace every occurrence (useful for
/// bulk rename-style edits).
///
/// Returns the number of replacements actually performed and a
/// short before/after preview.
#[derive(Debug)]
pub struct FileEditTool {
    policy: Arc<WorkspacePolicy>,
}

impl FileEditTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct FileEditInput {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &str {
        "FileEdit"
    }
    fn description(&self) -> &str {
        "Replace an exact substring in a file. Refuses ambiguous matches unless replace_all is set."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": false},
            },
            "required": ["path", "old_string", "new_string"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: FileEditInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("FileEdit invalid input: {e}")))?;
        if parsed.old_string.is_empty() {
            return Err(AgentError::other("FileEdit old_string must be non-empty"));
        }
        let resolved = self
            .policy
            .resolve(&parsed.path, true)
            .map_err(policy_to_agent_err)?;
        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|e| io_to_agent_err("read", &parsed.path, e))?;
        self.policy
            .check_size(bytes.len() as u64)
            .map_err(policy_to_agent_err)?;
        let text = String::from_utf8(bytes).map_err(|e| {
            AgentError::other(format!("FileEdit '{}' is not UTF-8: {e}", parsed.path))
        })?;
        let count = text.matches(&parsed.old_string).count();
        if count == 0 {
            return Err(AgentError::other(format!(
                "FileEdit: old_string not found in '{}'",
                parsed.path
            )));
        }
        if count > 1 && !parsed.replace_all {
            return Err(AgentError::other(format!(
                "FileEdit: old_string is ambiguous in '{}' ({count} matches). Pass replace_all=true to replace every occurrence."
            , parsed.path)));
        }
        let new_text = if parsed.replace_all {
            text.replace(&parsed.old_string, &parsed.new_string)
        } else {
            text.replacen(&parsed.old_string, &parsed.new_string, 1)
        };
        // Re-apply the policy size cap on the post-edit content. The
        // pre-read check covers the original file, but a replacement
        // can grow the file (e.g. `replace_all` with a longer
        // `new_string` doubling every match) and would otherwise
        // bypass the cap.
        self.policy
            .check_size(new_text.len() as u64)
            .map_err(policy_to_agent_err)?;
        tokio::fs::write(&resolved, new_text.as_bytes())
            .await
            .map_err(|e| io_to_agent_err("write", &parsed.path, e))?;
        Ok(json!({
            "path": resolved.display().to_string(),
            "replacements": if parsed.replace_all { count } else { 1 },
            "size_bytes": new_text.len(),
        }))
    }
}

// =========================================================================
// ListDir
// =========================================================================

#[derive(Debug)]
pub struct ListDirTool {
    policy: Arc<WorkspacePolicy>,
}

impl ListDirTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct ListDirInput {
    path: String,
}

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "ListDir"
    }
    fn description(&self) -> &str {
        "List the entries (files + subdirectories) of a directory."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
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
        let parsed: ListDirInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("ListDir invalid input: {e}")))?;
        let resolved = self
            .policy
            .resolve(&parsed.path, true)
            .map_err(policy_to_agent_err)?;
        let mut entries = tokio::fs::read_dir(&resolved)
            .await
            .map_err(|e| io_to_agent_err("read_dir", &parsed.path, e))?;
        let mut out: Vec<serde_json::Value> = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| io_to_agent_err("next_entry", &parsed.path, e))?
        {
            let kind = if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
                "dir"
            } else if entry
                .file_type()
                .await
                .map(|t| t.is_symlink())
                .unwrap_or(false)
            {
                "symlink"
            } else {
                "file"
            };
            out.push(json!({
                "name": entry.file_name().to_string_lossy(),
                "type": kind,
            }));
        }
        // Stable lex order for reproducible tool output.
        out.sort_by(|a, b| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .cmp(b["name"].as_str().unwrap_or(""))
        });
        Ok(json!({
            "path": resolved.display().to_string(),
            "entries": out,
        }))
    }
}

// =========================================================================
// Mkdir
// =========================================================================

#[derive(Debug)]
pub struct MkdirTool {
    policy: Arc<WorkspacePolicy>,
}

impl MkdirTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct MkdirInput {
    path: String,
    #[serde(default)]
    recursive: bool,
}

#[async_trait]
impl Tool for MkdirTool {
    fn name(&self) -> &str {
        "Mkdir"
    }
    fn description(&self) -> &str {
        "Create a directory. Set recursive=true to create parent directories as needed."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": false},
            },
            "required": ["path"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: MkdirInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("Mkdir invalid input: {e}")))?;
        let resolved = self
            .policy
            .resolve(&parsed.path, false)
            .map_err(policy_to_agent_err)?;
        let res = if parsed.recursive {
            tokio::fs::create_dir_all(&resolved).await
        } else {
            tokio::fs::create_dir(&resolved).await
        };
        res.map_err(|e| io_to_agent_err("mkdir", &parsed.path, e))?;
        Ok(json!({"path": resolved.display().to_string(), "status": "ok"}))
    }
}

// =========================================================================
// Move
// =========================================================================

#[derive(Debug)]
pub struct MoveTool {
    policy: Arc<WorkspacePolicy>,
}

impl MoveTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct MoveInput {
    from: String,
    to: String,
    #[serde(default)]
    overwrite: bool,
}

#[async_trait]
impl Tool for MoveTool {
    fn name(&self) -> &str {
        "Move"
    }
    fn description(&self) -> &str {
        "Move (rename) a file or directory. Refuses to overwrite an existing target unless overwrite=true."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "overwrite": {"type": "boolean", "default": false},
            },
            "required": ["from", "to"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: MoveInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("Move invalid input: {e}")))?;
        let from = self
            .policy
            .resolve(&parsed.from, true)
            .map_err(policy_to_agent_err)?;
        let to = self
            .policy
            .resolve(&parsed.to, false)
            .map_err(policy_to_agent_err)?;
        if !parsed.overwrite && tokio::fs::metadata(&to).await.is_ok() {
            return Err(AgentError::other(format!(
                "Move target '{}' already exists; pass overwrite=true to replace",
                parsed.to
            )));
        }
        tokio::fs::rename(&from, &to)
            .await
            .map_err(|e| io_to_agent_err("rename", &parsed.from, e))?;
        Ok(json!({
            "from": from.display().to_string(),
            "to": to.display().to_string(),
            "status": "ok",
        }))
    }
}

// =========================================================================
// Remove
// =========================================================================

/// Delete a file or directory. Tagged `Destructive` so default
/// permission policies require explicit confirmation. `recursive`
/// is required to delete a non-empty directory.
#[derive(Debug)]
pub struct RemoveTool {
    policy: Arc<WorkspacePolicy>,
}

impl RemoveTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

#[derive(Debug, Deserialize)]
struct RemoveInput {
    path: String,
    #[serde(default)]
    recursive: bool,
}

#[async_trait]
impl Tool for RemoveTool {
    fn name(&self) -> &str {
        "Remove"
    }
    fn description(&self) -> &str {
        "Delete a file or directory. Requires recursive=true for non-empty directories."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": false},
            },
            "required": ["path"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Destructive
    }
    async fn call(
        &self,
        _ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: RemoveInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("Remove invalid input: {e}")))?;
        let resolved = self
            .policy
            .resolve(&parsed.path, true)
            .map_err(policy_to_agent_err)?;
        let meta = tokio::fs::symlink_metadata(&resolved)
            .await
            .map_err(|e| io_to_agent_err("stat", &parsed.path, e))?;
        if meta.is_dir() {
            if parsed.recursive {
                tokio::fs::remove_dir_all(&resolved).await
            } else {
                tokio::fs::remove_dir(&resolved).await
            }
        } else {
            tokio::fs::remove_file(&resolved).await
        }
        .map_err(|e| io_to_agent_err("remove", &parsed.path, e))?;
        Ok(json!({"path": resolved.display().to_string(), "status": "ok"}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::abort::AbortController;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn ctx_for(_dir: &TempDir) -> ToolUseContext {
        // Minimal context — none of these tools use the file_cache /
        // permissions / hooks today, but we honor the trait shape.
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

    fn policy_for(dir: &TempDir) -> Arc<WorkspacePolicy> {
        WorkspacePolicy::new(dir.path()).unwrap().into_arc()
    }

    #[tokio::test]
    async fn file_read_returns_line_numbered_content() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"alpha\nbeta\ngamma\n").unwrap();
        let tool = FileReadTool::new(policy_for(&dir));
        let out = tool
            .call(&ctx_for(&dir), json!({"path": "a.txt"}))
            .await
            .unwrap();
        let content = out["content"].as_str().unwrap();
        assert!(content.contains("\talpha"));
        assert!(content.contains("\tbeta"));
        assert!(content.contains("\tgamma"));
        assert_eq!(out["total_lines"], 3);
    }

    #[tokio::test]
    async fn file_read_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let lines: Vec<String> = (1..=10).map(|i| format!("line{i}")).collect();
        std::fs::write(dir.path().join("a.txt"), lines.join("\n")).unwrap();
        let tool = FileReadTool::new(policy_for(&dir));
        let out = tool
            .call(
                &ctx_for(&dir),
                json!({"path": "a.txt", "offset": 3, "limit": 2}),
            )
            .await
            .unwrap();
        let content = out["content"].as_str().unwrap();
        assert!(content.contains("line3"));
        assert!(content.contains("line4"));
        assert!(!content.contains("line2"));
        assert!(!content.contains("line5"));
    }

    #[tokio::test]
    async fn file_read_rejects_non_utf8() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("bin.dat"), [0xff, 0xfe, 0x00, 0x00]).unwrap();
        let tool = FileReadTool::new(policy_for(&dir));
        let err = tool
            .call(&ctx_for(&dir), json!({"path": "bin.dat"}))
            .await
            .expect_err("should fail");
        assert!(err.to_string().contains("UTF-8"));
    }

    #[tokio::test]
    async fn file_read_rejects_oversized() {
        let dir = TempDir::new().unwrap();
        let big = "x".repeat(2 * 1024 * 1024);
        std::fs::write(dir.path().join("big.txt"), big).unwrap();
        // Tiny policy cap.
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_max_file_size(1024)
            .into_arc();
        let tool = FileReadTool::new(policy);
        let err = tool
            .call(&ctx_for(&dir), json!({"path": "big.txt"}))
            .await
            .expect_err("should fail");
        assert!(err.to_string().contains("too large"));
    }

    #[tokio::test]
    async fn file_write_creates_and_overwrites() {
        let dir = TempDir::new().unwrap();
        let policy = policy_for(&dir);
        let tool = FileWriteTool::new(policy);
        // Create
        let out = tool
            .call(&ctx_for(&dir), json!({"path": "x.txt", "content": "hello"}))
            .await
            .unwrap();
        assert_eq!(out["status"], "ok");
        // Overwrite
        let out = tool
            .call(&ctx_for(&dir), json!({"path": "x.txt", "content": "world"}))
            .await
            .unwrap();
        assert_eq!(out["status"], "ok");
        let read = std::fs::read_to_string(dir.path().join("x.txt")).unwrap();
        assert_eq!(read, "world");
    }

    #[tokio::test]
    async fn file_write_skips_identical_content_no_op() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"same").unwrap();
        let tool = FileWriteTool::new(policy_for(&dir));
        let out = tool
            .call(&ctx_for(&dir), json!({"path": "a.txt", "content": "same"}))
            .await
            .unwrap();
        assert_eq!(out["status"], "no_op_identical_content");
    }

    #[tokio::test]
    async fn file_edit_replaces_unique_match() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "fn old() {}\n").unwrap();
        let tool = FileEditTool::new(policy_for(&dir));
        let out = tool
            .call(
                &ctx_for(&dir),
                json!({"path": "a.txt", "old_string": "old", "new_string": "new"}),
            )
            .await
            .unwrap();
        assert_eq!(out["replacements"], 1);
        let read = std::fs::read_to_string(dir.path().join("a.txt")).unwrap();
        assert_eq!(read, "fn new() {}\n");
    }

    #[tokio::test]
    async fn file_edit_refuses_ambiguous_match_without_replace_all() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x x x").unwrap();
        let tool = FileEditTool::new(policy_for(&dir));
        let err = tool
            .call(
                &ctx_for(&dir),
                json!({"path": "a.txt", "old_string": "x", "new_string": "y"}),
            )
            .await
            .expect_err("ambiguous");
        assert!(err.to_string().contains("ambiguous"));
    }

    #[tokio::test]
    async fn file_edit_replace_all_replaces_every_match() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x x x").unwrap();
        let tool = FileEditTool::new(policy_for(&dir));
        let out = tool
            .call(
                &ctx_for(&dir),
                json!({"path": "a.txt", "old_string": "x", "new_string": "y", "replace_all": true}),
            )
            .await
            .unwrap();
        assert_eq!(out["replacements"], 3);
        let read = std::fs::read_to_string(dir.path().join("a.txt")).unwrap();
        assert_eq!(read, "y y y");
    }

    #[tokio::test]
    async fn file_edit_refuses_when_post_edit_size_exceeds_policy_cap() {
        // Pre-edit file fits under the cap, but `replace_all` with a
        // longer `new_string` would balloon it past the cap. The
        // post-edit size check must catch this before write.
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x".repeat(800)).unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_max_file_size(1024)
            .into_arc();
        let tool = FileEditTool::new(policy);
        let err = tool
            .call(
                &ctx_for(&dir),
                json!({
                    "path": "a.txt",
                    "old_string": "x",
                    "new_string": "yy", // doubles each match → 1600 bytes
                    "replace_all": true,
                }),
            )
            .await
            .expect_err("should fail");
        assert!(err.to_string().contains("too large"), "got {err}");
        // File on disk must be unchanged.
        let read = std::fs::read_to_string(dir.path().join("a.txt")).unwrap();
        assert_eq!(read.len(), 800);
    }

    #[tokio::test]
    async fn file_edit_refuses_empty_old_string() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "data").unwrap();
        let tool = FileEditTool::new(policy_for(&dir));
        let err = tool
            .call(
                &ctx_for(&dir),
                json!({"path": "a.txt", "old_string": "", "new_string": "x"}),
            )
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("non-empty"));
    }

    #[tokio::test]
    async fn list_dir_lex_sorted_with_kind() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), b"a").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"b").unwrap();
        let tool = ListDirTool::new(policy_for(&dir));
        let out = tool
            .call(&ctx_for(&dir), json!({"path": "."}))
            .await
            .unwrap();
        let entries = out["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["name"], "a.txt");
        assert_eq!(entries[1]["name"], "b.txt");
        assert_eq!(entries[2]["name"], "sub");
        assert_eq!(entries[2]["type"], "dir");
    }

    #[tokio::test]
    async fn mkdir_non_recursive_fails_when_parent_missing() {
        let dir = TempDir::new().unwrap();
        let tool = MkdirTool::new(policy_for(&dir));
        let err = tool
            .call(
                &ctx_for(&dir),
                json!({"path": "parent/child", "recursive": false}),
            )
            .await
            .expect_err("should fail");
        assert!(err.to_string().contains("mkdir"));
    }

    #[tokio::test]
    async fn mkdir_recursive_creates_intermediate() {
        let dir = TempDir::new().unwrap();
        let tool = MkdirTool::new(policy_for(&dir));
        let out = tool
            .call(&ctx_for(&dir), json!({"path": "a/b/c", "recursive": true}))
            .await
            .unwrap();
        assert_eq!(out["status"], "ok");
        assert!(dir.path().join("a/b/c").is_dir());
    }

    #[tokio::test]
    async fn move_renames_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        let tool = MoveTool::new(policy_for(&dir));
        let out = tool
            .call(&ctx_for(&dir), json!({"from": "a.txt", "to": "b.txt"}))
            .await
            .unwrap();
        assert_eq!(out["status"], "ok");
        assert!(!dir.path().join("a.txt").exists());
        assert!(dir.path().join("b.txt").exists());
    }

    #[tokio::test]
    async fn move_refuses_to_clobber_without_overwrite() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"y").unwrap();
        let tool = MoveTool::new(policy_for(&dir));
        let err = tool
            .call(&ctx_for(&dir), json!({"from": "a.txt", "to": "b.txt"}))
            .await
            .expect_err("clobber");
        assert!(err.to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn move_overwrite_replaces_target() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"new").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"old").unwrap();
        let tool = MoveTool::new(policy_for(&dir));
        tool.call(
            &ctx_for(&dir),
            json!({"from": "a.txt", "to": "b.txt", "overwrite": true}),
        )
        .await
        .unwrap();
        let read = std::fs::read_to_string(dir.path().join("b.txt")).unwrap();
        assert_eq!(read, "new");
    }

    #[tokio::test]
    async fn remove_file_works() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"x").unwrap();
        let tool = RemoveTool::new(policy_for(&dir));
        tool.call(&ctx_for(&dir), json!({"path": "a.txt"}))
            .await
            .unwrap();
        assert!(!dir.path().join("a.txt").exists());
    }

    #[tokio::test]
    async fn remove_non_empty_dir_requires_recursive() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("a.txt"), b"x").unwrap();
        let tool = RemoveTool::new(policy_for(&dir));
        // Without recursive → fail
        let err = tool
            .call(&ctx_for(&dir), json!({"path": "sub"}))
            .await
            .expect_err("non-empty");
        assert!(err.to_string().contains("remove"));
        // With recursive → success
        tool.call(&ctx_for(&dir), json!({"path": "sub", "recursive": true}))
            .await
            .unwrap();
        assert!(!sub.exists());
    }

    #[tokio::test]
    async fn remove_advertises_destructive_class() {
        let dir = TempDir::new().unwrap();
        let tool = RemoveTool::new(policy_for(&dir));
        assert_eq!(tool.safety_class(), SafetyClass::Destructive);
    }

    #[tokio::test]
    async fn read_classified_as_read_only() {
        let dir = TempDir::new().unwrap();
        let tool = FileReadTool::new(policy_for(&dir));
        assert_eq!(tool.safety_class(), SafetyClass::ReadOnly);
        assert!(!tool.is_mutating());
    }
}
