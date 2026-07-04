//! `NotebookEdit` — cell-level edits to a Jupyter `.ipynb` file.
//!
//! Modeled on Claude Code's `NotebookEdit` tool. The model picks a
//! cell by its stable `id` (or by zero-based index as a fallback)
//! and chooses one of three modes:
//!
//! - `replace` — overwrite the cell's `source` with `new_source`.
//! - `insert` — insert a new cell at the position; the existing
//!   cell at that index is pushed down.
//! - `delete` — remove the cell entirely.
//!
//! Cell type defaults to `code`; pass `cell_type: "markdown"` to
//! create / convert to a markdown cell. `raw` is also accepted.
//!
//! Why not a real notebook parser? `.ipynb` is a documented JSON
//! schema (nbformat 4). We parse and re-emit via `serde_json::Value`
//! so we never lose unknown fields (`outputs`, `metadata`, custom
//! tooling additions like `papermill`, etc.) round-tripping through
//! the edit. New cells get a synthesized id and the empty
//! `outputs` / `execution_count` shape that Jupyter expects on
//! disk; replace/delete leave existing per-cell metadata alone.
//!
//! Output capping: the file is read with the same `WorkspacePolicy`
//! size guard as `FileEdit`, so a 200 MiB notebook can't OOM the
//! agent.

use std::path::Path;
use std::sync::Arc;

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::policy::{PolicyError, WorkspacePolicy};

#[derive(Debug)]
pub struct NotebookEditTool {
    policy: Arc<WorkspacePolicy>,
}

impl NotebookEditTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

fn policy_to_agent_err(e: PolicyError) -> AgentError {
    AgentError::other(format!("policy: {e}"))
}

fn io_to_agent_err(action: &str, path: &str, e: std::io::Error) -> AgentError {
    AgentError::other(format!("{action} '{path}' failed: {e}"))
}

#[derive(Debug, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
enum EditMode {
    #[default]
    Replace,
    Insert,
    Delete,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
enum CellType {
    #[default]
    Code,
    Markdown,
    Raw,
}

#[derive(Debug, Deserialize)]
struct NotebookEditInput {
    /// Path to the `.ipynb` file (relative paths resolved against
    /// the workspace policy).
    path: String,
    /// Stable cell id from the notebook's `cells[].id` field. Either
    /// this OR `cell_index` must be provided. `cell_id` wins when
    /// both are set.
    #[serde(default)]
    cell_id: Option<String>,
    /// Zero-based index into `cells`. Used when `cell_id` isn't
    /// provided or when `mode == "insert"` and you want to position
    /// by index.
    #[serde(default)]
    cell_index: Option<usize>,
    /// `replace` (default) / `insert` / `delete`.
    #[serde(default)]
    mode: EditMode,
    /// New cell content. Required for `replace` and `insert`. The
    /// tool stores it as the JSON-array form (`["line\n", "line"]`)
    /// to match Jupyter's on-disk convention.
    #[serde(default)]
    new_source: Option<String>,
    /// Cell type for `insert` (or `replace` to convert). `code`
    /// (default) / `markdown` / `raw`.
    #[serde(default)]
    cell_type: CellType,
}

#[async_trait]
impl Tool for NotebookEditTool {
    fn name(&self) -> &str {
        "NotebookEdit"
    }
    fn description(&self) -> &str {
        "Edit a single cell in a Jupyter .ipynb file. Modes: replace (default) / insert / delete. Locate the cell by `cell_id` (preferred) or `cell_index`."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "cell_id": {"type": "string", "description": "Stable cell id from the notebook (preferred)."},
                "cell_index": {"type": "integer", "minimum": 0, "description": "Zero-based fallback when cell_id is unknown."},
                "mode": {"type": "string", "enum": ["replace", "insert", "delete"], "default": "replace"},
                "new_source": {"type": "string", "description": "Required for replace and insert."},
                "cell_type": {"type": "string", "enum": ["code", "markdown", "raw"], "default": "code"}
            },
            "required": ["path"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::Mutating
    }
    async fn call(&self, _ctx: &ToolUseContext, input: Value) -> Result<Value, AgentError> {
        let parsed: NotebookEditInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("NotebookEdit invalid input: {e}")))?;
        let resolved = self
            .policy
            .resolve(&parsed.path, true)
            .map_err(policy_to_agent_err)?;

        // Stat-then-bounded-read keeps the same TOCTOU posture as
        // FileEdit (round-7/8/9 work).
        let meta = tokio::fs::metadata(&resolved)
            .await
            .map_err(|e| io_to_agent_err("stat", &parsed.path, e))?;
        self.policy
            .check_size(meta.len())
            .map_err(policy_to_agent_err)?;
        let bytes = read_with_cap(&resolved, self.policy.max_file_size_bytes)
            .await
            .map_err(|e| io_to_agent_err("read", &parsed.path, e))?;
        self.policy
            .check_size(bytes.len() as u64)
            .map_err(policy_to_agent_err)?;

        let mut nb: Value = serde_json::from_slice(&bytes).map_err(|e| {
            AgentError::other(format!(
                "NotebookEdit '{}' is not valid JSON: {e}",
                parsed.path
            ))
        })?;

        let cells = nb
            .get_mut("cells")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                AgentError::other(format!(
                    "NotebookEdit '{}' has no `cells` array (not an .ipynb?)",
                    parsed.path
                ))
            })?;

        match parsed.mode {
            EditMode::Insert => {
                let new_source = parsed
                    .new_source
                    .ok_or_else(|| AgentError::other("NotebookEdit insert requires new_source"))?;
                let position = parsed.cell_index.unwrap_or(cells.len());
                if position > cells.len() {
                    return Err(AgentError::other(format!(
                        "NotebookEdit insert: cell_index {position} > cells len {}",
                        cells.len()
                    )));
                }
                let cell = build_cell(&parsed.cell_type, &new_source);
                let inserted_id = cell
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                cells.insert(position, cell);
                let out = serialize_notebook(&nb)?;
                self.policy
                    .write_file(&resolved, out.as_bytes())
                    .await
                    .map_err(|e| io_to_agent_err("write", &parsed.path, e))?;
                Ok(json!({
                    "path": resolved.display().to_string(),
                    "mode": "insert",
                    "cell_index": position,
                    "cell_id": inserted_id,
                }))
            }
            EditMode::Replace => {
                let new_source = parsed
                    .new_source
                    .ok_or_else(|| AgentError::other("NotebookEdit replace requires new_source"))?;
                let idx = locate_cell(cells, parsed.cell_id.as_deref(), parsed.cell_index)?;
                let existing_type = cells[idx]
                    .get("cell_type")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                cells[idx]["source"] = source_as_array(&new_source);
                if let Some(t) = cells[idx].get_mut("cell_type") {
                    *t = json!(cell_type_str(&parsed.cell_type));
                }
                // Code cells need outputs / execution_count after a
                // re-edit so Jupyter doesn't barf on load. Replace
                // resets the run-state since the new source hasn't
                // been executed.
                if matches!(parsed.cell_type, CellType::Code) {
                    cells[idx]["outputs"] = json!([]);
                    cells[idx]["execution_count"] = Value::Null;
                } else {
                    // markdown / raw cells must NOT carry outputs.
                    if let Some(obj) = cells[idx].as_object_mut() {
                        obj.remove("outputs");
                        obj.remove("execution_count");
                    }
                }
                let cell_id = cells[idx]
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                let out = serialize_notebook(&nb)?;
                self.policy
                    .write_file(&resolved, out.as_bytes())
                    .await
                    .map_err(|e| io_to_agent_err("write", &parsed.path, e))?;
                Ok(json!({
                    "path": resolved.display().to_string(),
                    "mode": "replace",
                    "cell_index": idx,
                    "cell_id": cell_id,
                    "previous_cell_type": existing_type,
                }))
            }
            EditMode::Delete => {
                let idx = locate_cell(cells, parsed.cell_id.as_deref(), parsed.cell_index)?;
                let removed = cells.remove(idx);
                let removed_id = removed
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                let out = serialize_notebook(&nb)?;
                self.policy
                    .write_file(&resolved, out.as_bytes())
                    .await
                    .map_err(|e| io_to_agent_err("write", &parsed.path, e))?;
                Ok(json!({
                    "path": resolved.display().to_string(),
                    "mode": "delete",
                    "removed_cell_index": idx,
                    "removed_cell_id": removed_id,
                }))
            }
        }
    }
}

fn cell_type_str(t: &CellType) -> &'static str {
    match t {
        CellType::Code => "code",
        CellType::Markdown => "markdown",
        CellType::Raw => "raw",
    }
}

/// Jupyter stores `source` as either a single string or an array of
/// line strings (each line keeps its trailing `\n` except possibly
/// the last). We always emit the array form because that's what
/// Jupyter's UI re-saves.
fn source_as_array(s: &str) -> Value {
    if s.is_empty() {
        return json!([]);
    }
    let mut lines: Vec<String> = Vec::new();
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        if c == '\n' {
            lines.push(s[start..=i].to_string());
            start = i + c.len_utf8();
        }
    }
    if start < s.len() {
        lines.push(s[start..].to_string());
    }
    json!(lines)
}

fn build_cell(cell_type: &CellType, source: &str) -> Value {
    let id = format!(
        "cell_{:x}",
        // 60-bit nanos-since-epoch is plenty for a per-edit unique id.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    );
    match cell_type {
        CellType::Code => json!({
            "cell_type": "code",
            "id": id,
            "execution_count": Value::Null,
            "metadata": {},
            "outputs": [],
            "source": source_as_array(source),
        }),
        CellType::Markdown => json!({
            "cell_type": "markdown",
            "id": id,
            "metadata": {},
            "source": source_as_array(source),
        }),
        CellType::Raw => json!({
            "cell_type": "raw",
            "id": id,
            "metadata": {},
            "source": source_as_array(source),
        }),
    }
}

/// Find the cell index by id (preferred) or numeric index.
fn locate_cell(
    cells: &[Value],
    cell_id: Option<&str>,
    cell_index: Option<usize>,
) -> Result<usize, AgentError> {
    if let Some(id) = cell_id {
        return cells
            .iter()
            .position(|c| c.get("id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| {
                AgentError::other(format!("NotebookEdit: cell with id '{id}' not found"))
            });
    }
    let idx = cell_index
        .ok_or_else(|| AgentError::other("NotebookEdit: provide cell_id or cell_index"))?;
    if idx >= cells.len() {
        return Err(AgentError::other(format!(
            "NotebookEdit: cell_index {idx} >= cells len {}",
            cells.len()
        )));
    }
    Ok(idx)
}

fn serialize_notebook(nb: &Value) -> Result<String, AgentError> {
    // Jupyter's on-disk format is pretty-printed with 1-space
    // indent. Match that so diffs against Jupyter-saved versions
    // stay legible.
    let mut bytes = Vec::with_capacity(8 * 1024);
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut ser = serde_json::Serializer::with_formatter(&mut bytes, formatter);
    serde::Serialize::serialize(nb, &mut ser)
        .map_err(|e| AgentError::other(format!("NotebookEdit serialize failed: {e}")))?;
    bytes.push(b'\n');
    String::from_utf8(bytes)
        .map_err(|e| AgentError::other(format!("NotebookEdit utf8 emit failed: {e}")))
}

/// Bounded async file read shared with FileEdit / FileRead. Defined
/// here as a private helper so `notebook` doesn't depend on the
/// `fs` feature being enabled.
async fn read_with_cap(path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
    use tokio::io::AsyncReadExt;
    let f = tokio::fs::File::open(path).await?;
    let limit = cap.saturating_add(1);
    let cap_usize = usize::try_from(cap).unwrap_or(usize::MAX);
    let mut buf = Vec::with_capacity(cap_usize.min(64 * 1024));
    f.take(limit).read_to_end(&mut buf).await?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use tempfile::TempDir;

    fn ctx() -> ToolUseContext {
        ToolUseContext {
            cwd: std::env::current_dir().unwrap(),
            abort: agent::abort::AbortController::new(),
            file_cache: Arc::new(agent::file_cache::FileStateCache::new(
                NonZeroUsize::new(8).unwrap(),
                1024 * 1024,
            )),
            permissions: Arc::new(agent::permission::PermissionManager::new()),
            hooks: Arc::new(agent::hook::HookRunner::new()),
            task_depth: 0,
        }
    }

    fn policy_for(dir: &TempDir) -> Arc<WorkspacePolicy> {
        WorkspacePolicy::new(dir.path()).unwrap().into_arc()
    }

    fn fixture_notebook() -> Value {
        json!({
            "cells": [
                {
                    "cell_type": "code",
                    "id": "abc123",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": ["print('hi')\n"],
                },
                {
                    "cell_type": "markdown",
                    "id": "md1",
                    "metadata": {},
                    "source": ["# heading\n"],
                },
            ],
            "metadata": {"kernelspec": {"name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        })
    }

    fn write_fixture(dir: &TempDir, name: &str) {
        let nb = fixture_notebook();
        let s = serde_json::to_string_pretty(&nb).unwrap();
        std::fs::write(dir.path().join(name), s).unwrap();
    }

    #[tokio::test]
    async fn replace_by_cell_id_updates_source_and_resets_run_state() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir, "n.ipynb");
        let tool = NotebookEditTool::new(policy_for(&dir));
        let out = tool
            .call(
                &ctx(),
                json!({
                    "path": "n.ipynb",
                    "cell_id": "abc123",
                    "mode": "replace",
                    "new_source": "print('updated')\n",
                    "cell_type": "code",
                }),
            )
            .await
            .unwrap();
        assert_eq!(out["mode"], "replace");
        assert_eq!(out["cell_index"], 0);
        // Re-read and verify on-disk shape.
        let s = std::fs::read_to_string(dir.path().join("n.ipynb")).unwrap();
        let parsed: Value = serde_json::from_str(&s).unwrap();
        let cell = &parsed["cells"][0];
        assert_eq!(cell["source"], json!(["print('updated')\n"]));
        // Run state reset.
        assert_eq!(cell["outputs"], json!([]));
        assert!(cell["execution_count"].is_null());
    }

    #[tokio::test]
    async fn replace_by_index_works_when_id_missing() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir, "n.ipynb");
        let tool = NotebookEditTool::new(policy_for(&dir));
        tool.call(
            &ctx(),
            json!({
                "path": "n.ipynb",
                "cell_index": 1,
                "mode": "replace",
                "new_source": "## new heading",
                "cell_type": "markdown",
            }),
        )
        .await
        .unwrap();
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("n.ipynb")).unwrap())
                .unwrap();
        // Markdown cells must not have outputs / execution_count.
        let cell = &parsed["cells"][1];
        assert_eq!(cell["cell_type"], "markdown");
        assert!(cell.get("outputs").is_none());
        assert!(cell.get("execution_count").is_none());
        assert_eq!(cell["source"], json!(["## new heading"]));
    }

    #[tokio::test]
    async fn insert_at_index_pushes_existing_cells_down() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir, "n.ipynb");
        let tool = NotebookEditTool::new(policy_for(&dir));
        tool.call(
            &ctx(),
            json!({
                "path": "n.ipynb",
                "cell_index": 1,
                "mode": "insert",
                "new_source": "# inserted",
                "cell_type": "markdown",
            }),
        )
        .await
        .unwrap();
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("n.ipynb")).unwrap())
                .unwrap();
        let cells = parsed["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 3);
        assert_eq!(cells[0]["id"], "abc123");
        assert_eq!(cells[1]["cell_type"], "markdown");
        assert_eq!(cells[1]["source"], json!(["# inserted"]));
        assert_eq!(cells[2]["id"], "md1");
    }

    #[tokio::test]
    async fn insert_without_index_appends() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir, "n.ipynb");
        let tool = NotebookEditTool::new(policy_for(&dir));
        tool.call(
            &ctx(),
            json!({
                "path": "n.ipynb",
                "mode": "insert",
                "new_source": "print('end')",
            }),
        )
        .await
        .unwrap();
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("n.ipynb")).unwrap())
                .unwrap();
        let cells = parsed["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 3);
        assert_eq!(cells[2]["source"], json!(["print('end')"]));
        assert_eq!(cells[2]["cell_type"], "code");
        assert_eq!(cells[2]["outputs"], json!([]));
    }

    #[tokio::test]
    async fn delete_by_id_removes_cell() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir, "n.ipynb");
        let tool = NotebookEditTool::new(policy_for(&dir));
        tool.call(
            &ctx(),
            json!({"path": "n.ipynb", "cell_id": "md1", "mode": "delete"}),
        )
        .await
        .unwrap();
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("n.ipynb")).unwrap())
                .unwrap();
        let cells = parsed["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0]["id"], "abc123");
    }

    #[tokio::test]
    async fn replace_missing_cell_id_errors() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir, "n.ipynb");
        let tool = NotebookEditTool::new(policy_for(&dir));
        let err = tool
            .call(
                &ctx(),
                json!({"path": "n.ipynb", "cell_id": "nope", "mode": "replace", "new_source": "x"}),
            )
            .await
            .expect_err("not found");
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn replace_without_new_source_errors() {
        let dir = TempDir::new().unwrap();
        write_fixture(&dir, "n.ipynb");
        let tool = NotebookEditTool::new(policy_for(&dir));
        let err = tool
            .call(
                &ctx(),
                json!({"path": "n.ipynb", "cell_id": "abc123", "mode": "replace"}),
            )
            .await
            .expect_err("missing source");
        assert!(err.to_string().contains("requires new_source"));
    }

    #[tokio::test]
    async fn rejects_non_notebook_json() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("not.ipynb"), r#"{"foo": 1}"#).unwrap();
        let tool = NotebookEditTool::new(policy_for(&dir));
        let err = tool
            .call(
                &ctx(),
                json!({"path": "not.ipynb", "cell_index": 0, "mode": "replace", "new_source": "x"}),
            )
            .await
            .expect_err("no cells");
        assert!(err.to_string().contains("`cells`"));
    }

    #[tokio::test]
    async fn source_as_array_preserves_internal_newlines() {
        let v = source_as_array("a\nb\nc");
        assert_eq!(v, json!(["a\n", "b\n", "c"]));
        // Empty string maps to empty array.
        assert_eq!(source_as_array(""), json!([]));
        // Single line no trailing newline.
        assert_eq!(source_as_array("hi"), json!(["hi"]));
        // Trailing newline preserved on last element.
        assert_eq!(source_as_array("hi\n"), json!(["hi\n"]));
    }

    #[tokio::test]
    async fn classified_mutating() {
        let dir = TempDir::new().unwrap();
        let tool = NotebookEditTool::new(policy_for(&dir));
        assert_eq!(tool.safety_class(), SafetyClass::Mutating);
    }

    #[tokio::test]
    async fn round_trip_preserves_unknown_fields() {
        // Plant a custom metadata field that no parser knows about
        // and confirm an edit doesn't strip it.
        let dir = TempDir::new().unwrap();
        let mut nb = fixture_notebook();
        nb["metadata"]["custom_tooling"] = json!({"papermill": {"parameters": {"a": 1}}});
        std::fs::write(
            dir.path().join("n.ipynb"),
            serde_json::to_string_pretty(&nb).unwrap(),
        )
        .unwrap();
        let tool = NotebookEditTool::new(policy_for(&dir));
        tool.call(
            &ctx(),
            json!({
                "path": "n.ipynb",
                "cell_id": "abc123",
                "mode": "replace",
                "new_source": "print('x')",
            }),
        )
        .await
        .unwrap();
        let parsed: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("n.ipynb")).unwrap())
                .unwrap();
        assert_eq!(
            parsed["metadata"]["custom_tooling"]["papermill"]["parameters"]["a"],
            1
        );
    }
}
