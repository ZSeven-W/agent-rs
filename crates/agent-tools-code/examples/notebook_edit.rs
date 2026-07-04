//! Standalone example: edit a Jupyter `.ipynb` cell without
//! involving an LLM. Demonstrates the `NotebookEditTool` surface:
//! locate by `cell_id`, swap source, write back.
//!
//! Run:
//!     cargo run --example notebook_edit \
//!         --features notebook \
//!         -p agent-tools-code

use agent::abort::AbortController;
use agent::file_cache::FileStateCache;
use agent::hook::HookRunner;
use agent::permission::PermissionManager;
use agent::tool::{Tool, ToolUseContext};
use agent_tools_code::{NotebookEditTool, WorkspacePolicy};
use serde_json::json;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Plant a tiny notebook in a temp dir so the example is self-
    // contained.
    let dir = tempfile::tempdir()?;
    let nb_path = dir.path().join("demo.ipynb");
    std::fs::write(
        &nb_path,
        serde_json::to_string_pretty(&json!({
            "cells": [
                {
                    "cell_type": "code",
                    "id": "first",
                    "execution_count": null,
                    "metadata": {},
                    "outputs": [],
                    "source": ["print('hello')\n"],
                }
            ],
            "metadata": {"kernelspec": {"name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }))?,
    )?;

    let policy = WorkspacePolicy::new(dir.path())?.into_arc();
    let tool = NotebookEditTool::new(policy);

    let ctx = ToolUseContext {
        cwd: dir.path().to_path_buf(),
        abort: AbortController::new(),
        file_cache: Arc::new(FileStateCache::new(
            NonZeroUsize::new(8).unwrap(),
            1024 * 1024,
        )),
        permissions: Arc::new(PermissionManager::new()),
        hooks: Arc::new(HookRunner::new()),
        task_depth: 0,
    };

    let result = tool
        .call(
            &ctx,
            json!({
                "path": "demo.ipynb",
                "cell_id": "first",
                "mode": "replace",
                "new_source": "print('edited from agent-rs')\n",
            }),
        )
        .await?;

    println!("tool result: {result:#}");
    let on_disk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&nb_path)?)?;
    println!(
        "notebook source after edit: {}",
        on_disk["cells"][0]["source"]
    );
    Ok(())
}
