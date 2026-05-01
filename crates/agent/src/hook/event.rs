//! Typed hook event vocabulary (Phase 3 / Task 3.3).
//!
//! Hooks fire on these events around the agent's core loop. Names follow
//! Claude Code's reference (see
//! `notes/2026-05-01-claude-code-feature-reference.md`) so existing
//! hook scripts that target Claude Code can be repointed at agent-rs
//! with minimal rewriting.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Tagged event payload. New variants are added with `#[non_exhaustive]`
/// so consumers must use the constructor (or pattern-match with a
/// catch-all) and won't break when a future event is introduced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HookEvent {
    BeforeToolUse {
        tool: String,
        input: serde_json::Value,
    },
    AfterToolUse {
        tool: String,
        input: serde_json::Value,
        output: serde_json::Value,
        ok: bool,
    },
    PostToolUseFailure {
        tool: String,
        input: serde_json::Value,
        error: String,
    },
    OnPermissionRequest {
        tool: String,
        input: serde_json::Value,
    },
    OnPermissionAllowed {
        tool: String,
    },
    OnPermissionDenied {
        tool: String,
        reason: String,
    },
    OnUserMessage {
        text: String,
    },
    OnAssistantMessage {
        text: String,
    },
    OnSystemMessage {
        text: String,
    },
    OnSessionStart,
    OnSessionEnd {
        exit_reason: String,
    },
    OnAbort {
        reason: String,
    },
    OnError {
        code: String,
        message: String,
    },
    OnUsage {
        input_tokens: u32,
        output_tokens: u32,
    },
    OnRetry {
        attempt: u32,
        reason: String,
    },
    OnContextWindowFull,
    OnCompact {
        reason: String,
    },
    PreCompact {
        trigger: String,
        custom_instructions: Option<String>,
    },
    PostCompact {
        pre_tokens: u32,
        post_tokens: u32,
        replaced_count: u32,
    },
    OnSubagentSpawn {
        id: String,
    },
    OnSubagentDone {
        id: String,
        ok: bool,
    },
    OnTeamMessage {
        from: String,
        to: String,
        content: String,
    },
    OnFileWrite {
        path: PathBuf,
    },
    OnFileRead {
        path: PathBuf,
    },
    OnShellExec {
        cmd: String,
        exit_code: i32,
    },
    PreSampling {
        model: String,
        input_tokens: u32,
    },
    PostSampling {
        model: String,
        output_tokens: u32,
    },
}

impl HookEvent {
    /// Stable string identifier — the same shape `event` field that
    /// serializes via serde, used by script hooks that match by name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::BeforeToolUse { .. } => "before_tool_use",
            Self::AfterToolUse { .. } => "after_tool_use",
            Self::PostToolUseFailure { .. } => "post_tool_use_failure",
            Self::OnPermissionRequest { .. } => "on_permission_request",
            Self::OnPermissionAllowed { .. } => "on_permission_allowed",
            Self::OnPermissionDenied { .. } => "on_permission_denied",
            Self::OnUserMessage { .. } => "on_user_message",
            Self::OnAssistantMessage { .. } => "on_assistant_message",
            Self::OnSystemMessage { .. } => "on_system_message",
            Self::OnSessionStart => "on_session_start",
            Self::OnSessionEnd { .. } => "on_session_end",
            Self::OnAbort { .. } => "on_abort",
            Self::OnError { .. } => "on_error",
            Self::OnUsage { .. } => "on_usage",
            Self::OnRetry { .. } => "on_retry",
            Self::OnContextWindowFull => "on_context_window_full",
            Self::OnCompact { .. } => "on_compact",
            Self::PreCompact { .. } => "pre_compact",
            Self::PostCompact { .. } => "post_compact",
            Self::OnSubagentSpawn { .. } => "on_subagent_spawn",
            Self::OnSubagentDone { .. } => "on_subagent_done",
            Self::OnTeamMessage { .. } => "on_team_message",
            Self::OnFileWrite { .. } => "on_file_write",
            Self::OnFileRead { .. } => "on_file_read",
            Self::OnShellExec { .. } => "on_shell_exec",
            Self::PreSampling { .. } => "pre_sampling",
            Self::PostSampling { .. } => "post_sampling",
        }
    }
}
