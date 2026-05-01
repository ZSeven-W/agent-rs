//! Permission decision types — shape ported verbatim from the Zig
//! `agent/src/permission.zig` (Tier A in the skeleton audit).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    #[default]
    Default,
    AcceptEdits,
    Bypass,
    Plan,
    DontAsk,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionBehavior {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleSource {
    Policy,
    User,
    Project,
    Local,
    Flag,
    CliArg,
    Command,
    Session,
}

/// A permission rule applies to a specific tool. `rule_content == None`
/// means the rule covers the **whole tool** regardless of input. A future
/// extension (Phase 3+ follow-up) will interpret `Some(pattern)` as an
/// input-shape matcher (exact / glob / regex); Phase 3 batch G keeps the
/// Zig-corpus semantics where only whole-tool rules are honored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRule {
    pub source: RuleSource,
    pub behavior: PermissionBehavior,
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_content: Option<String>,
}

impl PermissionRule {
    /// Convenience: whole-tool rule (the only kind interpreted in batch G).
    pub fn whole_tool(
        source: RuleSource,
        behavior: PermissionBehavior,
        tool_name: impl Into<String>,
    ) -> Self {
        Self {
            source,
            behavior,
            tool_name: tool_name.into(),
            rule_content: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionReason {
    Rule(PermissionRule),
    Mode(PermissionMode),
    SafetyCheck {
        reason: String,
        classifier_approvable: bool,
    },
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllowDecision {
    /// Optionally an updated/sanitised input to pass on to the tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_input: Option<serde_json::Value>,
    pub reason: DecisionReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskDecision {
    pub message_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<DecisionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenyDecision {
    pub message_text: String,
    pub reason: DecisionReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow(AllowDecision),
    Ask(AskDecision),
    Deny(DenyDecision),
}

impl PermissionDecision {
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow(_))
    }
    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask(_))
    }
    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny(_))
    }
}

/// Bundle of rules + mode flags that drive the 7-step chain. Build via
/// `PermissionContext::default()` + field assignments, or via the
/// [`crate::permission::PermissionManager`] builder.
#[derive(Debug, Clone, Default)]
pub struct PermissionContext {
    pub mode: PermissionMode,
    pub always_allow_rules: Vec<PermissionRule>,
    pub always_deny_rules: Vec<PermissionRule>,
    pub always_ask_rules: Vec<PermissionRule>,
    /// Set by the host to indicate the user has bypass available
    /// (some UIs hide the Bypass mode if the user lacks the role).
    pub is_bypass_available: bool,
    /// Set by the host to indicate auto-approve is available.
    pub is_auto_available: bool,
    /// Hint that the host is in a mode where prompting is undesirable
    /// (e.g., headless CI). Currently informational; see also `DontAsk` mode.
    pub should_avoid_prompts: bool,
}
