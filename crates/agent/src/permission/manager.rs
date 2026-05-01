use super::chain::{evaluate_permission, ToolPermissionCheckFn};
use super::types::{
    PermissionBehavior, PermissionContext, PermissionDecision, PermissionMode, PermissionRule,
    RuleSource,
};

/// Stateful wrapper that owns the [`PermissionContext`] and runs the
/// 7-step chain on every tool invocation.
///
/// Consumers wrap one in `Arc<PermissionManager>` and pass it around via
/// [`crate::tool::ToolUseContext`]. Build with the fluent `with_*` /
/// `allow` / `deny` / `ask` methods.
#[derive(Debug, Clone, Default)]
pub struct PermissionManager {
    context: PermissionContext,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: PermissionMode) -> Self {
        self.context.mode = mode;
        self
    }

    pub fn with_bypass_available(mut self, available: bool) -> Self {
        self.context.is_bypass_available = available;
        self
    }

    pub fn with_auto_available(mut self, available: bool) -> Self {
        self.context.is_auto_available = available;
        self
    }

    pub fn with_avoid_prompts(mut self, avoid: bool) -> Self {
        self.context.should_avoid_prompts = avoid;
        self
    }

    /// Add a whole-tool **allow** rule for `tool_name`.
    pub fn allow(mut self, source: RuleSource, tool_name: impl Into<String>) -> Self {
        self.context
            .always_allow_rules
            .push(PermissionRule::whole_tool(
                source,
                PermissionBehavior::Allow,
                tool_name,
            ));
        self
    }

    /// Add a whole-tool **deny** rule for `tool_name`.
    pub fn deny(mut self, source: RuleSource, tool_name: impl Into<String>) -> Self {
        self.context
            .always_deny_rules
            .push(PermissionRule::whole_tool(
                source,
                PermissionBehavior::Deny,
                tool_name,
            ));
        self
    }

    /// Add a whole-tool **ask** rule for `tool_name`.
    pub fn ask(mut self, source: RuleSource, tool_name: impl Into<String>) -> Self {
        self.context
            .always_ask_rules
            .push(PermissionRule::whole_tool(
                source,
                PermissionBehavior::Ask,
                tool_name,
            ));
        self
    }

    /// Add a pre-built rule (use this when `rule_content` matters or the
    /// builder shorthands above don't fit).
    pub fn add_rule(mut self, rule: PermissionRule) -> Self {
        match rule.behavior {
            PermissionBehavior::Allow => self.context.always_allow_rules.push(rule),
            PermissionBehavior::Deny => self.context.always_deny_rules.push(rule),
            PermissionBehavior::Ask => self.context.always_ask_rules.push(rule),
        }
        self
    }

    pub fn context(&self) -> &PermissionContext {
        &self.context
    }

    /// Run the 7-step chain. `callback` is the optional Step 1c tool-
    /// specific check.
    pub fn evaluate(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        callback: Option<&ToolPermissionCheckFn>,
    ) -> PermissionDecision {
        evaluate_permission(tool_name, input, &self.context, callback)
    }
}
