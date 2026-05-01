use async_trait::async_trait;

use super::chain::{evaluate_permission, ToolPermissionCheckFn};
use super::types::{
    AllowDecision, AskDecision, DecisionReason, DenyDecision, PermissionBehavior,
    PermissionContext, PermissionDecision, PermissionMode, PermissionRule, RuleSource,
};

/// Async equivalent of [`ToolPermissionCheckFn`] (Phase 3 batch H).
///
/// Implementations may genuinely await — e.g., bridging to an
/// [`super::ExternalQueue`] for human approval, or hitting a remote
/// policy service. Use this with [`PermissionManager::evaluate_async`].
#[async_trait]
pub trait AsyncToolPermissionCheck: Send + Sync {
    async fn check(&self, input: &serde_json::Value) -> PermissionDecision;
}

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

    /// Async variant — same 7-step chain, but the Step 1c callback is
    /// allowed to genuinely await (e.g., bridging to an
    /// [`super::ExternalQueue`] for human approval). Steps 1a, 1b, 2a,
    /// 2b, 3, 4 resolve synchronously; only Step 1c can yield.
    pub async fn evaluate_async(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        callback: Option<&dyn AsyncToolPermissionCheck>,
    ) -> PermissionDecision {
        // Step 1a — deny rule.
        if let Some(rule) = super::chain::find_rule_for_tool(
            &self.context.always_deny_rules,
            tool_name,
        ) {
            return PermissionDecision::Deny(DenyDecision {
                message_text: "Tool use denied by rule.".into(),
                reason: DecisionReason::rule(rule.clone()),
            });
        }
        // Step 1b — ask rule.
        if let Some(rule) = super::chain::find_rule_for_tool(
            &self.context.always_ask_rules,
            tool_name,
        ) {
            return PermissionDecision::Ask(AskDecision {
                message_text: "Tool use requires confirmation.".into(),
                reason: Some(DecisionReason::rule(rule.clone())),
            });
        }
        // Step 1c — async tool callback.
        if let Some(cb) = callback {
            match cb.check(input).await {
                decision @ PermissionDecision::Deny(_) => return decision,
                decision @ PermissionDecision::Ask(_) => return decision,
                PermissionDecision::Allow(_) => {
                    // Fall through — bypass / allow rule may upgrade.
                }
            }
        }
        // Step 2a — bypass mode.
        if self.context.mode == PermissionMode::Bypass {
            return PermissionDecision::Allow(AllowDecision {
                updated_input: None,
                reason: DecisionReason::mode(PermissionMode::Bypass),
            });
        }
        // Step 2b — whole-tool allow rule.
        if let Some(rule) = super::chain::find_rule_for_tool(
            &self.context.always_allow_rules,
            tool_name,
        ) {
            return PermissionDecision::Allow(AllowDecision {
                updated_input: None,
                reason: DecisionReason::rule(rule.clone()),
            });
        }
        // Step 3 + 4 — default-ask, with dont_ask escalating to deny.
        if self.context.mode == PermissionMode::DontAsk {
            return PermissionDecision::Deny(DenyDecision {
                message_text: "Permission required but prompting is disabled.".into(),
                reason: DecisionReason::mode(PermissionMode::DontAsk),
            });
        }
        PermissionDecision::Ask(AskDecision {
            message_text: "Permission required.".into(),
            reason: None,
        })
    }
}
