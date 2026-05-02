//! 7-step permission evaluation chain.
//!
//! Logic ports from Zig `agent/src/permission.zig::evaluatePermission`.
//! Step ordering (matches Claude Code reference):
//!
//! 1. **Step 1a — deny rule** → DENY immediately.
//! 2. **Step 1b — ask rule** → ASK.
//! 3. **Step 1c — tool-specific callback** — if it returns deny/ask, use
//!    that; if it returns allow, fall through (bypass + allow rule still
//!    apply, callback is permissive).
//! 4. **Step 2a — bypass mode** → ALLOW (mode-driven catch-all).
//! 5. **Step 2b — whole-tool allow rule** → ALLOW.
//! 6. **Step 3 — default** → ASK.
//! 7. **Step 4 — dont_ask mode** converts the default-ask path to DENY
//!    (note: only the *default* ask, not the explicit Step 1b ask rule —
//!    those still produce ASK so the host can render an "operator must
//!    intervene" notice).

use super::types::{
    AllowDecision, AskDecision, DecisionReason, DenyDecision, PermissionContext,
    PermissionDecision, PermissionMode, PermissionRule,
};

/// Optional per-tool **synchronous** callback. Receives the input JSON
/// and returns its own decision. If it returns `Allow`, the chain
/// continues (bypass + allow rule may still upgrade to ALLOW); if it
/// returns `Deny` or `Ask`, the chain returns immediately.
///
/// This is the in-process / no-wait callback shape — useful for cheap
/// checks like "is this path inside cwd?". Phase 3 batch H adds an
/// async variant and a separate `evaluate_async` entry point that can
/// integrate the external approval queue (UI prompt, Slack approval,
/// etc.) without blocking a runtime thread. **Do not rely on this
/// alias name as a stable surface** — it may be repackaged once the
/// async path lands; prefer constructing the boxed Fn inline at call
/// sites.
pub type ToolPermissionCheckFn = dyn Fn(&serde_json::Value) -> PermissionDecision + Send + Sync;

pub fn evaluate_permission(
    tool_name: &str,
    input: &serde_json::Value,
    ctx: &PermissionContext,
    tool_check: Option<&ToolPermissionCheckFn>,
) -> PermissionDecision {
    // Step 1a — deny rule blocks immediately.
    if let Some(rule) = find_rule_for_tool(&ctx.always_deny_rules, tool_name) {
        return PermissionDecision::Deny(DenyDecision {
            message_text: "Tool use denied by rule.".into(),
            reason: DecisionReason::rule(rule.clone()),
        });
    }

    // Step 1b — ask rule triggers confirmation.
    if let Some(rule) = find_rule_for_tool(&ctx.always_ask_rules, tool_name) {
        return PermissionDecision::Ask(AskDecision {
            message_text: "Tool use requires confirmation.".into(),
            reason: Some(DecisionReason::rule(rule.clone())),
        });
    }

    // Step 1c — tool-specific callback.
    if let Some(check_fn) = tool_check {
        match check_fn(input) {
            decision @ PermissionDecision::Deny(_) => return decision,
            decision @ PermissionDecision::Ask(_) => return decision,
            PermissionDecision::Allow(_) => {
                // Fall through — bypass / allow-rule checks still apply.
            }
        }
    }

    // Step 2a — bypass mode allows everything.
    if ctx.mode == PermissionMode::Bypass {
        return PermissionDecision::Allow(AllowDecision {
            updated_input: None,
            reason: DecisionReason::mode(PermissionMode::Bypass),
        });
    }

    // Step 2b — whole-tool allow rule.
    if let Some(rule) = find_rule_for_tool(&ctx.always_allow_rules, tool_name) {
        return PermissionDecision::Allow(AllowDecision {
            updated_input: None,
            reason: DecisionReason::rule(rule.clone()),
        });
    }

    // Step 3 + 4 — default-ask, with dont_ask escalating to deny.
    if ctx.mode == PermissionMode::DontAsk {
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

/// Find the first whole-tool rule (`rule_content == None`) for `tool_name`.
pub fn find_rule_for_tool<'a>(
    rules: &'a [PermissionRule],
    tool_name: &str,
) -> Option<&'a PermissionRule> {
    rules
        .iter()
        .find(|r| r.rule_content.is_none() && r.tool_name == tool_name)
}
