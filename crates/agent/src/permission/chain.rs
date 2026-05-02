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

/// Find the first rule for `tool_name` whose [`PermissionMatcher`]
/// accepts `input`. Whole-tool rules (`matcher == Always`) match any
/// input by definition, so they continue to apply for callers that
/// haven't started using structured matchers yet.
pub fn find_matching_rule<'a>(
    rules: &'a [PermissionRule],
    tool_name: &str,
    input: &serde_json::Value,
) -> Option<&'a PermissionRule> {
    rules.iter().find(|r| r.applies_to(tool_name, input))
}

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
    if let Some(rule) = find_matching_rule(&ctx.always_deny_rules, tool_name, input) {
        return PermissionDecision::Deny(DenyDecision {
            message_text: "Tool use denied by rule.".into(),
            reason: DecisionReason::rule(rule.clone()),
        });
    }

    // Step 1b — ask rule triggers confirmation.
    if let Some(rule) = find_matching_rule(&ctx.always_ask_rules, tool_name, input) {
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

    // Step 2b — whole-tool or matcher-bearing allow rule.
    if let Some(rule) = find_matching_rule(&ctx.always_allow_rules, tool_name, input) {
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

/// Legacy helper: find the first whole-tool rule for `tool_name`.
/// Mirrors the original Zig `findRuleForTool` semantics: the rule
/// must have BOTH `rule_content == None` AND `matcher == Always`. A
/// rule with only one of those is treated as scoped — likely a
/// pattern the host hasn't taught the matcher yet — and skipped.
///
/// New code should use [`find_matching_rule`] so structured matchers
/// are honored. This shim exists to keep deprecated callers
/// compiling; it does NOT consider input shape and will be removed
/// in a future major version.
pub fn find_rule_for_tool<'a>(
    rules: &'a [PermissionRule],
    tool_name: &str,
) -> Option<&'a PermissionRule> {
    rules.iter().find(|r| {
        r.tool_name == tool_name
            && r.rule_content.is_none()
            && matches!(r.matcher, super::types::PermissionMatcher::Always)
    })
}

#[cfg(test)]
mod chain_tests {
    use super::super::types::{PermissionBehavior, RuleSource, StringPattern};
    use super::*;

    fn ctx_with_deny(rule: PermissionRule) -> PermissionContext {
        PermissionContext {
            always_deny_rules: vec![rule],
            ..Default::default()
        }
    }

    fn ctx_with_allow(rule: PermissionRule) -> PermissionContext {
        PermissionContext {
            always_allow_rules: vec![rule],
            ..Default::default()
        }
    }

    #[test]
    fn deny_matcher_fires_only_on_matching_input() {
        let rule = PermissionRule::with_input_match(
            RuleSource::Project,
            PermissionBehavior::Deny,
            "Bash",
            "/command",
            StringPattern::glob("rm -rf *"),
        );
        let ctx = ctx_with_deny(rule);
        let danger = serde_json::json!({"command": "rm -rf /tmp"});
        let safe = serde_json::json!({"command": "ls -la"});
        assert!(evaluate_permission("Bash", &danger, &ctx, None).is_deny());
        // Safe input falls through to default-ask.
        assert!(evaluate_permission("Bash", &safe, &ctx, None).is_ask());
    }

    #[test]
    fn allow_matcher_specific_to_path_prefix() {
        let rule = PermissionRule::with_matcher(
            RuleSource::Project,
            PermissionBehavior::Allow,
            "FileEdit",
            super::super::types::PermissionMatcher::field_prefix("/file_path", "/tmp/"),
        );
        let ctx = ctx_with_allow(rule);
        let inside = serde_json::json!({"file_path": "/tmp/scratch.txt"});
        let outside = serde_json::json!({"file_path": "/etc/passwd"});
        assert!(evaluate_permission("FileEdit", &inside, &ctx, None).is_allow());
        // Outside the prefix → default-ask.
        assert!(evaluate_permission("FileEdit", &outside, &ctx, None).is_ask());
    }

    #[test]
    fn whole_tool_rule_unchanged_after_matcher_addition() {
        // Ensure pre-existing whole_tool rules still fire identically.
        let rule = PermissionRule::whole_tool(
            RuleSource::Project,
            PermissionBehavior::Deny,
            "DangerousTool",
        );
        let ctx = ctx_with_deny(rule);
        let any_input = serde_json::json!({"anything": true});
        assert!(evaluate_permission("DangerousTool", &any_input, &ctx, None).is_deny());
    }

    #[test]
    fn deny_takes_precedence_over_allow_when_both_match() {
        let deny = PermissionRule::with_input_match(
            RuleSource::Project,
            PermissionBehavior::Deny,
            "Bash",
            "/command",
            StringPattern::glob("rm *"),
        );
        let allow =
            PermissionRule::whole_tool(RuleSource::Project, PermissionBehavior::Allow, "Bash");
        let ctx = PermissionContext {
            always_deny_rules: vec![deny],
            always_allow_rules: vec![allow],
            ..Default::default()
        };
        let input = serde_json::json!({"command": "rm /tmp/x"});
        assert!(evaluate_permission("Bash", &input, &ctx, None).is_deny());
    }

    #[test]
    fn find_rule_for_tool_only_returns_whole_tool_rules() {
        // Backward-compat: legacy callers must NOT see matcher-bearing rules.
        let whole =
            PermissionRule::whole_tool(RuleSource::Project, PermissionBehavior::Deny, "Bash");
        let scoped = PermissionRule::with_input_match(
            RuleSource::Project,
            PermissionBehavior::Deny,
            "Bash",
            "/command",
            StringPattern::glob("rm *"),
        );
        let rules = vec![scoped.clone(), whole.clone()];
        let found = find_rule_for_tool(&rules, "Bash").unwrap();
        assert_eq!(found, &whole, "legacy lookup should skip the scoped rule");
    }
}
