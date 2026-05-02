//! Permission 7-step chain corpus.
//!
//! Six tests ported verbatim from Zig `agent/src/permission.zig`:
//!
//! - Step 1a: deny rule blocks tool
//! - Step 2a: bypass mode allows everything
//! - Step 2b: allow rule allows tool
//! - Step 3:  default falls through to ask
//! - Step 4:  dont_ask converts ask to deny
//! - precedence: deny rule takes priority over allow rule
//!
//! Plus additional cases covering Step 1b (explicit ask rule survives
//! dont_ask) and Step 1c (callback short-circuits / falls through).

use agent::permission::{
    evaluate_permission, AskDecision, DecisionReason, PermissionBehavior, PermissionContext,
    PermissionDecision, PermissionMode, PermissionRule, RuleSource,
};

fn null_input() -> serde_json::Value {
    serde_json::Value::Null
}

fn whole_tool(source: RuleSource, behavior: PermissionBehavior, tool_name: &str) -> PermissionRule {
    PermissionRule::whole_tool(source, behavior, tool_name)
}

fn make_ctx(
    mode: PermissionMode,
    allow: Vec<PermissionRule>,
    deny: Vec<PermissionRule>,
    ask: Vec<PermissionRule>,
) -> PermissionContext {
    PermissionContext {
        mode,
        always_allow_rules: allow,
        always_deny_rules: deny,
        always_ask_rules: ask,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Zig-corpus port (verbatim test names)
// ---------------------------------------------------------------------------

#[test]
fn step_1a_deny_rule_blocks_tool() {
    let deny_rule = whole_tool(RuleSource::User, PermissionBehavior::Deny, "Bash");
    let ctx = make_ctx(PermissionMode::Default, vec![], vec![deny_rule], vec![]);
    let decision = evaluate_permission("Bash", &null_input(), &ctx, None);
    assert!(decision.is_deny(), "expected deny, got {decision:?}");
}

#[test]
fn step_2a_bypass_mode_allows_everything() {
    let ctx = make_ctx(PermissionMode::Bypass, vec![], vec![], vec![]);
    let decision = evaluate_permission("Bash", &null_input(), &ctx, None);
    assert!(decision.is_allow(), "expected allow, got {decision:?}");
}

#[test]
fn step_2b_allow_rule_allows_tool() {
    let allow_rule = whole_tool(RuleSource::User, PermissionBehavior::Allow, "Read");
    let ctx = make_ctx(PermissionMode::Default, vec![allow_rule], vec![], vec![]);
    let decision = evaluate_permission("Read", &null_input(), &ctx, None);
    assert!(decision.is_allow(), "expected allow, got {decision:?}");
}

#[test]
fn step_3_default_falls_through_to_ask() {
    let ctx = make_ctx(PermissionMode::Default, vec![], vec![], vec![]);
    let decision = evaluate_permission("Write", &null_input(), &ctx, None);
    assert!(decision.is_ask(), "expected ask, got {decision:?}");
}

#[test]
fn step_4_dont_ask_converts_ask_to_deny() {
    let ctx = make_ctx(PermissionMode::DontAsk, vec![], vec![], vec![]);
    let decision = evaluate_permission("Write", &null_input(), &ctx, None);
    assert!(decision.is_deny(), "expected deny, got {decision:?}");
}

#[test]
fn deny_rule_takes_priority_over_allow_rule() {
    let deny_rule = whole_tool(RuleSource::Policy, PermissionBehavior::Deny, "Bash");
    let allow_rule = whole_tool(RuleSource::User, PermissionBehavior::Allow, "Bash");
    let ctx = make_ctx(
        PermissionMode::Default,
        vec![allow_rule],
        vec![deny_rule],
        vec![],
    );
    let decision = evaluate_permission("Bash", &null_input(), &ctx, None);
    assert!(decision.is_deny(), "expected deny, got {decision:?}");
}

// ---------------------------------------------------------------------------
// Beyond-corpus cases (still rooted in the chain spec from the Zig file)
// ---------------------------------------------------------------------------

#[test]
fn step_1b_ask_rule_triggers_ask_with_rule_reason() {
    let ask_rule = whole_tool(RuleSource::Project, PermissionBehavior::Ask, "Bash");
    let ctx = make_ctx(
        PermissionMode::Default,
        vec![],
        vec![],
        vec![ask_rule.clone()],
    );
    let decision = evaluate_permission("Bash", &null_input(), &ctx, None);
    match decision {
        PermissionDecision::Ask(AskDecision { reason, .. }) => match reason {
            Some(DecisionReason::Rule { rule: r }) => assert_eq!(r, ask_rule),
            other => panic!("expected Rule reason, got {other:?}"),
        },
        other => panic!("expected ask, got {other:?}"),
    }
}

#[test]
fn step_1b_ask_rule_survives_dont_ask_mode() {
    // A user-installed ask rule says "always ask for Bash". Even in
    // dont_ask mode the explicit ask rule wins (Step 1b runs before the
    // Step 4 conversion). The host UI is responsible for surfacing
    // "operator must approve" rather than crashing.
    let ask_rule = whole_tool(RuleSource::User, PermissionBehavior::Ask, "Bash");
    let ctx = make_ctx(PermissionMode::DontAsk, vec![], vec![], vec![ask_rule]);
    let decision = evaluate_permission("Bash", &null_input(), &ctx, None);
    assert!(
        decision.is_ask(),
        "explicit ask rule must survive dont_ask, got {decision:?}"
    );
}

#[test]
fn step_1c_callback_deny_short_circuits() {
    let allow_rule = whole_tool(RuleSource::User, PermissionBehavior::Allow, "Bash");
    let ctx = make_ctx(PermissionMode::Default, vec![allow_rule], vec![], vec![]);
    let cb: Box<dyn Fn(&serde_json::Value) -> PermissionDecision + Send + Sync> =
        Box::new(|_input| {
            PermissionDecision::Deny(agent::permission::DenyDecision {
                message_text: "callback says no".into(),
                reason: DecisionReason::other("custom"),
            })
        });
    let decision = evaluate_permission("Bash", &null_input(), &ctx, Some(&*cb));
    assert!(
        decision.is_deny(),
        "callback deny must override allow rule, got {decision:?}"
    );
}

#[test]
fn step_1c_callback_allow_falls_through_to_allow_rule() {
    let allow_rule = whole_tool(RuleSource::User, PermissionBehavior::Allow, "Bash");
    let ctx = make_ctx(PermissionMode::Default, vec![allow_rule], vec![], vec![]);
    let cb: Box<dyn Fn(&serde_json::Value) -> PermissionDecision + Send + Sync> =
        Box::new(|_input| {
            PermissionDecision::Allow(agent::permission::AllowDecision {
                updated_input: None,
                reason: DecisionReason::other("callback ok"),
            })
        });
    let decision = evaluate_permission("Bash", &null_input(), &ctx, Some(&*cb));
    // Allow rule reason wins (callback Allow falls through to Step 2b).
    match decision {
        PermissionDecision::Allow(allow) => match allow.reason {
            DecisionReason::Rule { rule: r } => assert_eq!(r.tool_name, "Bash"),
            other => panic!("expected allow rule reason, got {other:?}"),
        },
        other => panic!("expected allow, got {other:?}"),
    }
}

#[test]
fn step_1c_callback_allow_falls_through_to_default_ask() {
    let ctx = make_ctx(PermissionMode::Default, vec![], vec![], vec![]);
    let cb: Box<dyn Fn(&serde_json::Value) -> PermissionDecision + Send + Sync> =
        Box::new(|_input| {
            PermissionDecision::Allow(agent::permission::AllowDecision {
                updated_input: None,
                reason: DecisionReason::other("ok"),
            })
        });
    let decision = evaluate_permission("Bash", &null_input(), &ctx, Some(&*cb));
    // Without a Step 2a (bypass) or 2b (allow rule), the callback's allow
    // is "permissive but not authoritative" — fall through to Step 3 ask.
    assert!(
        decision.is_ask(),
        "expected default ask after permissive callback, got {decision:?}"
    );
}

#[test]
fn input_pattern_rules_are_ignored_in_phase_3_batch_g() {
    // Batch G mirrors the Zig `findRuleForTool`, which only matches
    // whole-tool rules (rule_content == None). A rule with a pattern is
    // not honored and falls through to default ask.
    let pattern_rule = PermissionRule {
        source: RuleSource::User,
        behavior: PermissionBehavior::Deny,
        tool_name: "Bash".into(),
        rule_content: Some("rm *".into()),
    };
    let ctx = make_ctx(PermissionMode::Default, vec![], vec![pattern_rule], vec![]);
    let decision = evaluate_permission("Bash", &serde_json::json!({"cmd": "rm /"}), &ctx, None);
    assert!(
        decision.is_ask(),
        "pattern rules are ignored in batch G; default ask, got {decision:?}"
    );
}

#[test]
fn manager_builder_round_trip() {
    use agent::permission::PermissionManager;
    let mgr = PermissionManager::new()
        .with_mode(PermissionMode::Default)
        .deny(RuleSource::Policy, "Bash")
        .allow(RuleSource::User, "Read")
        .ask(RuleSource::Project, "Write");
    assert!(mgr.evaluate("Bash", &null_input(), None).is_deny());
    assert!(mgr.evaluate("Read", &null_input(), None).is_allow());
    assert!(mgr.evaluate("Write", &null_input(), None).is_ask());
    assert!(mgr.evaluate("Unlisted", &null_input(), None).is_ask());
}
