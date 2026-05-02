//! Compile-time smoke test: every type that's supposed to be in the
//! prelude is reachable via `agent::prelude::*`. Catches regressions
//! where a re-export is silently dropped during refactors.

#![allow(dead_code)]
#![allow(unused_imports)]

use agent::prelude::*;

fn _matcher_in_prelude() {
    let _: PermissionMatcher = PermissionMatcher::Always;
    let _: StringPattern = StringPattern::glob("rm *");
}

fn _safety_class_in_prelude() {
    let _: SafetyClass = SafetyClass::ReadOnly;
}

fn _existing_exports_still_present() {
    // Sanity check the sibling re-exports the new ones live next to.
    let _: AbortController = AbortController::new();
    let _: AgentError = AgentError::other("noop");
    let _: PermissionManager = PermissionManager::new();
}

#[test]
fn matcher_constructed_from_prelude_actually_works() {
    let m = PermissionMatcher::field_glob("/command", "rm *");
    assert!(m.matches(&serde_json::json!({"command": "rm /tmp/x"})));
    assert!(!m.matches(&serde_json::json!({"command": "ls"})));
}
