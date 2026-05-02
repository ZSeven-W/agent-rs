//! 7-step permission evaluation chain (Phase 3 / Task 3.1).
//!
//! Logic + types ported from Zig `agent/src/permission.zig` (Tier A in
//! the audit). Six Zig unit tests are mirrored as Rust integration tests
//! at `tests/permission_corpus.rs` (red-then-green corpus).

mod chain;
mod external_queue;
mod manager;
mod types;

pub use chain::{
    evaluate_permission, find_matching_rule, find_rule_for_tool, ToolPermissionCheckFn,
};
pub use external_queue::{
    external_queue, timeout_default_deny, ExternalQueue, ExternalQueueError, ExternalQueueReceiver,
    ExternalRequest,
};
pub use manager::{AsyncToolPermissionCheck, PermissionManager};
pub use types::{
    AllowDecision, AskDecision, DecisionReason, DenyDecision, PermissionBehavior,
    PermissionContext, PermissionDecision, PermissionMatcher, PermissionMode, PermissionRule,
    RuleSource, StringPattern,
};
