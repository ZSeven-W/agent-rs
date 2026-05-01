//! Hook system — Phase 3 fills this with the typed event registry
//! (PreToolUse / PostToolUse / PostToolUseFailure / PermissionRequest /
//! PermissionDenied / etc.).
//!
//! Phase 2 ships only the type stub so [`crate::tool::ToolUseContext`] can
//! reference it. `HookRunner::run(...)` lands in Phase 3.

#[derive(Debug, Default)]
pub struct HookRunner {
    _phase_2_stub: (),
}

impl HookRunner {
    pub fn new() -> Self {
        Self { _phase_2_stub: () }
    }
}
