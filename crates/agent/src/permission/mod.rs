//! Permission system — Phase 3 fills this with the 7-step evaluation chain
//! ported from the Zig corpus (see `notes/2026-05-01-zig-skeleton-audit.md`,
//! Tier A).
//!
//! Phase 2 ships only the type stub so [`crate::tool::ToolUseContext`] can
//! reference it. `PermissionManager::evaluate(...)` lands in Phase 3.

#[derive(Debug, Default)]
pub struct PermissionManager {
    _phase_2_stub: (),
}

impl PermissionManager {
    pub fn new() -> Self {
        Self { _phase_2_stub: () }
    }
}
