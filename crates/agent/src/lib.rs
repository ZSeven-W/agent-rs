//! `agent` — cross-product Rust agent runtime.
//!
//! Phase 0 placeholder. Full crate surfaces (Provider / Tool / EventStream / QueryEngine)
//! land in Phase 1+. See `2026-04-19-agent-crate.md` plan.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod prelude;

/// Crate version, sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!VERSION.is_empty());
    }
}
