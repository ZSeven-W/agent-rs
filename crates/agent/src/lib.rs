//! `agent` — cross-product Rust agent runtime.
//!
//! Phase 1 surface live: `error`, `message`, `stream`. Provider trait + abort
//! controller land in Phase 1 batch B; JSON helpers + file cache in batch C.
//! See `2026-04-19-agent-crate.md` plan.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod error;
pub mod message;
pub mod prelude;
pub mod stream;

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
