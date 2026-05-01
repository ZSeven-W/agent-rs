//! `agent` — cross-product Rust agent runtime.
//!
//! Phase 1 surface live: `error`, `message`, `stream`, `abort`, `provider`.
//! JSON helpers + file cache land in batch C. See `2026-04-19-agent-crate.md`
//! plan.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod abort;
pub mod error;
pub mod file_cache;
pub mod hook;
pub mod json;
pub mod message;
pub mod permission;
pub mod prelude;
pub mod provider;
pub mod stream;
pub mod testing;
pub mod tool;

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
