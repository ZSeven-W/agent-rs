//! Plugins (Tier 2 / claude-code parity).
//!
//! Mirrors `services/plugins/`. The agent supports two flavors of
//! plugin per the project decision (memory entry
//! `project_plugins_wasm_third_party.md`, 2026-05-02):
//!
//! - **Built-in / first-party** plugins are native Rust trait
//!   objects — full async API, direct dispatch, no sandboxing
//!   overhead. Implement [`Plugin`] in your own crate and register
//!   via [`PluginRegistry::install`].
//!
//! - **Third-party** plugins run in a WASM sandbox (WASI-compatible)
//!   to isolate untrusted code. The runtime contract is captured by
//!   [`WasmPluginHost`] — a host-supplied loader that consumes a
//!   `.wasm` file and returns a [`Plugin`] handle. The default build
//!   ships an in-process [`NoopWasmHost`] that always errors with
//!   "wasm host not configured"; consumers (OP, Zode) wire a real
//!   `wasmtime` / `wasmer` host into it. Keeping the WASM dep tree
//!   out of agent-rs core lets us stay compile-time-friendly.
//!
//! Plugin capabilities (all optional):
//! - Tools — additional [`crate::tool::Tool`] entries injected into
//!   the agent's [`crate::tool::ToolRegistry`].
//! - Hooks — additional [`crate::hook::HookHandler`] entries
//!   injected into the [`crate::hook::HookRunner`].
//! - Skills — additional [`crate::skills::Skill`] entries injected
//!   into a [`crate::skills::SkillRegistry`].

pub mod manifest;
pub mod plugin;
pub mod registry;
pub mod wasm_host;

pub use manifest::{PluginManifest, PluginManifestError};
pub use plugin::{
    InstallContext, NativePlugin, Plugin, PluginCapabilities, PluginError, PluginKind,
};
pub use registry::PluginRegistry;
pub use wasm_host::{NoopWasmHost, WasmPluginHost};
