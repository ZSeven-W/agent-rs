//! WASM plugin host trait (Tier 2 / claude-code parity).
//!
//! agent-rs core does NOT pull `wasmtime` / `wasmer` into the dep
//! tree — that would be a several-MB-binary cost for every consumer
//! whether or not they actually want WASM plugins. Instead we ship
//! the trait + a no-op default; consumers (OpenPencil, Zode) wire
//! their own wasmtime-backed implementation by implementing
//! [`WasmPluginHost`] in their own crate.
//!
//! This is also why third-party plugins are sandboxed — they run in
//! the consumer's wasmtime instance with whatever capabilities the
//! consumer chose to expose. Internal plugins skip this entirely
//! and run as native [`super::Plugin`] trait objects.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use super::plugin::{Plugin, PluginError};

/// A host that knows how to load a `.wasm` file into a runtime and
/// produce a [`Plugin`] handle for it. Implementations live outside
/// agent-rs (in OP / Zode / test code).
#[async_trait]
pub trait WasmPluginHost: std::fmt::Debug + Send + Sync {
    /// Load `wasm_path` and instantiate it. The returned plugin
    /// must implement the same [`Plugin`] interface as native
    /// plugins, so the registry can treat both flavors uniformly.
    async fn load(&self, wasm_path: &Path) -> Result<Arc<dyn Plugin>, PluginError>;
}

/// Default placeholder host. Returns `PluginError::Wasm("...")` for
/// every load. Useful for tests + default builds where WASM is not
/// configured.
#[derive(Debug, Default, Clone)]
pub struct NoopWasmHost;

#[async_trait]
impl WasmPluginHost for NoopWasmHost {
    async fn load(&self, _wasm_path: &Path) -> Result<Arc<dyn Plugin>, PluginError> {
        Err(PluginError::Wasm(
            "wasm host not configured — wire a real WasmPluginHost (e.g. wasmtime-backed) into the registry"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_host_returns_wasm_error() {
        let host = NoopWasmHost;
        match host.load(Path::new("does-not-matter.wasm")).await {
            Err(PluginError::Wasm(_)) => {}
            other => panic!("expected Wasm error, got {other:?}"),
        }
    }
}
