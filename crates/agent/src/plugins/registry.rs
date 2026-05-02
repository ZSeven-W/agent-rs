//! Plugin registry — manages installed plugins (native + WASM).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::plugin::{InstallContext, Plugin, PluginError};
use super::wasm_host::WasmPluginHost;

/// Shared registry of installed plugins. Cheap to clone (Arc-wrapped).
#[derive(Debug, Clone)]
pub struct PluginRegistry {
    inner: Arc<Mutex<BTreeMap<String, Arc<dyn Plugin>>>>,
    wasm_host: Arc<dyn WasmPluginHost>,
}

impl PluginRegistry {
    /// Construct a registry. `wasm_host` is consulted only when a
    /// host calls [`Self::load_wasm`]; default is
    /// [`super::NoopWasmHost`] which errors on every load.
    pub fn new(wasm_host: Arc<dyn WasmPluginHost>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BTreeMap::new())),
            wasm_host,
        }
    }

    /// Install a plugin (native or pre-built WASM). Calls the
    /// plugin's `install()` so it can push tools/hooks/skills into
    /// the supplied context. Errors:
    ///
    /// - [`PluginError::Install("duplicate name")`] if a plugin
    ///   with the same `name()` is already installed.
    /// - Whatever the plugin returns from its own `install()`.
    pub async fn install(
        &self,
        plugin: Arc<dyn Plugin>,
        ctx: &InstallContext,
    ) -> Result<(), PluginError> {
        {
            let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if g.contains_key(plugin.name()) {
                return Err(PluginError::Install(format!(
                    "duplicate plugin name `{}`",
                    plugin.name()
                )));
            }
        }
        plugin.install(ctx).await?;
        let name = plugin.name().to_string();
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.insert(name, plugin);
        Ok(())
    }

    /// Load a WASM plugin from a file via the registered host, then
    /// install it. Convenience wrapper around `wasm_host.load` +
    /// `install`.
    pub async fn load_wasm(
        &self,
        wasm_path: &std::path::Path,
        ctx: &InstallContext,
    ) -> Result<(), PluginError> {
        let plugin = self.wasm_host.load(wasm_path).await?;
        self.install(plugin, ctx).await
    }

    /// Uninstall a plugin by name. Calls the plugin's `uninstall()`
    /// before removing the registry entry. The plugin entry is
    /// dropped even if `uninstall()` fails — the host can re-install
    /// later if needed. Returns `true` if the plugin existed.
    pub async fn uninstall(&self, name: &str, ctx: &InstallContext) -> Result<bool, PluginError> {
        let plugin = {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.remove(name)
        };
        let plugin = match plugin {
            Some(p) => p,
            None => return Ok(false),
        };
        plugin.uninstall(ctx).await?;
        Ok(true)
    }

    pub fn list_names(&self) -> Vec<String> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.keys().cloned().collect()
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Plugin>> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.get(name).cloned()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::super::plugin::{NativePlugin, PluginCapabilities};
    use super::super::wasm_host::NoopWasmHost;
    use super::*;
    use crate::hook::HookRunner;
    use crate::skills::SkillRegistry;
    use crate::tool::ToolRegistry;

    fn ctx() -> InstallContext {
        InstallContext {
            tools: Arc::new(ToolRegistry::new()),
            hooks: Arc::new(HookRunner::new()),
            skills: SkillRegistry::new(),
        }
    }

    fn registry() -> PluginRegistry {
        PluginRegistry::new(Arc::new(NoopWasmHost))
    }

    fn noop_plugin(name: &str) -> Arc<dyn Plugin> {
        Arc::new(NativePlugin::new(
            name,
            "test",
            "0.1.0",
            PluginCapabilities::default(),
            |_| Ok(()),
            |_| Ok(()),
        ))
    }

    #[tokio::test]
    async fn install_then_uninstall_round_trip() {
        let reg = registry();
        let ctx = ctx();
        reg.install(noop_plugin("a"), &ctx).await.unwrap();
        assert_eq!(reg.list_names(), vec!["a".to_string()]);
        assert!(reg.uninstall("a", &ctx).await.unwrap());
        assert!(reg.is_empty());
    }

    #[tokio::test]
    async fn duplicate_install_errors() {
        let reg = registry();
        let ctx = ctx();
        reg.install(noop_plugin("a"), &ctx).await.unwrap();
        let err = reg.install(noop_plugin("a"), &ctx).await.unwrap_err();
        assert!(matches!(err, PluginError::Install(ref m) if m.contains("duplicate")));
    }

    #[tokio::test]
    async fn install_propagates_plugin_error() {
        let reg = registry();
        let ctx = ctx();
        let p = Arc::new(NativePlugin::new(
            "bad",
            "x",
            "0.1.0",
            PluginCapabilities::default(),
            |_| Err(PluginError::Install("plugin says no".into())),
            |_| Ok(()),
        ));
        let err = reg.install(p, &ctx).await.unwrap_err();
        assert!(matches!(err, PluginError::Install(ref m) if m.contains("plugin says no")));
        // Failed install must NOT leave a registry entry.
        assert!(reg.is_empty());
    }

    #[tokio::test]
    async fn uninstall_unknown_plugin_returns_false() {
        let reg = registry();
        let ctx = ctx();
        assert!(!reg.uninstall("ghost", &ctx).await.unwrap());
    }

    #[tokio::test]
    async fn install_records_install_call_count_via_capabilities_hook() {
        let installs = Arc::new(AtomicU32::new(0));
        let installs_c = installs.clone();
        let p = Arc::new(NativePlugin::new(
            "p",
            "test",
            "0.1.0",
            PluginCapabilities::default(),
            move |_| {
                installs_c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            |_| Ok(()),
        ));
        let reg = registry();
        let ctx = ctx();
        reg.install(p, &ctx).await.unwrap();
        assert_eq!(installs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn list_names_is_lex_sorted() {
        let reg = registry();
        let ctx = ctx();
        reg.install(noop_plugin("zeta"), &ctx).await.unwrap();
        reg.install(noop_plugin("alpha"), &ctx).await.unwrap();
        reg.install(noop_plugin("mu"), &ctx).await.unwrap();
        assert_eq!(reg.list_names(), vec!["alpha", "mu", "zeta"]);
    }

    #[tokio::test]
    async fn get_returns_arc_to_installed_plugin() {
        let reg = registry();
        let ctx = ctx();
        reg.install(noop_plugin("x"), &ctx).await.unwrap();
        let got = reg.get("x").unwrap();
        assert_eq!(got.name(), "x");
        assert!(reg.get("ghost").is_none());
    }

    #[tokio::test]
    async fn load_wasm_with_noop_host_errors() {
        let reg = registry();
        let ctx = ctx();
        let err = reg
            .load_wasm(std::path::Path::new("nope.wasm"), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, PluginError::Wasm(_)));
    }
}
