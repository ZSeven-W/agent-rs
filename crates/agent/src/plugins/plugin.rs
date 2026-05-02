//! Plugin trait + capabilities.
//!
//! `Plugin` is the runtime handle the registry holds. Both native
//! and WASM plugins implement the same interface; the registry
//! doesn't care which.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::hook::HookRunner;
use crate::skills::SkillRegistry;
use crate::tool::ToolRegistry;

#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("plugin install: {0}")]
    Install(String),
    #[error("plugin uninstall: {0}")]
    Uninstall(String),
    #[error("wasm host: {0}")]
    Wasm(String),
}

/// Whether the plugin is host-trusted (native) or sandboxed (WASM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PluginKind {
    Native,
    Wasm,
}

/// What a plugin advertises before install. The registry lists
/// these so the host UI can render a "what does this plugin do?"
/// surface.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginCapabilities {
    pub provides_tools: Vec<String>,
    pub provides_hooks: Vec<String>,
    pub provides_skills: Vec<String>,
}

/// Bundle of registries the host hands to a plugin during install.
/// The plugin pushes its tools/hooks/skills into them.
#[derive(Debug, Clone)]
pub struct InstallContext {
    pub tools: Arc<ToolRegistry>,
    pub hooks: Arc<HookRunner>,
    pub skills: SkillRegistry,
}

#[async_trait]
pub trait Plugin: std::fmt::Debug + Send + Sync {
    /// Stable plugin identifier — duplicates rejected at install time.
    fn name(&self) -> &str;
    /// One-line human-readable description.
    fn description(&self) -> &str;
    /// SemVer-style version string the host displays + uses for
    /// upgrade-detection.
    fn version(&self) -> &str;
    fn kind(&self) -> PluginKind;
    fn capabilities(&self) -> PluginCapabilities;

    /// Install the plugin into the supplied registries. Idempotent
    /// from the plugin's perspective — calling install on an
    /// already-installed plugin is a host error caught by the
    /// registry, not the plugin.
    async fn install(&self, ctx: &InstallContext) -> Result<(), PluginError>;

    /// Reverse of `install` — remove anything the plugin contributed.
    /// Best-effort; failures are reported but do not undo the
    /// registry's removal of the plugin.
    async fn uninstall(&self, ctx: &InstallContext) -> Result<(), PluginError>;
}

/// Convenience adapter for plugins that want a simpler closure-based
/// build path without writing a full `impl Plugin`.
pub struct NativePlugin<I, U>
where
    I: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
    U: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
{
    name: String,
    description: String,
    version: String,
    capabilities: PluginCapabilities,
    install_fn: I,
    uninstall_fn: U,
}

impl<I, U> NativePlugin<I, U>
where
    I: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
    U: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
{
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        version: impl Into<String>,
        capabilities: PluginCapabilities,
        install_fn: I,
        uninstall_fn: U,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version: version.into(),
            capabilities,
            install_fn,
            uninstall_fn,
        }
    }
}

impl<I, U> std::fmt::Debug for NativePlugin<I, U>
where
    I: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
    U: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativePlugin")
            .field("name", &self.name)
            .field("version", &self.version)
            .finish()
    }
}

#[async_trait]
impl<I, U> Plugin for NativePlugin<I, U>
where
    I: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
    U: Fn(&InstallContext) -> Result<(), PluginError> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn version(&self) -> &str {
        &self.version
    }
    fn kind(&self) -> PluginKind {
        PluginKind::Native
    }
    fn capabilities(&self) -> PluginCapabilities {
        self.capabilities.clone()
    }
    async fn install(&self, ctx: &InstallContext) -> Result<(), PluginError> {
        (self.install_fn)(ctx)
    }
    async fn uninstall(&self, ctx: &InstallContext) -> Result<(), PluginError> {
        (self.uninstall_fn)(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn install_ctx() -> InstallContext {
        InstallContext {
            tools: Arc::new(ToolRegistry::new()),
            hooks: Arc::new(HookRunner::new()),
            skills: SkillRegistry::new(),
        }
    }

    #[tokio::test]
    async fn native_plugin_install_and_uninstall_fire() {
        let installs = Arc::new(AtomicU32::new(0));
        let uninstalls = Arc::new(AtomicU32::new(0));
        let installs_c = installs.clone();
        let uninstalls_c = uninstalls.clone();
        let p = NativePlugin::new(
            "x",
            "test",
            "0.1.0",
            PluginCapabilities::default(),
            move |_| {
                installs_c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            move |_| {
                uninstalls_c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        );
        let ctx = install_ctx();
        p.install(&ctx).await.unwrap();
        p.uninstall(&ctx).await.unwrap();
        assert_eq!(installs.load(Ordering::SeqCst), 1);
        assert_eq!(uninstalls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn native_plugin_propagates_install_error() {
        let p = NativePlugin::new(
            "x",
            "t",
            "0.1.0",
            PluginCapabilities::default(),
            |_| Err(PluginError::Install("nope".into())),
            |_| Ok(()),
        );
        let err = p.install(&install_ctx()).await.unwrap_err();
        assert!(matches!(err, PluginError::Install(_)));
    }

    #[test]
    fn capabilities_default_is_empty() {
        let c = PluginCapabilities::default();
        assert!(c.provides_tools.is_empty());
        assert!(c.provides_hooks.is_empty());
        assert!(c.provides_skills.is_empty());
    }

    #[test]
    fn capabilities_serde_roundtrip() {
        let c = PluginCapabilities {
            provides_tools: vec!["a".into()],
            provides_hooks: vec!["b".into()],
            provides_skills: vec!["c".into()],
        };
        let json = serde_json::to_string(&c).unwrap();
        let parsed: PluginCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, c);
    }
}
