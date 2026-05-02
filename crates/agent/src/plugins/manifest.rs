//! Plugin manifest parser — `plugin.json` schema.
//!
//! Hosts ship a tree like:
//!
//! ```text
//! plugins/
//! ├── awesome-tools/
//! │   ├── plugin.json
//! │   └── plugin.wasm        (third-party only)
//! └── my-builtin/
//!     └── plugin.json        (kind: native; loader supplies via code)
//! ```
//!
//! `plugin.json` parses into a [`PluginManifest`]. The host then
//! decides how to instantiate: native plugins are typically
//! constructed via direct Rust code (the manifest exists for
//! discovery/listing), WASM plugins are loaded by the
//! [`super::wasm_host::WasmPluginHost`] trait.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::plugin::{PluginCapabilities, PluginKind};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub version: String,
    pub kind: PluginKind,
    /// For WASM plugins, the relative path to the `.wasm` file.
    /// Ignored for native.
    #[serde(default)]
    pub wasm_path: Option<PathBuf>,
    #[serde(default)]
    pub capabilities: PluginCapabilities,
    /// Optional homepage / repo URL surfaced in the host UI.
    #[serde(default)]
    pub homepage: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    /// SPDX license expression. Hosts may filter out plugins whose
    /// license doesn't match the user's allowlist.
    #[serde(default)]
    pub license: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum PluginManifestError {
    #[error("manifest parse: {0}")]
    Parse(String),
    #[error("manifest validate: {0}")]
    Validate(String),
}

/// Parse + validate a `plugin.json` byte slice.
pub fn parse(bytes: &[u8]) -> Result<PluginManifest, PluginManifestError> {
    let m: PluginManifest =
        serde_json::from_slice(bytes).map_err(|e| PluginManifestError::Parse(e.to_string()))?;
    if m.name.is_empty() {
        return Err(PluginManifestError::Validate(
            "name must not be empty".into(),
        ));
    }
    if m.version.is_empty() {
        return Err(PluginManifestError::Validate(
            "version must not be empty".into(),
        ));
    }
    if m.kind == PluginKind::Wasm && m.wasm_path.is_none() {
        return Err(PluginManifestError::Validate(
            "kind=wasm requires wasm_path".into(),
        ));
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_native_manifest() {
        let json = r#"{
            "name": "my-plugin",
            "version": "0.1.0",
            "kind": "native"
        }"#;
        let m = parse(json.as_bytes()).unwrap();
        assert_eq!(m.name, "my-plugin");
        assert_eq!(m.kind, PluginKind::Native);
    }

    #[test]
    fn parse_wasm_manifest_requires_wasm_path() {
        let json = r#"{
            "name": "x",
            "version": "1",
            "kind": "wasm"
        }"#;
        match parse(json.as_bytes()) {
            Err(PluginManifestError::Validate(ref m)) if m.contains("wasm_path") => {}
            other => panic!("expected wasm_path validation error, got {other:?}"),
        }
    }

    #[test]
    fn parse_wasm_manifest_with_path() {
        let json = r#"{
            "name": "x",
            "version": "1",
            "kind": "wasm",
            "wasm_path": "plugin.wasm"
        }"#;
        let m = parse(json.as_bytes()).unwrap();
        assert_eq!(m.wasm_path, Some(PathBuf::from("plugin.wasm")));
    }

    #[test]
    fn empty_name_errors() {
        let json = r#"{"name": "", "version": "1", "kind": "native"}"#;
        assert!(matches!(
            parse(json.as_bytes()),
            Err(PluginManifestError::Validate(_))
        ));
    }

    #[test]
    fn empty_version_errors() {
        let json = r#"{"name": "x", "version": "", "kind": "native"}"#;
        assert!(matches!(
            parse(json.as_bytes()),
            Err(PluginManifestError::Validate(_))
        ));
    }

    #[test]
    fn capabilities_round_trip() {
        let json = r#"{
            "name": "x",
            "version": "1",
            "kind": "native",
            "capabilities": {
                "provides_tools": ["read"],
                "provides_hooks": ["log"],
                "provides_skills": ["greet"]
            }
        }"#;
        let m = parse(json.as_bytes()).unwrap();
        assert_eq!(m.capabilities.provides_tools, vec!["read"]);
        assert_eq!(m.capabilities.provides_hooks, vec!["log"]);
        assert_eq!(m.capabilities.provides_skills, vec!["greet"]);
    }

    #[test]
    fn full_manifest_serde_roundtrip() {
        let m = PluginManifest {
            name: "x".into(),
            description: "test".into(),
            version: "1.2.3".into(),
            kind: PluginKind::Wasm,
            wasm_path: Some("p.wasm".into()),
            capabilities: PluginCapabilities {
                provides_tools: vec!["t".into()],
                ..Default::default()
            },
            homepage: Some("https://example".into()),
            author: Some("alice".into()),
            license: Some("MIT".into()),
        };
        let json = serde_json::to_string(&m).unwrap();
        let parsed: PluginManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn malformed_json_errors() {
        match parse(b"{ broken") {
            Err(PluginManifestError::Parse(_)) => {}
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
