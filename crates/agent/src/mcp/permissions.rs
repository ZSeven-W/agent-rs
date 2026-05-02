//! Per-channel MCP permissions (Tier 1 / claude-code parity).
//!
//! Mirrors `services/mcp/channelPermissions.ts`. Each registered MCP
//! server (channel) has its own allow/deny lists for tools and
//! resources, layered on top of the agent-wide
//! [`crate::permission::PermissionManager`]. The MCP-specific layer
//! exists because tools advertised by an MCP server share a server
//! namespace — a host may want to allow tool X from server A but
//! deny it from server B.
//!
//! ## Resolution
//!
//! 1. Channel deny → return `Deny`.
//! 2. Channel allow → return `Allow`.
//! 3. Channel default → return `Default` (let the outer
//!    [`crate::permission::PermissionManager`] decide).
//!
//! ## Composition with the outer permission manager
//!
//! Channel `Allow` on its own does NOT automatically bypass an
//! outer deny rule. Callers should apply the policy:
//!
//! - If the channel returns `Deny` → final `Deny` (channel veto wins).
//! - If the channel returns `Default` → defer entirely to outer.
//! - If the channel returns `Allow` → check outer; outer `Deny` still
//!   wins, otherwise `Allow`.
//!
//! [`combine`] implements that composition for callers that don't
//! want to write the matrix by hand.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelDecision {
    Allow,
    Deny,
    /// Defer to the outer permission manager.
    Default,
}

/// Outer (agent-wide) decision shape. The host's permission manager
/// returns one of these; pair with [`combine`] to fold the channel-
/// specific decision into a final answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterDecision {
    Allow,
    Deny,
    /// Outer didn't have a strong opinion (e.g., no matching rule).
    Default,
}

/// Combine a channel decision with the outer permission manager's
/// answer. The composition rule:
///
/// - Channel `Deny` → `Deny` (channel veto wins outright).
/// - Channel `Default` → whatever outer says.
/// - Channel `Allow` → `Allow` UNLESS outer is `Deny`, in which case
///   outer wins.
pub fn combine(channel: ChannelDecision, outer: OuterDecision) -> OuterDecision {
    match channel {
        ChannelDecision::Deny => OuterDecision::Deny,
        ChannelDecision::Default => outer,
        ChannelDecision::Allow => match outer {
            OuterDecision::Deny => OuterDecision::Deny,
            // Channel allow trumps outer Default; outer Allow + channel
            // Allow stays Allow.
            _ => OuterDecision::Allow,
        },
    }
}

/// Permission set for a single MCP channel.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelPermissions {
    /// Tools explicitly allowed. `*` is a wildcard match.
    #[serde(default)]
    pub allow_tools: BTreeSet<String>,
    /// Tools explicitly denied. `*` is a wildcard match.
    #[serde(default)]
    pub deny_tools: BTreeSet<String>,
    /// Resource URIs explicitly allowed. Prefix-match.
    #[serde(default)]
    pub allow_resources: BTreeSet<String>,
    /// Resource URIs explicitly denied. Prefix-match.
    #[serde(default)]
    pub deny_resources: BTreeSet<String>,
}

impl ChannelPermissions {
    pub fn evaluate_tool(&self, tool_name: &str) -> ChannelDecision {
        if self.deny_tools.contains("*") || self.deny_tools.contains(tool_name) {
            return ChannelDecision::Deny;
        }
        if self.allow_tools.contains("*") || self.allow_tools.contains(tool_name) {
            return ChannelDecision::Allow;
        }
        ChannelDecision::Default
    }

    pub fn evaluate_resource(&self, uri: &str) -> ChannelDecision {
        if self.deny_resources.contains("*")
            || self.deny_resources.iter().any(|p| uri.starts_with(p))
        {
            return ChannelDecision::Deny;
        }
        if self.allow_resources.contains("*")
            || self.allow_resources.iter().any(|p| uri.starts_with(p))
        {
            return ChannelDecision::Allow;
        }
        ChannelDecision::Default
    }

    pub fn allow_tool(mut self, name: impl Into<String>) -> Self {
        self.allow_tools.insert(name.into());
        self
    }
    pub fn deny_tool(mut self, name: impl Into<String>) -> Self {
        self.deny_tools.insert(name.into());
        self
    }
    pub fn allow_resource(mut self, uri_prefix: impl Into<String>) -> Self {
        self.allow_resources.insert(uri_prefix.into());
        self
    }
    pub fn deny_resource(mut self, uri_prefix: impl Into<String>) -> Self {
        self.deny_resources.insert(uri_prefix.into());
        self
    }
}

/// Per-channel permission registry — one [`ChannelPermissions`] per
/// MCP server name.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpPermissionRegistry {
    pub channels: BTreeMap<String, ChannelPermissions>,
}

impl McpPermissionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(&mut self, channel: impl Into<String>, perms: ChannelPermissions) {
        self.channels.insert(channel.into(), perms);
    }

    pub fn evaluate_tool(&self, channel: &str, tool_name: &str) -> ChannelDecision {
        match self.channels.get(channel) {
            Some(p) => p.evaluate_tool(tool_name),
            None => ChannelDecision::Default,
        }
    }

    pub fn evaluate_resource(&self, channel: &str, uri: &str) -> ChannelDecision {
        match self.channels.get(channel) {
            Some(p) => p.evaluate_resource(uri),
            None => ChannelDecision::Default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_beats_allow() {
        let p = ChannelPermissions::default().allow_tool("x").deny_tool("x");
        assert_eq!(p.evaluate_tool("x"), ChannelDecision::Deny);
    }

    #[test]
    fn unknown_tool_is_default() {
        let p = ChannelPermissions::default().allow_tool("known");
        assert_eq!(p.evaluate_tool("other"), ChannelDecision::Default);
    }

    #[test]
    fn wildcard_allows_everything() {
        let p = ChannelPermissions::default().allow_tool("*");
        assert_eq!(p.evaluate_tool("anything"), ChannelDecision::Allow);
    }

    #[test]
    fn wildcard_deny_blanks() {
        let p = ChannelPermissions::default().allow_tool("*").deny_tool("*");
        assert_eq!(p.evaluate_tool("foo"), ChannelDecision::Deny);
    }

    #[test]
    fn resource_prefix_match() {
        let p = ChannelPermissions::default().allow_resource("file://repo/");
        assert_eq!(
            p.evaluate_resource("file://repo/x.rs"),
            ChannelDecision::Allow
        );
        assert_eq!(
            p.evaluate_resource("file://other/x.rs"),
            ChannelDecision::Default
        );
    }

    #[test]
    fn resource_deny_prefix_blocks() {
        let p = ChannelPermissions::default().deny_resource("file:///etc/");
        assert_eq!(
            p.evaluate_resource("file:///etc/passwd"),
            ChannelDecision::Deny
        );
        assert_eq!(
            p.evaluate_resource("file:///home/x"),
            ChannelDecision::Default
        );
    }

    #[test]
    fn resource_deny_wildcard_blocks_everything() {
        let p = ChannelPermissions::default().deny_resource("*");
        assert_eq!(p.evaluate_resource("file:///x"), ChannelDecision::Deny);
        assert_eq!(p.evaluate_resource("https://x"), ChannelDecision::Deny);
    }

    #[test]
    fn resource_deny_beats_allow_with_overlapping_prefix() {
        let p = ChannelPermissions::default()
            .allow_resource("file://repo/")
            .deny_resource("file://repo/secrets/");
        // allow root prefix; deny narrower prefix. Deny wins for the
        // narrower path.
        assert_eq!(
            p.evaluate_resource("file://repo/secrets/key.pem"),
            ChannelDecision::Deny
        );
        assert_eq!(
            p.evaluate_resource("file://repo/src/main.rs"),
            ChannelDecision::Allow
        );
    }

    #[test]
    fn registry_per_channel_evaluation() {
        let mut reg = McpPermissionRegistry::new();
        reg.upsert(
            "github",
            ChannelPermissions::default().allow_tool("create_issue"),
        );
        reg.upsert("linear", ChannelPermissions::default().deny_tool("delete"));
        assert_eq!(
            reg.evaluate_tool("github", "create_issue"),
            ChannelDecision::Allow
        );
        assert_eq!(reg.evaluate_tool("linear", "delete"), ChannelDecision::Deny);
        assert_eq!(
            reg.evaluate_tool("unknown", "any"),
            ChannelDecision::Default
        );
    }

    #[test]
    fn combine_channel_deny_always_wins() {
        for outer in [
            OuterDecision::Allow,
            OuterDecision::Deny,
            OuterDecision::Default,
        ] {
            assert_eq!(combine(ChannelDecision::Deny, outer), OuterDecision::Deny);
        }
    }

    #[test]
    fn combine_channel_default_defers_to_outer() {
        assert_eq!(
            combine(ChannelDecision::Default, OuterDecision::Allow),
            OuterDecision::Allow
        );
        assert_eq!(
            combine(ChannelDecision::Default, OuterDecision::Deny),
            OuterDecision::Deny
        );
        assert_eq!(
            combine(ChannelDecision::Default, OuterDecision::Default),
            OuterDecision::Default
        );
    }

    #[test]
    fn combine_channel_allow_bows_to_outer_deny() {
        assert_eq!(
            combine(ChannelDecision::Allow, OuterDecision::Deny),
            OuterDecision::Deny
        );
        assert_eq!(
            combine(ChannelDecision::Allow, OuterDecision::Allow),
            OuterDecision::Allow
        );
        assert_eq!(
            combine(ChannelDecision::Allow, OuterDecision::Default),
            OuterDecision::Allow
        );
    }

    #[test]
    fn roundtrip_serialization() {
        let mut reg = McpPermissionRegistry::new();
        reg.upsert(
            "x",
            ChannelPermissions::default().allow_tool("a").deny_tool("b"),
        );
        let json = serde_json::to_string(&reg).unwrap();
        let parsed: McpPermissionRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.channels.len(), 1);
    }
}
