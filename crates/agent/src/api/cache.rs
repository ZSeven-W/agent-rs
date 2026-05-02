//! Prompt cache break detection (Tier 1 / claude-code parity).
//!
//! Mirrors `services/api/promptCacheBreakDetection.ts`. Anthropic's
//! prompt cache hashes the system prompt + tool schema set + beta
//! header set; if any one of those changes, the cache breaks and the
//! next request gets billed at full input rates again.
//!
//! This module:
//!
//! - Snapshots the three components into a stable [`PromptCacheState`].
//! - Compares two states to produce a [`CacheBreakObservation`] with
//!   the specific [`CacheBreakKind`] that broke.
//! - Tracks observations over time via [`PromptCacheTracker`] so the
//!   host can correlate cache breaks with billing surprises.
//!
//! The host is responsible for hashing — we accept already-hashed
//! identifiers so callers can choose their cost-vs-fidelity trade-off
//! (full-text hash vs. content-addressed pointer).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::provider::ToolDefinition;

/// Snapshot of the prompt-cache-affecting inputs. Equal snapshots
/// hit cache; any difference breaks it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheState {
    /// Hash or content-pointer for the system prompt.
    pub system_hash: String,
    /// Hash or content-pointer for the tool schema set (typically the
    /// JSON-serialized list of tool definitions).
    pub tool_schema_hash: String,
    /// Sorted set of beta headers active for the request — order-
    /// independent because Anthropic dedups on the server side.
    pub beta_headers: BTreeSet<String>,
}

impl PromptCacheState {
    pub fn new(
        system_hash: impl Into<String>,
        tool_schema_hash: impl Into<String>,
        beta_headers: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            system_hash: system_hash.into(),
            tool_schema_hash: tool_schema_hash.into(),
            beta_headers: beta_headers.into_iter().collect(),
        }
    }

    /// Empty placeholder used by [`PromptCacheTracker`] before the
    /// first request lands.
    pub fn empty() -> Self {
        Self {
            system_hash: String::new(),
            tool_schema_hash: String::new(),
            beta_headers: BTreeSet::new(),
        }
    }

    /// Convenience: build a state from a system prompt + a list of tool
    /// definitions + beta headers. The tool schema hash is computed by
    /// [`fingerprint_tools`] (canonical name-sorted JSON + FNV-1a 64).
    /// The system prompt is hashed identically so the result is fully
    /// self-contained.
    pub fn fingerprint(
        system: &str,
        tools: &[ToolDefinition],
        beta_headers: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            system_hash: fingerprint_string(system),
            tool_schema_hash: fingerprint_tools(tools),
            beta_headers: beta_headers.into_iter().collect(),
        }
    }

    /// Compare two states; returns [`None`] if they match (cache hit),
    /// or [`Some`] with the kind of break and a human-readable
    /// description.
    pub fn diff(&self, other: &Self) -> Option<CacheBreakObservation> {
        if self.system_hash != other.system_hash {
            return Some(CacheBreakObservation {
                kind: CacheBreakKind::SystemPromptChanged,
                from: self.clone(),
                to: other.clone(),
                detail: format!(
                    "system hash {} → {}",
                    short(&self.system_hash),
                    short(&other.system_hash)
                ),
            });
        }
        if self.tool_schema_hash != other.tool_schema_hash {
            return Some(CacheBreakObservation {
                kind: CacheBreakKind::ToolSchemaChanged,
                from: self.clone(),
                to: other.clone(),
                detail: format!(
                    "tool schema hash {} → {}",
                    short(&self.tool_schema_hash),
                    short(&other.tool_schema_hash)
                ),
            });
        }
        if self.beta_headers != other.beta_headers {
            let added: Vec<_> = other
                .beta_headers
                .difference(&self.beta_headers)
                .cloned()
                .collect();
            let removed: Vec<_> = self
                .beta_headers
                .difference(&other.beta_headers)
                .cloned()
                .collect();
            return Some(CacheBreakObservation {
                kind: CacheBreakKind::BetaHeadersChanged,
                from: self.clone(),
                to: other.clone(),
                detail: format!("beta headers added={added:?}, removed={removed:?}"),
            });
        }
        None
    }
}

/// FNV-1a 64-bit hash with the standard offset basis. Returns hex
/// without leading zeros so two structurally-distinct inputs produce
/// different strings even when the underlying numeric hashes start
/// with leading zeros.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Stable fingerprint of a single string. Used for the system prompt;
/// the format is `"sys-<hex>"` so a `tool_schema_hash` that happens to
/// match a numeric system hash never collides.
pub fn fingerprint_string(s: &str) -> String {
    format!("sys-{:016x}", fnv1a_64(s.as_bytes()))
}

/// Stable fingerprint of a tool schema set. The fingerprint is
/// invariant under both registration order AND object-key insertion
/// order inside the input schemas:
///
/// 1. Tools are sorted by name.
/// 2. Each tool's `input_schema` is recursively canonicalized so all
///    JSON object keys appear in lexicographic order. (This matters
///    because `ollama-rs` enables `serde_json/preserve_order`, which
///    would otherwise let two equivalent schemas hash differently
///    depending on insertion order.)
/// 3. The canonical JSON bytes are FNV-1a-64 hashed and rendered as a
///    `tools-<hex>` string for symmetry with [`fingerprint_string`].
///
/// FNV-1a is not a cryptographic hash — it's used here for a
/// best-effort cache-break signal, not a security boundary.
pub fn fingerprint_tools(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return "tools-empty".to_string();
    }
    let mut sorted: Vec<&ToolDefinition> = tools.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut canonical = String::with_capacity(64 * sorted.len());
    canonical.push('[');
    for (i, t) in sorted.iter().enumerate() {
        if i > 0 {
            canonical.push(',');
        }
        canonical.push_str("{\"name\":");
        if let Ok(s) = serde_json::to_string(&t.name) {
            canonical.push_str(&s);
        }
        canonical.push_str(",\"description\":");
        if let Ok(s) = serde_json::to_string(&t.description) {
            canonical.push_str(&s);
        }
        canonical.push_str(",\"input_schema\":");
        write_canonical(&mut canonical, &t.input_schema);
        canonical.push('}');
    }
    canonical.push(']');
    format!("tools-{:016x}", fnv1a_64(canonical.as_bytes()))
}

/// Recursively serialize a `serde_json::Value` with object keys sorted
/// lexicographically by Unicode scalar value. Other shapes are
/// delegated to `serde_json::to_string` (which is itself stable for
/// arrays / strings / numbers / bools / nulls — only object key order
/// is feature-flagged through `preserve_order`).
fn write_canonical(out: &mut String, v: &serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                // Use serde_json's escaping for the key by serializing
                // a String Value to a string and appending.
                if let Ok(key_json) = serde_json::to_string(k) {
                    out.push_str(&key_json);
                }
                out.push(':');
                write_canonical(out, &map[*k]);
            }
            out.push('}');
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            }
            out.push(']');
        }
        // Strings / numbers / bools / null have a single canonical
        // serialization in JSON; defer to serde_json.
        _ => {
            if let Ok(s) = serde_json::to_string(v) {
                out.push_str(&s);
            }
        }
    }
}

fn short(s: &str) -> String {
    let count = s.chars().count();
    if count <= 8 {
        return s.to_string();
    }
    let head: String = s.chars().take(4).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}…{tail}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheBreakKind {
    /// System prompt text changed.
    SystemPromptChanged,
    /// Tool schema (definition list) changed.
    ToolSchemaChanged,
    /// Beta header set changed (added or removed an opt-in feature).
    BetaHeadersChanged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheBreakObservation {
    pub kind: CacheBreakKind,
    pub from: PromptCacheState,
    pub to: PromptCacheState,
    pub detail: String,
}

/// Stateful tracker — keeps the most recent state and emits an
/// observation when it changes.
#[derive(Debug, Clone)]
pub struct PromptCacheTracker {
    last: PromptCacheState,
    /// Total number of breaks observed since construction. Useful for
    /// telemetry assertions ("we shouldn't see >N breaks per session").
    pub break_count: u64,
}

impl Default for PromptCacheTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptCacheTracker {
    pub fn new() -> Self {
        Self {
            last: PromptCacheState::empty(),
            break_count: 0,
        }
    }

    /// Observe a new state. Returns [`Some`] iff this differs from the
    /// previously-observed state (and was non-empty), i.e., a real
    /// cache break — the first call after construction is a baseline
    /// rather than a break.
    pub fn observe(&mut self, next: PromptCacheState) -> Option<CacheBreakObservation> {
        if self.last == PromptCacheState::empty() {
            self.last = next;
            return None;
        }
        let observation = self.last.diff(&next);
        self.last = next;
        if observation.is_some() {
            self.break_count = self.break_count.saturating_add(1);
        }
        observation
    }

    /// The most recently observed state. Returns the empty placeholder
    /// before any observation has been made.
    pub fn last(&self) -> &PromptCacheState {
        &self.last
    }

    /// Forget the previous baseline. Use when starting a new logical
    /// session inside the same long-lived process — the next
    /// `observe()` will be a fresh baseline rather than a spurious
    /// "break" against the prior session's state.
    ///
    /// **Preserves [`Self::break_count`]** so lifetime telemetry
    /// survives across logical sessions. If you want to reset the
    /// metrics too, call [`Self::reset_metrics`].
    pub fn reset(&mut self) {
        self.last = PromptCacheState::empty();
    }

    /// Reset the lifetime metrics counter. Does NOT touch the
    /// observed-state baseline; call [`Self::reset`] separately for
    /// that.
    pub fn reset_metrics(&mut self) {
        self.break_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(hash: &str, tool: &str, betas: &[&str]) -> PromptCacheState {
        PromptCacheState::new(
            hash,
            tool,
            betas.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn short_handles_multibyte_chars_without_panic() {
        // 12 CJK chars, > 8 boundary; previously sliced bytes [..4] which
        // landed mid-char and panicked.
        let s = "你好世界你好世界你好世界";
        let out = super::short(s);
        // Result should contain ellipsis and not crash.
        assert!(out.contains('…'));
    }

    #[test]
    fn equal_states_diff_to_none() {
        let a = s("h1", "t1", &["b1"]);
        let b = s("h1", "t1", &["b1"]);
        assert!(a.diff(&b).is_none());
    }

    #[test]
    fn system_hash_difference_breaks() {
        let a = s("h1", "t1", &["b1"]);
        let b = s("h2", "t1", &["b1"]);
        let obs = a.diff(&b).unwrap();
        assert_eq!(obs.kind, CacheBreakKind::SystemPromptChanged);
    }

    #[test]
    fn tool_schema_difference_breaks() {
        let a = s("h1", "t1", &["b1"]);
        let b = s("h1", "t2", &["b1"]);
        let obs = a.diff(&b).unwrap();
        assert_eq!(obs.kind, CacheBreakKind::ToolSchemaChanged);
    }

    #[test]
    fn beta_headers_difference_breaks_with_added_and_removed() {
        let a = s("h1", "t1", &["b1", "b2"]);
        let b = s("h1", "t1", &["b1", "b3"]);
        let obs = a.diff(&b).unwrap();
        assert_eq!(obs.kind, CacheBreakKind::BetaHeadersChanged);
        assert!(obs.detail.contains("b3"));
        assert!(obs.detail.contains("b2"));
    }

    #[test]
    fn beta_headers_order_independent() {
        let a = s("h1", "t1", &["b1", "b2"]);
        let b = s("h1", "t1", &["b2", "b1"]);
        assert!(a.diff(&b).is_none());
    }

    #[test]
    fn tracker_first_observation_is_baseline_not_break() {
        let mut t = PromptCacheTracker::new();
        let initial = s("h1", "t1", &["b1"]);
        assert!(t.observe(initial.clone()).is_none());
        assert_eq!(t.break_count, 0);
        assert_eq!(t.last(), &initial);
    }

    #[test]
    fn tracker_subsequent_change_is_break() {
        let mut t = PromptCacheTracker::new();
        t.observe(s("h1", "t1", &["b1"]));
        let obs = t.observe(s("h2", "t1", &["b1"])).unwrap();
        assert_eq!(obs.kind, CacheBreakKind::SystemPromptChanged);
        assert_eq!(t.break_count, 1);
    }

    #[test]
    fn tracker_subsequent_match_is_no_op() {
        let mut t = PromptCacheTracker::new();
        t.observe(s("h1", "t1", &["b1"]));
        assert!(t.observe(s("h1", "t1", &["b1"])).is_none());
        assert_eq!(t.break_count, 0);
    }

    #[test]
    fn tracker_reset_re_baselines_next_observation() {
        let mut t = PromptCacheTracker::new();
        t.observe(s("h1", "t1", &["b1"]));
        t.observe(s("h2", "t1", &["b1"]));
        assert_eq!(t.break_count, 1);
        t.reset();
        // First observe after reset is a baseline, not a break.
        assert!(t.observe(s("h99", "t99", &["b99"])).is_none());
        assert_eq!(t.break_count, 1, "reset preserves lifetime metrics");
    }

    #[test]
    fn fingerprint_tools_is_order_invariant() {
        let a = ToolDefinition::new("alpha", "first", serde_json::json!({"type": "object"}));
        let b = ToolDefinition::new("beta", "second", serde_json::json!({"type": "object"}));
        let h1 = fingerprint_tools(&[a.clone(), b.clone()]);
        let h2 = fingerprint_tools(&[b, a]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn fingerprint_tools_changes_when_schema_changes() {
        let a = ToolDefinition::new("t", "d", serde_json::json!({"type": "object"}));
        let b = ToolDefinition::new(
            "t",
            "d",
            serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
        );
        assert_ne!(fingerprint_tools(&[a]), fingerprint_tools(&[b]));
    }

    #[test]
    fn fingerprint_tools_invariant_under_object_key_order() {
        // With serde_json/preserve_order (enabled transitively by
        // ollama-rs in the `full` feature), insertion order is
        // preserved in `serde_json::Value`. The canonicalizer must
        // sort object keys so equivalent schemas hash identically.
        // Build two Values with deliberately different insertion
        // orders.
        let mut a_obj = serde_json::Map::new();
        a_obj.insert("type".to_string(), serde_json::json!("object"));
        a_obj.insert(
            "properties".to_string(),
            serde_json::json!({"x": {"type": "number"}, "y": {"type": "string"}}),
        );
        let a = ToolDefinition::new("t", "d", serde_json::Value::Object(a_obj));

        let mut b_obj = serde_json::Map::new();
        b_obj.insert(
            "properties".to_string(),
            serde_json::json!({"y": {"type": "string"}, "x": {"type": "number"}}),
        );
        b_obj.insert("type".to_string(), serde_json::json!("object"));
        let b = ToolDefinition::new("t", "d", serde_json::Value::Object(b_obj));

        assert_eq!(fingerprint_tools(&[a]), fingerprint_tools(&[b]));
    }

    #[test]
    fn fingerprint_tools_empty_marker() {
        let h = fingerprint_tools(&[]);
        assert_eq!(h, "tools-empty");
    }

    #[test]
    fn fingerprint_string_is_deterministic() {
        let h1 = fingerprint_string("you are concise");
        let h2 = fingerprint_string("you are concise");
        assert_eq!(h1, h2);
        assert_ne!(h1, fingerprint_string("you are verbose"));
    }

    #[test]
    fn prompt_cache_state_fingerprint_helper() {
        let tools = vec![ToolDefinition::new(
            "calc",
            "math",
            serde_json::json!({"type": "object"}),
        )];
        let s = PromptCacheState::fingerprint("be concise", &tools, ["beta-1".to_string()]);
        assert_eq!(s.system_hash, fingerprint_string("be concise"));
        assert_eq!(s.tool_schema_hash, fingerprint_tools(&tools));
        assert!(s.beta_headers.contains("beta-1"));
    }

    #[test]
    fn fingerprint_tools_reflects_description_change() {
        let a = ToolDefinition::new("t", "first", serde_json::json!({"type": "object"}));
        let b = ToolDefinition::new("t", "second", serde_json::json!({"type": "object"}));
        assert_ne!(fingerprint_tools(&[a]), fingerprint_tools(&[b]));
    }

    #[test]
    fn tracker_break_count_accumulates() {
        let mut t = PromptCacheTracker::new();
        t.observe(s("h1", "t1", &["b1"]));
        t.observe(s("h2", "t1", &["b1"]));
        t.observe(s("h2", "t2", &["b1"]));
        t.observe(s("h2", "t2", &["b1", "b2"]));
        assert_eq!(t.break_count, 3);
    }
}
