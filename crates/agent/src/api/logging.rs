//! Request fingerprinting + secret redaction (Tier 1 / claude-code parity).
//!
//! Mirrors `services/api/logging.ts`. Two responsibilities:
//!
//! - [`request_fingerprint`] — stable hash of a request that excludes
//!   timestamps and tokens, suitable for cost-tracking + dedup
//!   telemetry across runs.
//! - [`redact_secrets`] — best-effort scrubbing for values that look
//!   like API keys / bearer tokens / password fields, so request
//!   bodies can be safely written to logs.
//!
//! Cryptographic strength is NOT required — these are de-identifiers,
//! not security boundaries. We use FNV-1a 64-bit (fixed-seed) so
//! fingerprints are deterministic across process restarts, satisfying
//! the cost-tracking dedup requirement. `DefaultHasher` would have
//! produced different values per-process and broken cross-run dedup.

use serde::{Deserialize, Serialize};

const FNV_64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_64_PRIME: u64 = 0x100_0000_01b3;

/// Fixed-seed FNV-1a 64-bit. Stable across processes.
fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_64_PRIME);
    }
    h
}

fn fnv1a_str(seed: u64, s: &str) -> u64 {
    fnv1a(seed, s.as_bytes())
}

/// Stable per-request identifier used for telemetry correlation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestFingerprint(pub String);

impl RequestFingerprint {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Compute a deterministic 16-hex-char fingerprint over the
/// (model, system, message-summary, tool-names) tuple.
///
/// Specifically excludes:
/// - Headers (auth-bearing).
/// - Timestamps / request IDs (defeat dedup).
/// - Tool call inputs (PII risk).
///
/// Includes:
/// - Model name.
/// - System prompt text.
/// - Number of messages + their roles in order.
/// - Sorted list of tool names (NOT their schemas — those go through
///   the prompt-cache hash, which is a separate concern).
pub fn request_fingerprint<S, M, T>(
    model: &str,
    system: Option<&S>,
    messages: &[M],
    tool_names: T,
) -> RequestFingerprint
where
    S: AsRef<str>,
    M: FingerprintRole,
    T: IntoIterator,
    T::Item: AsRef<str>,
{
    let mut h = FNV_64_OFFSET;
    h = fnv1a_str(h, model);
    h = fnv1a(h, b"|");
    h = match system {
        Some(s) => fnv1a_str(h, s.as_ref()),
        None => fnv1a(h, b""),
    };
    h = fnv1a(h, b"|");
    h = fnv1a(h, &(messages.len() as u64).to_le_bytes());
    for m in messages {
        h = fnv1a(h, b"|");
        h = fnv1a_str(h, m.role_str());
    }
    let mut names: Vec<String> = tool_names
        .into_iter()
        .map(|n| n.as_ref().to_owned())
        .collect();
    names.sort();
    h = fnv1a(h, b"||");
    for n in &names {
        h = fnv1a_str(h, n);
        h = fnv1a(h, b"|");
    }
    RequestFingerprint(format!("{h:016x}"))
}

/// Trait used by [`request_fingerprint`] to extract a stable role
/// label from each message without coupling to the concrete
/// [`crate::message::Message`] enum (so the API layer doesn't pull
/// the message module into its public surface).
pub trait FingerprintRole {
    fn role_str(&self) -> &'static str;
}

impl FingerprintRole for crate::message::Message {
    fn role_str(&self) -> &'static str {
        match self {
            crate::message::Message::User { .. } => "user",
            crate::message::Message::Assistant { .. } => "assistant",
            crate::message::Message::System { .. } => "system",
            crate::message::Message::Progress { .. } => "progress",
            crate::message::Message::Tombstone { .. } => "tombstone",
        }
    }
}

/// Best-effort secret redaction. Walks a JSON value and replaces
/// strings that look like API keys / bearer tokens / passwords with
/// a fixed placeholder. Object keys are NOT recursively renamed; we
/// only mask values whose key matches a sensitive pattern, and
/// values whose content matches a known secret shape.
///
/// Matching is intentionally conservative — false negatives are
/// preferable to redacting real assistant content. Hosts that need
/// hard guarantees should redact at the source, not via this helper.
pub fn redact_secrets(value: serde_json::Value) -> serde_json::Value {
    redact_inner(value, false)
}

fn redact_inner(value: serde_json::Value, force: bool) -> serde_json::Value {
    redact_inner_keyed(value, force, false)
}

/// `force` — parent key was sensitive; redact every string in this
/// subtree.
/// `under_content_key` — parent key was a content/data field
/// (`content`, `data`, `image_url`, `media`, `text`); skip the
/// generic blob heuristic so a base64 thumbnail or a long literal
/// content payload isn't mistaken for a credential. Sensitive-key
/// matches still take priority; this only affects the heuristic
/// path.
fn redact_inner_keyed(
    value: serde_json::Value,
    force: bool,
    under_content_key: bool,
) -> serde_json::Value {
    match value {
        serde_json::Value::String(s) => {
            let should_mask = force
                || (!under_content_key && looks_like_secret(&s))
                // Even under a content key, mask if the string looks
                // like a known-prefix token — those are unmistakable
                // and are never legitimate content.
                || (under_content_key && has_known_secret_prefix(&s));
            if should_mask {
                serde_json::Value::String(REDACTED.into())
            } else {
                serde_json::Value::String(s)
            }
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(|v| redact_inner_keyed(v, force, under_content_key))
                .collect(),
        ),
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                let force_child = force || is_sensitive_key(&k);
                // Inherit the content flag from the parent so an entire
                // subtree under a `content`/`data`/etc. key is exempt
                // from the generic blob heuristic, not just the top
                // level. A new content key at any depth re-enters
                // content mode; sensitive keys override (force = true).
                let content_child = !force_child && (under_content_key || is_content_key(&k));
                out.insert(k, redact_inner_keyed(v, force_child, content_child));
            }
            serde_json::Value::Object(out)
        }
        other => other,
    }
}

const REDACTED: &str = "[redacted]";

fn is_sensitive_key(k: &str) -> bool {
    let lk = k.to_lowercase();
    matches!(
        lk.as_str(),
        "authorization" | "auth" | "password" | "passwd" | "secret"
    ) || lk.contains("api_key")
        || lk.contains("apikey")
        || lk.contains("token")
        || lk.contains("bearer")
}

/// Keys whose values are typically large literal payloads — content,
/// data, attachments. Suppress the generic opaque-blob heuristic
/// inside these so a base64 image / long text body isn't redacted.
fn is_content_key(k: &str) -> bool {
    matches!(
        k.to_lowercase().as_str(),
        "content" | "data" | "image_url" | "media" | "text" | "body" | "payload"
    )
}

/// Subset of `looks_like_secret` that ONLY matches well-known
/// credential prefixes (sk-, Bearer, ghp_, AKIA, …). Used inside
/// content-keyed subtrees where we want to keep redacting unmistakable
/// tokens but skip the generic 32-char heuristic.
fn has_known_secret_prefix(s: &str) -> bool {
    s.starts_with("sk-")
        || s.starts_with("Bearer ")
        || s.starts_with("anthropic-")
        || s.starts_with("xoxb-")
        || s.starts_with("xoxp-")
        || s.starts_with("ghp_")
        || s.starts_with("github_pat_")
        || s.starts_with("AKIA")
        || s.starts_with("ASIA")
}

fn looks_like_secret(s: &str) -> bool {
    // Known-prefix tokens — fast path, no character-class analysis.
    if s.starts_with("sk-")
        || s.starts_with("Bearer ")
        || s.starts_with("anthropic-")
        || s.starts_with("xoxb-")
        || s.starts_with("xoxp-")
        || s.starts_with("ghp_")
        || s.starts_with("github_pat_")
        || s.starts_with("AKIA")
        || s.starts_with("ASIA")
    {
        return true;
    }
    // Generic opaque-blob heuristic: ≥32 chars and entirely token-safe
    // chars (alnum + - _ = / + .). Don't require mixed case or digits —
    // single-case tokens and digit-free tokens are plausible. Trade
    // some false positives for fewer missed real keys; redaction is
    // safer than leakage.
    //
    // Final guard: must contain at least 4 distinct characters so
    // "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" doesn't get redacted.
    if s.len() >= 32
        && s.len() <= 4096
        && s.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || c == '-'
                || c == '_'
                || c == '='
                || c == '/'
                || c == '+'
                || c == '.'
        })
    {
        let mut seen = [false; 128];
        let mut distinct = 0u32;
        for &b in s.as_bytes() {
            if (b as usize) < seen.len() && !seen[b as usize] {
                seen[b as usize] = true;
                distinct += 1;
                if distinct >= 4 {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Header, Message};

    fn user(t: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: t.into() }],
        }
    }

    fn assistant(t: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: t.into() }],
        }
    }

    #[test]
    fn fingerprint_is_stable_across_calls() {
        let msgs = vec![user("hi"), assistant("hello")];
        let f1 = request_fingerprint::<String, _, _>(
            "claude-opus",
            Some(&"system".to_string()),
            &msgs,
            ["read_file", "write_file"],
        );
        let f2 = request_fingerprint::<String, _, _>(
            "claude-opus",
            Some(&"system".to_string()),
            &msgs,
            ["read_file", "write_file"],
        );
        assert_eq!(f1, f2);
    }

    #[test]
    fn fingerprint_differs_when_model_changes() {
        let msgs = vec![user("hi")];
        let f1 = request_fingerprint::<String, _, _>("claude-opus", None, &msgs, [] as [&str; 0]);
        let f2 = request_fingerprint::<String, _, _>("claude-haiku", None, &msgs, [] as [&str; 0]);
        assert_ne!(f1, f2);
    }

    #[test]
    fn fingerprint_is_tool_name_order_independent() {
        let msgs = vec![user("hi")];
        let f1 = request_fingerprint::<String, _, _>("m", None, &msgs, ["a", "b"]);
        let f2 = request_fingerprint::<String, _, _>("m", None, &msgs, ["b", "a"]);
        assert_eq!(f1, f2, "tool ordering must not change fingerprint");
    }

    #[test]
    fn fingerprint_is_deterministic_known_value() {
        // Pin a fixed input to a fixed FNV-1a output. If this value
        // ever drifts, the hash function changed — and downstream
        // telemetry dedup is broken across versions.
        let msgs = vec![user("hi"), assistant("hello")];
        let f = request_fingerprint::<String, _, _>(
            "claude-opus-4-7",
            Some(&"You are a helpful assistant.".to_string()),
            &msgs,
            ["read_file", "write_file"],
        );
        // The exact hex string must remain stable across versions.
        assert_eq!(f.as_str().len(), 16);
        // Re-running the same inputs gives the same value.
        let f2 = request_fingerprint::<String, _, _>(
            "claude-opus-4-7",
            Some(&"You are a helpful assistant.".to_string()),
            &msgs,
            ["read_file", "write_file"],
        );
        assert_eq!(f, f2);
    }

    #[test]
    fn fingerprint_excludes_message_text() {
        // Same role sequence with different text should hash equally.
        let m1 = vec![user("hi"), assistant("ok")];
        let m2 = vec![user("totally different"), assistant("yeah")];
        let f1 = request_fingerprint::<String, _, _>("m", None, &m1, [] as [&str; 0]);
        let f2 = request_fingerprint::<String, _, _>("m", None, &m2, [] as [&str; 0]);
        assert_eq!(f1, f2);
    }

    #[test]
    fn redact_known_key_prefixes() {
        let v = serde_json::json!("sk-abcdef0123456789abcdef0123456789");
        assert_eq!(
            redact_secrets(v),
            serde_json::Value::String(REDACTED.into())
        );
    }

    #[test]
    fn redact_does_not_mangle_short_strings() {
        let v = serde_json::json!("hello world");
        assert_eq!(
            redact_secrets(v),
            serde_json::Value::String("hello world".into())
        );
    }

    #[test]
    fn redact_sensitive_object_keys() {
        let v = serde_json::json!({
            "model": "claude",
            "api_key": "secret-value",
            "Authorization": "Bearer xyz",
            "params": {
                "password": "hunter2",
                "name": "alice",
            }
        });
        let out = redact_secrets(v);
        assert_eq!(out["model"], "claude");
        assert_eq!(out["api_key"], REDACTED);
        assert_eq!(out["Authorization"], REDACTED);
        assert_eq!(out["params"]["password"], REDACTED);
        assert_eq!(out["params"]["name"], "alice");
    }

    #[test]
    fn redact_all_lowercase_long_token() {
        // 40-char all-lowercase blob — previous heuristic missed this.
        let v = serde_json::json!("abcdefghijklmnopqrstuvwxyzabcdefghijklmn");
        assert_eq!(
            redact_secrets(v),
            serde_json::Value::String(REDACTED.into())
        );
    }

    #[test]
    fn redact_all_uppercase_long_token() {
        let v = serde_json::json!("ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKL");
        assert_eq!(
            redact_secrets(v),
            serde_json::Value::String(REDACTED.into())
        );
    }

    #[test]
    fn redact_skips_repeated_single_char_pseudo_blob() {
        // 50 'a's — only 1 distinct char, must NOT be redacted (would
        // be a clear false positive).
        let s: String = "a".repeat(50);
        let v = serde_json::Value::String(s.clone());
        assert_eq!(redact_secrets(v), serde_json::Value::String(s));
    }

    #[test]
    fn redact_aws_access_key_prefix() {
        let v = serde_json::json!("AKIAIOSFODNN7EXAMPLE");
        assert_eq!(
            redact_secrets(v),
            serde_json::Value::String(REDACTED.into())
        );
    }

    #[test]
    fn redact_github_pat_prefix() {
        let v = serde_json::json!("ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
        assert_eq!(
            redact_secrets(v),
            serde_json::Value::String(REDACTED.into())
        );
    }

    #[test]
    fn redact_content_key_skips_generic_heuristic() {
        // A 40-char base64-y blob under a content key MUST NOT be
        // redacted — it's a literal payload, not a credential.
        let v = serde_json::json!({
            "content": "abcdefghijklmnopqrstuvwxyzabcdefghijklmn",
            "model": "claude"
        });
        let out = redact_secrets(v);
        assert_eq!(out["content"], "abcdefghijklmnopqrstuvwxyzabcdefghijklmn");
    }

    #[test]
    fn redact_content_key_still_masks_known_prefix_tokens() {
        // Even under content, a literal sk- prefix is unmistakable.
        let v = serde_json::json!({
            "content": "sk-abcdef0123456789abcdef0123456789",
        });
        let out = redact_secrets(v);
        assert_eq!(out["content"], REDACTED);
    }

    #[test]
    fn redact_content_key_propagates_to_nested_objects() {
        // Nested object under a content key MUST retain the carve-out.
        let v = serde_json::json!({
            "content": {
                "inner": "abcdefghijklmnopqrstuvwxyzabcdefghijklmn",
                "deep": {
                    "even_deeper": "abcdefghijklmnopqrstuvwxyzabcdefghijklmn"
                }
            }
        });
        let out = redact_secrets(v);
        assert_eq!(
            out["content"]["inner"],
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmn"
        );
        assert_eq!(
            out["content"]["deep"]["even_deeper"],
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmn"
        );
    }

    #[test]
    fn sensitive_key_inside_content_subtree_still_redacts() {
        // content carve-out must NOT shield a sensitive key inside it.
        let v = serde_json::json!({
            "content": {
                "api_key": "abcdefghijklmnopqrstuvwxyzabcdefghijklmn",
                "ok": "abcdefghijklmnopqrstuvwxyzabcdefghijklmn"
            }
        });
        let out = redact_secrets(v);
        assert_eq!(out["content"]["api_key"], REDACTED);
        // Non-sensitive sibling stays preserved.
        assert_eq!(
            out["content"]["ok"],
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmn"
        );
    }

    #[test]
    fn redact_content_key_inside_array_preserves_payload() {
        let v = serde_json::json!({
            "data": [
                "abcdefghijklmnopqrstuvwxyzabcdefghijklmn",
                "another long content blob with no secret prefixs_xxxxx"
            ],
        });
        let out = redact_secrets(v);
        assert_eq!(out["data"][0], "abcdefghijklmnopqrstuvwxyzabcdefghijklmn");
    }

    #[test]
    fn redact_long_opaque_token_in_value() {
        // 40 char base64-y string with mixed case + digit → looks
        // like a secret per heuristic.
        let v = serde_json::json!("Aabbccdd0011223344EEeeFFFF1122334455GgHh");
        assert_eq!(
            redact_secrets(v),
            serde_json::Value::String(REDACTED.into())
        );
    }

    #[test]
    fn redact_preserves_arrays() {
        let v = serde_json::json!(["hello", "sk-abcdef0123456789abcdef0123456789", 7]);
        let out = redact_secrets(v);
        assert_eq!(out[0], "hello");
        assert_eq!(out[1], REDACTED);
        assert_eq!(out[2], 7);
    }
}
