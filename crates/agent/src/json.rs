//! JSON Schema generation + small helper functions (Phase 1 / Task 1.5).
//!
//! `schema::<T>()` produces a draft-07 JSON Schema for any type implementing
//! [`schemars::JsonSchema`]. The Tool trait in Phase 2 uses this to declare
//! input shape declaratively from a struct definition — no manual schema
//! editing.

use schemars::{schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Serialize};

use crate::error::AgentError;

/// Generate a JSON Schema for `T`. Returns `serde_json::Value` so callers
/// can embed it into Anthropic's `tools` payload directly.
///
/// Panics only if `T`'s schema produced by `schemars` cannot be serialized
/// to JSON, which would be a bug in `schemars` itself.
pub fn schema<T: JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schema_for!(T)).expect("schemars output is valid JSON")
}

/// Serialize-then-deserialize through JSON. Useful for normalizing
/// `serde_json::Value` blobs and for validating that two compatible types
/// share a common JSON encoding.
pub fn roundtrip<T: Serialize + DeserializeOwned>(value: &T) -> Result<T, AgentError> {
    let s = serde_json::to_string(value)?;
    let back = serde_json::from_str(&s)?;
    Ok(back)
}

/// Drill into a JSON value via a string-segment path. Returns `None` if any
/// segment is missing or wrong type.
///
/// ```ignore
/// let v = serde_json::json!({"a": {"b": {"c": 1}}});
/// assert_eq!(pluck(&v, &["a", "b", "c"]), Some(&serde_json::json!(1)));
/// ```
pub fn pluck<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for segment in path {
        current = current.get(segment)?;
    }
    Some(current)
}

/// Same as [`pluck`] but returns the value moved-out (cloned). Convenient
/// when crossing function boundaries.
pub fn pluck_owned(value: &serde_json::Value, path: &[&str]) -> Option<serde_json::Value> {
    pluck(value, path).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
    struct ReadFileInput {
        /// The path to read.
        path: String,
        /// Optional max bytes (default: unbounded).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_bytes: Option<u64>,
    }

    #[test]
    fn schema_emits_draft_2020_12() {
        // Anthropic's tool API validates against draft 2020-12. schemars 1.x
        // emits this draft by default; schemars 0.8 was draft-07.
        let s = schema::<ReadFileInput>();
        let dialect = s.get("$schema").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            dialect.contains("2020-12"),
            "expected 2020-12 dialect, got: {dialect}"
        );
    }

    #[test]
    fn schema_includes_field_descriptions() {
        let s = schema::<ReadFileInput>();
        // Schema is wrapped in a RootSchema — the actual properties live
        // under `properties`. Drill in with pluck.
        let path_desc = pluck(&s, &["properties", "path", "description"]);
        assert!(path_desc.is_some());
        assert_eq!(path_desc.unwrap().as_str(), Some("The path to read."));
    }

    #[test]
    fn schema_marks_required_fields() {
        let s = schema::<ReadFileInput>();
        let required = pluck(&s, &["required"]).unwrap();
        let arr = required.as_array().unwrap();
        let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"path"));
        assert!(!names.contains(&"max_bytes"));
    }

    #[test]
    fn roundtrip_preserves_value() {
        let v = ReadFileInput {
            path: "/tmp/x".into(),
            max_bytes: Some(1024),
        };
        let back = roundtrip(&v).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn pluck_basic() {
        let v = serde_json::json!({"a": {"b": {"c": 1}}});
        assert_eq!(pluck(&v, &["a", "b", "c"]), Some(&serde_json::json!(1)));
        assert_eq!(pluck(&v, &["a", "b"]).map(|v| v.is_object()), Some(true));
        assert!(pluck(&v, &["a", "missing"]).is_none());
        assert!(pluck(&v, &["a", "b", "c", "deeper"]).is_none());
    }

    #[test]
    fn pluck_owned_clones() {
        let v = serde_json::json!({"a": "hello"});
        let owned = pluck_owned(&v, &["a"]).unwrap();
        assert_eq!(owned, serde_json::json!("hello"));
    }
}
