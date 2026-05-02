//! Structured-output config (Tier 1 / claude-code parity).
//!
//! Tells the provider to constrain assistant output to a JSON shape:
//!
//! - **Anthropic**: no first-class JSON-schema mode in the public
//!   Messages API at the time of writing — the host applies the
//!   schema as a system-prompt directive + post-stream validation
//!   ([`OutputConfig::validate_text`]).
//! - **OpenAI / OpenAI-compatible**: `response_format = {"type":
//!   "json_schema", "json_schema": {...}}` — the API enforces it.
//!
//! [`OutputConfig`] is provider-agnostic; the wire shape is decided
//! per adapter.

use serde::{Deserialize, Serialize};

/// What kind of structured-output behavior we want.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum OutputMode {
    /// No constraint — the default, free-form text.
    #[default]
    FreeForm,
    /// JSON object only — provider should still emit valid JSON, but
    /// no schema is enforced.
    JsonObject,
    /// JSON object that conforms to `schema`. Adapters should pass the
    /// schema through to the provider when supported, or wrap it in a
    /// system-prompt directive when not. **The built-in
    /// [`OutputConfig::validate_text`] only enforces `type`,
    /// `properties`, and `required` for parity with claude-code's
    /// host-side guard** — richer keywords (`oneOf`, `items`,
    /// `additionalProperties`, `pattern`, `enum`, …) are passed
    /// through to the provider but NOT enforced post-stream. Hosts
    /// that need full draft-2020-12 validation should plug in a
    /// dedicated validator after [`validate_text`] returns.
    JsonSchema {
        /// Stable name for the schema — surfaced to the model and
        /// useful for telemetry.
        name: String,
        /// JSON schema (typically draft 2020-12).
        schema: serde_json::Value,
        /// When `true`, validation failures should NOT be retried —
        /// the caller wants visibility, not auto-correction.
        strict: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    pub mode: OutputMode,
    /// Optional: the maximum number of validation retries when
    /// `mode = JsonSchema { strict: false }`. The provider adapter
    /// uses this to bound a "your output didn't match the schema —
    /// try again" feedback loop.
    pub max_validation_retries: u32,
}

impl OutputConfig {
    pub fn free_form() -> Self {
        Self {
            mode: OutputMode::FreeForm,
            max_validation_retries: 0,
        }
    }

    pub fn json_object() -> Self {
        Self {
            mode: OutputMode::JsonObject,
            max_validation_retries: 0,
        }
    }

    pub fn json_schema(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            mode: OutputMode::JsonSchema {
                name: name.into(),
                schema,
                strict: true,
            },
            max_validation_retries: 0,
        }
    }

    pub fn lenient(mut self, retries: u32) -> Self {
        self.max_validation_retries = retries;
        if let OutputMode::JsonSchema { strict, .. } = &mut self.mode {
            *strict = false;
        }
        self
    }

    /// Produce the OpenAI `response_format` value when applicable.
    /// Returns `None` for [`OutputMode::FreeForm`].
    pub fn as_openai_response_format(&self) -> Option<serde_json::Value> {
        match &self.mode {
            OutputMode::FreeForm => None,
            OutputMode::JsonObject => Some(serde_json::json!({"type": "json_object"})),
            OutputMode::JsonSchema {
                name,
                schema,
                strict,
            } => Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                },
            })),
        }
    }

    /// Produce a system-prompt fragment that nudges Anthropic into
    /// JSON shape. Returns `None` for [`OutputMode::FreeForm`].
    /// `JsonSchema` includes the schema in pretty-printed form so the
    /// model can read it without one-line clutter.
    pub fn as_anthropic_system_fragment(&self) -> Option<String> {
        match &self.mode {
            OutputMode::FreeForm => None,
            OutputMode::JsonObject => {
                Some("Respond ONLY with a single JSON object. No prose, no markdown fences.".into())
            }
            OutputMode::JsonSchema { name, schema, .. } => {
                let pretty =
                    serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
                Some(format!(
                    "Respond ONLY with a single JSON object matching the `{name}` schema. \
                     No prose, no markdown fences.\n\nSchema:\n{pretty}",
                ))
            }
        }
    }

    /// Validate `text` against this config's mode. Returns `Ok(value)`
    /// for [`OutputMode::FreeForm`] (no constraint, returns the raw
    /// string wrapped), parses + checks shape for
    /// [`OutputMode::JsonObject`], and parses + structural
    /// schema-walks for [`OutputMode::JsonSchema`]. The walker
    /// enforces `type`, `properties`, and `required` — sufficient for
    /// post-stream validation parity with the host-side guard Claude
    /// Code applies; richer keywords belong in a dedicated validator
    /// crate.
    pub fn validate_text(&self, text: &str) -> Result<serde_json::Value, ValidationError> {
        match &self.mode {
            OutputMode::FreeForm => Ok(serde_json::Value::String(text.to_owned())),
            OutputMode::JsonObject => {
                let v = serde_json::from_str::<serde_json::Value>(text)
                    .map_err(|e| ValidationError::ParseError(e.to_string()))?;
                if !v.is_object() {
                    return Err(ValidationError::NotAnObject);
                }
                Ok(v)
            }
            OutputMode::JsonSchema { schema, .. } => {
                let v = serde_json::from_str::<serde_json::Value>(text)
                    .map_err(|e| ValidationError::ParseError(e.to_string()))?;
                walk_schema(&v, schema, "$")?;
                Ok(v)
            }
        }
    }
}

/// Errors surfaced by [`OutputConfig::validate_text`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// Body did not parse as valid JSON.
    ParseError(String),
    /// Mode required a JSON object but a non-object was provided.
    NotAnObject,
    /// Property value did not match its schema-declared type.
    TypeMismatch {
        path: String,
        expected: String,
        got: String,
    },
    /// Required property missing from object.
    MissingRequired { path: String, property: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(s) => write!(f, "json parse error: {s}"),
            Self::NotAnObject => write!(f, "expected JSON object"),
            Self::TypeMismatch {
                path,
                expected,
                got,
            } => {
                write!(f, "{path}: expected {expected}, got {got}")
            }
            Self::MissingRequired { path, property } => {
                write!(f, "{path}: missing required property `{property}`")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

fn walk_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> Result<(), ValidationError> {
    let Some(schema_obj) = schema.as_object() else {
        return Ok(());
    };
    if let Some(t) = schema_obj.get("type").and_then(|v| v.as_str()) {
        let matches = match t {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value.is_i64() || value.is_u64(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => true,
        };
        if !matches {
            return Err(ValidationError::TypeMismatch {
                path: path.to_string(),
                expected: t.to_string(),
                got: json_type_name(value).to_string(),
            });
        }
    }
    if let Some(props) = schema_obj.get("properties").and_then(|v| v.as_object()) {
        if let Some(obj) = value.as_object() {
            for (name, prop_schema) in props {
                if let Some(child) = obj.get(name) {
                    let child_path = format!("{path}.{name}");
                    walk_schema(child, prop_schema, &child_path)?;
                }
            }
        }
    }
    if let Some(req) = schema_obj.get("required").and_then(|v| v.as_array()) {
        if let Some(obj) = value.as_object() {
            for r in req {
                if let Some(name) = r.as_str() {
                    if !obj.contains_key(name) {
                        return Err(ValidationError::MissingRequired {
                            path: path.to_string(),
                            property: name.to_string(),
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_free_form() {
        let c = OutputConfig::default();
        assert_eq!(c.mode, OutputMode::FreeForm);
        assert!(c.as_openai_response_format().is_none());
        assert!(c.as_anthropic_system_fragment().is_none());
    }

    #[test]
    fn json_object_emits_minimal_openai_format() {
        let c = OutputConfig::json_object();
        let v = c.as_openai_response_format().unwrap();
        assert_eq!(v["type"], "json_object");
    }

    #[test]
    fn json_schema_emits_full_openai_format() {
        let schema = serde_json::json!({"type": "object"});
        let c = OutputConfig::json_schema("Foo", schema.clone());
        let v = c.as_openai_response_format().unwrap();
        assert_eq!(v["type"], "json_schema");
        assert_eq!(v["json_schema"]["name"], "Foo");
        assert_eq!(v["json_schema"]["schema"], schema);
        assert_eq!(v["json_schema"]["strict"], true);
    }

    #[test]
    fn lenient_clears_strict_and_sets_retries() {
        let c = OutputConfig::json_schema("F", serde_json::json!({})).lenient(3);
        assert_eq!(c.max_validation_retries, 3);
        match c.mode {
            OutputMode::JsonSchema { strict, .. } => assert!(!strict),
            _ => panic!("expected JsonSchema"),
        }
    }

    #[test]
    fn anthropic_fragment_is_clear_directive() {
        let c = OutputConfig::json_object();
        let frag = c.as_anthropic_system_fragment().unwrap();
        assert!(frag.contains("JSON"));
        assert!(frag.contains("ONLY"));
    }

    #[test]
    fn anthropic_schema_fragment_includes_schema_text() {
        let c = OutputConfig::json_schema("Foo", serde_json::json!({"type": "object"}));
        let frag = c.as_anthropic_system_fragment().unwrap();
        assert!(frag.contains("Foo"));
        assert!(frag.contains("type"));
    }

    #[test]
    fn validate_free_form_always_ok() {
        let c = OutputConfig::free_form();
        let v = c.validate_text("anything goes").unwrap();
        assert!(v.is_string());
    }

    #[test]
    fn validate_json_object_rejects_non_object() {
        let c = OutputConfig::json_object();
        assert!(matches!(
            c.validate_text("[1,2,3]").unwrap_err(),
            ValidationError::NotAnObject
        ));
        assert!(matches!(
            c.validate_text("not json").unwrap_err(),
            ValidationError::ParseError(_)
        ));
        c.validate_text(r#"{"x":1}"#).unwrap();
    }

    #[test]
    fn validate_json_schema_type_check() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"}
            },
            "required": ["age", "name"]
        });
        let c = OutputConfig::json_schema("Person", schema);
        // Happy path.
        c.validate_text(r#"{"age": 30, "name": "alice"}"#).unwrap();
        // Wrong type.
        let err = c
            .validate_text(r#"{"age": "thirty", "name": "alice"}"#)
            .unwrap_err();
        assert!(matches!(
            err,
            ValidationError::TypeMismatch { ref expected, .. } if expected == "integer"
        ));
        // Missing required field.
        let err = c.validate_text(r#"{"age": 30}"#).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::MissingRequired { ref property, .. } if property == "name"
        ));
    }
}
