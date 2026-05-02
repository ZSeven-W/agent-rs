//! Skill data model (Tier 2 / claude-code parity).
//!
//! A `Skill` is a strongly-typed callable prompt:
//!
//! - [`Skill::name`] is its registry key.
//! - [`Skill::render`] takes user-supplied params and produces the
//!   prompt text the host then sends to the model.
//! - [`Skill::input_schema`] is a JSON schema for the params; the
//!   registry validates inputs against it before render.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("skill `{name}`: missing required param `{param}`")]
    MissingParam { name: String, param: String },
    #[error("skill `{name}`: unknown param `{param}`")]
    UnknownParam { name: String, param: String },
    #[error("skill `{name}`: param `{param}`: expected {expected}, got {got}")]
    TypeMismatch {
        name: String,
        param: String,
        expected: String,
        got: String,
    },
    #[error("skill `{0}` not registered")]
    NotFound(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    /// One-sentence description — surfaced to the model + UI list.
    pub description: String,
    /// Prompt template. Uses `{name}` placeholders; replaced with
    /// stringified param values during [`Skill::render`].
    pub prompt: String,
    /// Optional model override. When `None`, host picks the default.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional tool allowlist. When non-empty, the host should
    /// scope the agent's [`crate::tool::ToolRegistry`] to this set
    /// during the skill's invocation.
    #[serde(default)]
    pub allow_tools: BTreeSet<String>,
    /// Input schema (typically draft 2020-12 JSON Schema). Empty
    /// object = no params required.
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

impl Skill {
    /// Render the prompt with `params` substituted into placeholders.
    /// Returns [`SkillError::MissingParam`] when a placeholder lacks
    /// a value AND the schema lists the param as required.
    pub fn render(&self, params: &serde_json::Value) -> Result<String, SkillError> {
        validate_params(self, params)?;
        let mut out = self.prompt.clone();
        if let Some(map) = params.as_object() {
            for (k, v) in map {
                let placeholder = format!("{{{k}}}");
                let s = scalar_to_string(v);
                out = out.replace(&placeholder, &s);
            }
        }
        Ok(out)
    }
}

/// Bundle the host hands to the LLM caller after rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillInvocation {
    pub skill_name: String,
    pub prompt: String,
    pub model: Option<String>,
    pub allow_tools: BTreeSet<String>,
}

fn validate_params(skill: &Skill, params: &serde_json::Value) -> Result<(), SkillError> {
    let schema = match skill.input_schema.as_object() {
        Some(s) => s,
        None => return Ok(()),
    };
    let params_obj = params.as_object();
    // Check required.
    if let Some(req) = schema.get("required").and_then(|v| v.as_array()) {
        for r in req {
            if let Some(name) = r.as_str() {
                let present = params_obj.map(|o| o.contains_key(name)).unwrap_or(false);
                if !present {
                    return Err(SkillError::MissingParam {
                        name: skill.name.clone(),
                        param: name.to_string(),
                    });
                }
            }
        }
    }
    // Check unknown + types.
    if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
        if let Some(obj) = params_obj {
            for (k, v) in obj {
                let prop = match props.get(k) {
                    Some(p) => p,
                    None => {
                        return Err(SkillError::UnknownParam {
                            name: skill.name.clone(),
                            param: k.clone(),
                        });
                    }
                };
                if let Some(t) = prop.get("type").and_then(|v| v.as_str()) {
                    let got = json_type_name(v);
                    let matches = match t {
                        "string" => v.is_string(),
                        "number" => v.is_number(),
                        "integer" => v.is_i64() || v.is_u64(),
                        "boolean" => v.is_boolean(),
                        "array" => v.is_array(),
                        "object" => v.is_object(),
                        "null" => v.is_null(),
                        _ => true,
                    };
                    if !matches {
                        return Err(SkillError::TypeMismatch {
                            name: skill.name.clone(),
                            param: k.clone(),
                            expected: t.to_string(),
                            got: got.to_string(),
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

fn scalar_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(prompt: &str, schema: serde_json::Value) -> Skill {
        Skill {
            name: "test".into(),
            description: "x".into(),
            prompt: prompt.into(),
            model: None,
            allow_tools: BTreeSet::new(),
            input_schema: schema,
        }
    }

    #[test]
    fn render_substitutes_string_param() {
        let s = skill("Hello {name}!", serde_json::json!({}));
        let out = s.render(&serde_json::json!({"name": "alice"})).unwrap();
        assert_eq!(out, "Hello alice!");
    }

    #[test]
    fn render_substitutes_number() {
        let s = skill("Age = {age}", serde_json::json!({}));
        let out = s.render(&serde_json::json!({"age": 30})).unwrap();
        assert_eq!(out, "Age = 30");
    }

    #[test]
    fn render_missing_required_param_errors() {
        let s = skill(
            "Hello {name}!",
            serde_json::json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
        );
        assert!(matches!(
            s.render(&serde_json::json!({})).unwrap_err(),
            SkillError::MissingParam { ref param, .. } if param == "name"
        ));
    }

    #[test]
    fn render_unknown_param_errors() {
        let s = skill(
            "x",
            serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "string"}}
            }),
        );
        assert!(matches!(
            s.render(&serde_json::json!({"b": "x"})).unwrap_err(),
            SkillError::UnknownParam { ref param, .. } if param == "b"
        ));
    }

    #[test]
    fn render_type_mismatch_errors() {
        let s = skill(
            "x",
            serde_json::json!({
                "properties": {"age": {"type": "integer"}}
            }),
        );
        assert!(matches!(
            s.render(&serde_json::json!({"age": "thirty"})).unwrap_err(),
            SkillError::TypeMismatch { ref param, .. } if param == "age"
        ));
    }

    #[test]
    fn render_no_schema_accepts_anything() {
        let s = skill("hi {x}", serde_json::Value::Null);
        let out = s.render(&serde_json::json!({"x": 1})).unwrap();
        assert_eq!(out, "hi 1");
    }

    #[test]
    fn skill_serde_roundtrip() {
        let s = Skill {
            name: "t".into(),
            description: "d".into(),
            prompt: "p {x}".into(),
            model: Some("claude".into()),
            allow_tools: ["read", "write"].iter().map(|s| s.to_string()).collect(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: Skill = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn render_replaces_repeated_placeholders() {
        let s = skill("{x}-{x}-{x}", serde_json::Value::Null);
        let out = s.render(&serde_json::json!({"x": "a"})).unwrap();
        assert_eq!(out, "a-a-a");
    }

    #[test]
    fn render_null_param_substitutes_empty_string() {
        let s = skill("[{x}]", serde_json::Value::Null);
        let out = s.render(&serde_json::json!({"x": null})).unwrap();
        assert_eq!(out, "[]");
    }
}
