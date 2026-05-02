use std::collections::HashMap;
use std::sync::Arc;

use super::Tool;
use crate::provider::ToolDefinition;

fn schema_kind_str(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn truncate_for_log(v: &serde_json::Value, max_chars: usize) -> String {
    let s = serde_json::to_string(v).unwrap_or_default();
    if s.chars().count() <= max_chars {
        return s;
    }
    let head: String = s.chars().take(max_chars).collect();
    format!("{head}…(truncated)")
}

/// Registry of tools available to the QueryEngine for one turn.
///
/// Cheap to clone (each clone shares the same `Arc<dyn Tool>` instances
/// via the inner `HashMap`). New tools registered after `clone()` will
/// not appear in the original — the registry is conceptually a snapshot.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. If a tool with the same name was already
    /// registered, it is replaced and the previous value is returned.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        let name = tool.name().to_string();
        self.tools.insert(name, tool)
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.values().cloned().collect()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Materialize provider-neutral [`ToolDefinition`]s for every
    /// registered tool. Output is sorted by tool name so the rendered
    /// request body — and therefore [`crate::api::PromptCacheState`] —
    /// is stable across registrations made in different orders.
    ///
    /// **Defensive schema sanitization**: Anthropic and OpenAI both
    /// require `input_schema` to be a JSON object. If a host's `Tool`
    /// impl returns something else (null, bool, number, string, array)
    /// — typically a bug in a hand-written schema builder — the
    /// definition is materialized with a permissive `{"type":"object"}`
    /// fallback and a warning is logged. This keeps a malformed tool
    /// invocable instead of producing an opaque provider 400; if the
    /// host wants strict validation up-front, they should construct
    /// definitions via [`ToolDefinition::try_new`] themselves.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self
            .tools
            .values()
            .map(|t| {
                let schema = t.input_schema();
                let safe_schema = if schema.is_object() {
                    schema
                } else {
                    tracing::warn!(
                        target: "agent::tool::registry",
                        tool = t.name(),
                        kind = schema_kind_str(&schema),
                        sample = %truncate_for_log(&schema, 256),
                        "tool input_schema is not a JSON object; substituting {{\"type\":\"object\"}}"
                    );
                    serde_json::json!({"type": "object"})
                };
                ToolDefinition::new(
                    t.name().to_string(),
                    t.description().to_string(),
                    safe_schema,
                )
            })
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::error::AgentError;
    use crate::tool::ToolUseContext;

    #[derive(Debug)]
    struct NamedTool(&'static str);

    #[async_trait]
    impl Tool for NamedTool {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            Ok(serde_json::json!({"name": self.0}))
        }
    }

    #[test]
    fn register_and_get() {
        let mut r = ToolRegistry::new();
        assert!(r.is_empty());
        r.register(Arc::new(NamedTool("a")));
        r.register(Arc::new(NamedTool("b")));
        assert_eq!(r.len(), 2);
        assert!(r.get("a").is_some());
        assert!(r.get("b").is_some());
        assert!(r.get("c").is_none());
    }

    #[test]
    fn re_register_same_name_replaces() {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(NamedTool("a")));
        let prev = r.register(Arc::new(NamedTool("a")));
        assert!(prev.is_some());
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn list_and_names() {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(NamedTool("a")));
        r.register(Arc::new(NamedTool("b")));
        let names: Vec<&str> = r.names().collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
        assert_eq!(r.list().len(), 2);
    }

    #[test]
    fn definitions_are_sorted_by_name() {
        let mut r = ToolRegistry::new();
        // Insert in non-alphabetic order — definitions() must still
        // yield a stable, sorted slice so cache fingerprints don't
        // depend on registration order.
        r.register(Arc::new(NamedTool("zeta")));
        r.register(Arc::new(NamedTool("alpha")));
        r.register(Arc::new(NamedTool("mu")));
        let defs = r.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mu", "zeta"]);
    }

    #[derive(Debug)]
    struct BadSchemaTool;

    #[async_trait]
    impl Tool for BadSchemaTool {
        fn name(&self) -> &str {
            "bad"
        }
        fn description(&self) -> &str {
            "returns a non-object schema"
        }
        fn input_schema(&self) -> serde_json::Value {
            // Programmer bug: should be an object.
            serde_json::Value::Null
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            Ok(serde_json::Value::Null)
        }
    }

    #[test]
    fn definitions_substitute_permissive_schema_for_non_object() {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(BadSchemaTool));
        let defs = r.definitions();
        assert_eq!(defs.len(), 1);
        assert!(
            defs[0].input_schema.is_object(),
            "non-object schema should be substituted with permissive object"
        );
        assert_eq!(defs[0].input_schema["type"], "object");
    }

    #[test]
    fn definitions_carry_name_description_and_schema() {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(NamedTool("only")));
        let defs = r.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "only");
        assert_eq!(defs[0].description, "test tool");
        assert_eq!(defs[0].input_schema, serde_json::json!({}));
    }
}
