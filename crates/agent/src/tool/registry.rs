use std::collections::HashMap;
use std::sync::Arc;

use super::Tool;

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
}
