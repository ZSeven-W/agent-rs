//! Skill registry — runtime lookup + invocation.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::skill::{Skill, SkillError, SkillInvocation};

/// Shared registry of installed skills. Cheap to clone (Arc-wrapped).
#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    inner: Arc<Mutex<BTreeMap<String, Skill>>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, skill: Skill) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.insert(skill.name.clone(), skill);
    }

    pub fn remove(&self, name: &str) -> bool {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.remove(name).is_some()
    }

    pub fn get(&self, name: &str) -> Option<Skill> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.get(name).cloned()
    }

    /// Snapshot of all installed skills, sorted by name.
    pub fn list(&self) -> Vec<Skill> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.values().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Invoke a registered skill. Returns the rendered prompt
    /// bundle the host then sends to the model.
    pub fn invoke(
        &self,
        name: &str,
        params: &serde_json::Value,
    ) -> Result<SkillInvocation, SkillError> {
        let skill = self
            .get(name)
            .ok_or_else(|| SkillError::NotFound(name.to_string()))?;
        let prompt = skill.render(params)?;
        Ok(SkillInvocation {
            skill_name: skill.name,
            prompt,
            model: skill.model,
            allow_tools: skill.allow_tools,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &str, prompt: &str) -> Skill {
        Skill {
            name: name.into(),
            description: format!("desc-{name}"),
            prompt: prompt.into(),
            model: None,
            allow_tools: Default::default(),
            input_schema: serde_json::Value::Null,
        }
    }

    #[test]
    fn insert_get_remove_round_trip() {
        let r = SkillRegistry::new();
        r.insert(skill("a", "hi {x}"));
        assert!(r.get("a").is_some());
        assert!(r.remove("a"));
        assert!(r.get("a").is_none());
    }

    #[test]
    fn list_returns_sorted() {
        let r = SkillRegistry::new();
        r.insert(skill("z", "z"));
        r.insert(skill("a", "a"));
        r.insert(skill("m", "m"));
        let list = r.list();
        let names: Vec<_> = list.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "m", "z"]);
    }

    #[test]
    fn invoke_renders_prompt() {
        let r = SkillRegistry::new();
        r.insert(skill("greet", "Hello {name}!"));
        let inv = r
            .invoke("greet", &serde_json::json!({"name": "alice"}))
            .unwrap();
        assert_eq!(inv.skill_name, "greet");
        assert_eq!(inv.prompt, "Hello alice!");
    }

    #[test]
    fn invoke_unknown_skill_errors() {
        let r = SkillRegistry::new();
        assert!(matches!(
            r.invoke("ghost", &serde_json::Value::Null).unwrap_err(),
            SkillError::NotFound(_)
        ));
    }

    #[test]
    fn registry_propagates_render_errors() {
        let r = SkillRegistry::new();
        let mut s = skill("g", "Hello {n}");
        s.input_schema = serde_json::json!({
            "properties": {"n": {"type": "string"}},
            "required": ["n"]
        });
        r.insert(s);
        assert!(matches!(
            r.invoke("g", &serde_json::json!({})).unwrap_err(),
            SkillError::MissingParam { ref param, .. } if param == "n"
        ));
    }

    #[test]
    fn len_and_is_empty() {
        let r = SkillRegistry::new();
        assert!(r.is_empty());
        r.insert(skill("a", "a"));
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
    }
}
