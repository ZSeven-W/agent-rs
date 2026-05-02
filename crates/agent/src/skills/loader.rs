//! Skill loader — walks a directory tree and parses each
//! `<name>/SKILL.md` into a [`super::Skill`].

use std::path::{Path, PathBuf};

use crate::memdir::frontmatter::{parse as parse_frontmatter, FrontmatterError};

use super::skill::Skill;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("skills dir `{0}` does not exist")]
    Missing(PathBuf),
    #[error("io: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadWarning {
    pub path: PathBuf,
    pub kind: WarningKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningKind {
    Frontmatter(String),
    /// Required `description` field missing from frontmatter.
    MissingDescription,
    /// Body (the prompt template) was empty.
    EmptyPrompt,
    /// `input_schema` was set but didn't parse as JSON.
    BadInputSchema(String),
}

#[derive(Debug, Clone)]
pub struct LoadOutcome {
    pub skills: Vec<Skill>,
    pub warnings: Vec<LoadWarning>,
}

/// Walk `dir` and parse every `<entry>/SKILL.md` into a Skill.
/// Subdirectory name becomes the skill name (lowercased + slugified).
pub fn load_dir(dir: &Path) -> Result<LoadOutcome, LoadError> {
    if !dir.exists() {
        return Err(LoadError::Missing(dir.to_path_buf()));
    }
    let entries = std::fs::read_dir(dir).map_err(|source| LoadError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut skills: Vec<Skill> = Vec::new();
    let mut warnings: Vec<LoadWarning> = Vec::new();
    let mut entry_paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| LoadError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let p = entry.path();
        if p.is_dir() {
            entry_paths.push(p);
        }
    }
    entry_paths.sort();

    for sub in entry_paths {
        let manifest = sub.join("SKILL.md");
        if !manifest.is_file() {
            continue;
        }
        let content = std::fs::read_to_string(&manifest).map_err(|source| LoadError::Io {
            path: manifest.clone(),
            source,
        })?;
        let fm = match parse_frontmatter(&content) {
            Ok(f) => f,
            Err(e) => {
                warnings.push(LoadWarning {
                    path: manifest.clone(),
                    kind: WarningKind::Frontmatter(match e {
                        FrontmatterError::Unterminated => "unterminated".into(),
                        FrontmatterError::BadList { line, detail } => {
                            format!("bad list at line {line}: {detail}")
                        }
                        FrontmatterError::DuplicateKey(k) => format!("duplicate key `{k}`"),
                    }),
                });
                continue;
            }
        };
        let name = sub
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed")
            .to_string();
        let description = match fm.fields.get("description").and_then(|v| v.as_scalar()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                warnings.push(LoadWarning {
                    path: manifest.clone(),
                    kind: WarningKind::MissingDescription,
                });
                String::new()
            }
        };
        let model = fm
            .fields
            .get("model")
            .and_then(|v| v.as_scalar())
            .map(|s| s.to_string());
        let allow_tools = fm
            .fields
            .get("allow_tools")
            .and_then(|v| v.as_list())
            .map(|items| items.iter().cloned().collect())
            .unwrap_or_default();
        let input_schema = match fm.fields.get("input_schema").and_then(|v| v.as_scalar()) {
            Some(s) if !s.is_empty() => match serde_json::from_str(s) {
                Ok(v) => v,
                Err(e) => {
                    warnings.push(LoadWarning {
                        path: manifest.clone(),
                        kind: WarningKind::BadInputSchema(e.to_string()),
                    });
                    serde_json::Value::Null
                }
            },
            _ => serde_json::Value::Null,
        };
        let prompt = fm.body.trim().to_string();
        if prompt.is_empty() {
            warnings.push(LoadWarning {
                path: manifest.clone(),
                kind: WarningKind::EmptyPrompt,
            });
        }
        skills.push(Skill {
            name,
            description,
            prompt,
            model,
            allow_tools,
            input_schema,
        });
    }
    Ok(LoadOutcome { skills, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn load_simple_skill() {
        let dir = tempdir().unwrap();
        let manifest = dir.path().join("greet/SKILL.md");
        write(
            &manifest,
            "---\ndescription: Greet someone\n---\nHello {name}!\n",
        );
        let out = load_dir(dir.path()).unwrap();
        assert_eq!(out.skills.len(), 1);
        let s = &out.skills[0];
        assert_eq!(s.name, "greet");
        assert_eq!(s.description, "Greet someone");
        assert_eq!(s.prompt, "Hello {name}!");
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn load_with_model_and_allow_tools() {
        let dir = tempdir().unwrap();
        write(
            &dir.path().join("review/SKILL.md"),
            "---\ndescription: Review code\nmodel: claude-opus-4\nallow_tools: [read, grep]\n---\nReview the code.",
        );
        let out = load_dir(dir.path()).unwrap();
        let s = &out.skills[0];
        assert_eq!(s.model.as_deref(), Some("claude-opus-4"));
        assert!(s.allow_tools.contains("read"));
        assert!(s.allow_tools.contains("grep"));
    }

    #[test]
    fn missing_description_warns() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("x/SKILL.md"), "---\n---\nbody");
        let out = load_dir(dir.path()).unwrap();
        assert_eq!(out.skills.len(), 1);
        assert!(out
            .warnings
            .iter()
            .any(|w| matches!(w.kind, WarningKind::MissingDescription)));
    }

    #[test]
    fn empty_prompt_warns() {
        let dir = tempdir().unwrap();
        write(
            &dir.path().join("x/SKILL.md"),
            "---\ndescription: x\n---\n   \n",
        );
        let out = load_dir(dir.path()).unwrap();
        assert!(out
            .warnings
            .iter()
            .any(|w| matches!(w.kind, WarningKind::EmptyPrompt)));
    }

    #[test]
    fn bad_input_schema_warns() {
        let dir = tempdir().unwrap();
        write(
            &dir.path().join("x/SKILL.md"),
            "---\ndescription: x\ninput_schema: not-json\n---\nbody",
        );
        let out = load_dir(dir.path()).unwrap();
        assert!(out
            .warnings
            .iter()
            .any(|w| matches!(w.kind, WarningKind::BadInputSchema(_))));
        // The skill still loads with a null schema.
        assert_eq!(out.skills.len(), 1);
    }

    #[test]
    fn missing_dir_errors() {
        match load_dir(Path::new("/tmp/agent-skills-nope-12345")) {
            Err(LoadError::Missing(_)) => {}
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn empty_dir_returns_empty_outcome() {
        let dir = tempdir().unwrap();
        let out = load_dir(dir.path()).unwrap();
        assert!(out.skills.is_empty());
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn skill_dir_without_manifest_skipped() {
        let dir = tempdir().unwrap();
        fs::create_dir(dir.path().join("incomplete")).unwrap();
        let out = load_dir(dir.path()).unwrap();
        assert!(out.skills.is_empty());
    }

    #[test]
    fn skills_loaded_in_lex_order() {
        let dir = tempdir().unwrap();
        write(
            &dir.path().join("z/SKILL.md"),
            "---\ndescription: z\n---\nz",
        );
        write(
            &dir.path().join("a/SKILL.md"),
            "---\ndescription: a\n---\na",
        );
        write(
            &dir.path().join("m/SKILL.md"),
            "---\ndescription: m\n---\nm",
        );
        let out = load_dir(dir.path()).unwrap();
        let names: Vec<_> = out.skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "m", "z"]);
    }
}
