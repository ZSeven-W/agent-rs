//! High-level memory directory loader (Tier 1 / claude-code parity).
//!
//! Pulls together [`super::scan`], [`super::frontmatter`],
//! [`super::memory_type`], and [`super::age`] to walk a memory
//! directory, parse every `.md` file's frontmatter + body, validate
//! against the 4-type taxonomy, and yield strongly-typed
//! [`super::Memory`] entries.
//!
//! Errors are split into two streams:
//!
//! - **Hard errors** (I/O, missing directory) bubble up via
//!   [`LoadError`] — caller should typically log + continue.
//! - **Per-file warnings** (frontmatter unterminated, unknown type,
//!   body validation hits) are accumulated into the
//!   [`LoadOutcome::warnings`] vector so the caller can surface them
//!   without losing the rest of the corpus.

use std::path::Path;
use std::time::SystemTime;

use super::age::file_age;
use super::frontmatter::{parse as parse_frontmatter, FrontmatterError};
use super::memory_type::{validate_body, MemoryType, ValidationWarning};
use super::scan::{scan_dir, ScanError, ScannedFile};
use super::Memory;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadWarning {
    pub path: std::path::PathBuf,
    pub kind: WarningKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningKind {
    /// Frontmatter present but malformed.
    Frontmatter(String),
    /// Frontmatter missing or `type:` field absent.
    MissingType,
    /// `type:` was present but didn't match the 4-type taxonomy.
    UnknownType(String),
    /// `name:` was missing — file is loaded with the file stem as a
    /// fallback.
    MissingName,
    /// `description:` was missing — file is loaded with empty desc.
    MissingDescription,
    /// Body validation produced advisory warnings.
    Body(Vec<ValidationWarning>),
    /// File metadata (mtime) couldn't be read; age defaults to zero.
    /// Surfaced so the host can disambiguate "ancient" from "unread"
    /// in retrieval logs.
    AgeUnknown(String),
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("scan: {0}")]
    Scan(#[from] ScanError),
    #[error("io: {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Clone)]
pub struct LoadOutcome {
    pub memories: Vec<Memory>,
    pub warnings: Vec<LoadWarning>,
}

/// Load every memory file in `dir`. Files with malformed frontmatter
/// or unknown types are dropped from `memories` and recorded in
/// `warnings`; the caller can decide whether to surface those.
///
/// `now` is the wall-clock used for age computation. Pass
/// `SystemTime::now()` in normal use; pass a fixed value in tests.
pub fn load_dir(dir: &Path, now: SystemTime) -> Result<LoadOutcome, LoadError> {
    let mut memories: Vec<Memory> = Vec::new();
    let mut warnings: Vec<LoadWarning> = Vec::new();
    let scanned = scan_dir(dir)?;

    for ScannedFile { path, content } in scanned {
        let fm = match parse_frontmatter(&content) {
            Ok(f) => f,
            Err(e) => {
                warnings.push(LoadWarning {
                    path: path.clone(),
                    kind: match e {
                        FrontmatterError::Unterminated => {
                            WarningKind::Frontmatter("frontmatter unterminated".into())
                        }
                        FrontmatterError::BadList { line, detail } => WarningKind::Frontmatter(
                            format!("malformed list at line {line}: {detail}"),
                        ),
                        FrontmatterError::DuplicateKey(k) => {
                            WarningKind::Frontmatter(format!("duplicate key `{k}`"))
                        }
                    },
                });
                continue;
            }
        };

        let kind = match fm.fields.get("type").and_then(|v| v.as_scalar()) {
            None => {
                warnings.push(LoadWarning {
                    path: path.clone(),
                    kind: WarningKind::MissingType,
                });
                continue;
            }
            Some(s) => match MemoryType::from_str_ci(s) {
                Some(t) => t,
                None => {
                    warnings.push(LoadWarning {
                        path: path.clone(),
                        kind: WarningKind::UnknownType(s.to_string()),
                    });
                    continue;
                }
            },
        };

        let name = match fm.fields.get("name").and_then(|v| v.as_scalar()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                warnings.push(LoadWarning {
                    path: path.clone(),
                    kind: WarningKind::MissingName,
                });
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unnamed")
                    .to_string()
            }
        };

        let description = match fm.fields.get("description").and_then(|v| v.as_scalar()) {
            Some(s) => s.to_string(),
            None => {
                warnings.push(LoadWarning {
                    path: path.clone(),
                    kind: WarningKind::MissingDescription,
                });
                String::new()
            }
        };

        let body_warnings = validate_body(&fm.body);
        if !body_warnings.is_empty() {
            warnings.push(LoadWarning {
                path: path.clone(),
                kind: WarningKind::Body(body_warnings),
            });
        }

        let age = match file_age(&path, now) {
            Ok(d) => d,
            Err(e) => {
                warnings.push(LoadWarning {
                    path: path.clone(),
                    kind: WarningKind::AgeUnknown(e.to_string()),
                });
                std::time::Duration::ZERO
            }
        };

        memories.push(Memory {
            kind,
            name,
            description,
            body: fm.body,
            path,
            age,
        });
    }

    Ok(LoadOutcome { memories, warnings })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use tempfile::tempdir;

    fn write(dir: &Path, name: &str, contents: &str) {
        fs::write(dir.join(name), contents).unwrap();
    }

    fn happy(typ: &str) -> String {
        format!("---\nname: Test\ndescription: A test memory\ntype: {typ}\n---\nBody content.")
    }

    #[test]
    fn happy_path_loads_user_memory() {
        let dir = tempdir().unwrap();
        write(dir.path(), "a.md", &happy("user"));
        let out = load_dir(dir.path(), SystemTime::now()).unwrap();
        assert_eq!(out.memories.len(), 1);
        assert_eq!(out.memories[0].kind, MemoryType::User);
        assert_eq!(out.memories[0].name, "Test");
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn unknown_type_is_warning_not_failure() {
        let dir = tempdir().unwrap();
        write(dir.path(), "a.md", &happy("nonsense"));
        let out = load_dir(dir.path(), SystemTime::now()).unwrap();
        assert!(out.memories.is_empty());
        assert_eq!(out.warnings.len(), 1);
        assert!(matches!(out.warnings[0].kind, WarningKind::UnknownType(_)));
    }

    #[test]
    fn missing_type_is_warning() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "a.md",
            "---\nname: x\ndescription: y\n---\nbody",
        );
        let out = load_dir(dir.path(), SystemTime::now()).unwrap();
        assert!(out.memories.is_empty());
        assert!(matches!(out.warnings[0].kind, WarningKind::MissingType));
    }

    #[test]
    fn missing_name_falls_back_to_file_stem() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "fallback.md",
            "---\ntype: user\ndescription: y\n---\nbody",
        );
        let out = load_dir(dir.path(), SystemTime::now()).unwrap();
        assert_eq!(out.memories.len(), 1);
        assert_eq!(out.memories[0].name, "fallback");
        assert!(matches!(out.warnings[0].kind, WarningKind::MissingName));
    }

    #[test]
    fn malformed_frontmatter_surfaces_warning() {
        let dir = tempdir().unwrap();
        write(dir.path(), "a.md", "---\nname: x\nstill in fm");
        let out = load_dir(dir.path(), SystemTime::now()).unwrap();
        assert!(out.memories.is_empty());
        assert!(matches!(out.warnings[0].kind, WarningKind::Frontmatter(_)));
    }

    #[test]
    fn body_validation_warning_does_not_block_load() {
        let dir = tempdir().unwrap();
        // 220 lines → "TooLong" warning, but should still load.
        let body: String = "line\n".repeat(220);
        let content = format!("---\nname: Test\ndescription: y\ntype: user\n---\n{body}");
        write(dir.path(), "a.md", &content);
        let out = load_dir(dir.path(), SystemTime::now()).unwrap();
        assert_eq!(out.memories.len(), 1);
        assert!(matches!(out.warnings[0].kind, WarningKind::Body(_)));
    }

    #[test]
    fn missing_dir_errors() {
        let r = load_dir(Path::new("/tmp/agent-memdir-nope-12345"), SystemTime::now());
        assert!(r.is_err());
    }

    #[test]
    fn deterministic_order_across_runs() {
        let dir = tempdir().unwrap();
        write(dir.path(), "z.md", &happy("user"));
        write(dir.path(), "a.md", &happy("feedback"));
        write(dir.path(), "m.md", &happy("project"));
        let out1 = load_dir(dir.path(), SystemTime::now()).unwrap();
        let out2 = load_dir(dir.path(), SystemTime::now()).unwrap();
        let names1: Vec<_> = out1.memories.iter().map(|m| m.path.clone()).collect();
        let names2: Vec<_> = out2.memories.iter().map(|m| m.path.clone()).collect();
        assert_eq!(names1, names2);
    }
}
