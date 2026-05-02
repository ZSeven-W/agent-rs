//! Bootstrap + migrations (Tier 2 / claude-code parity).
//!
//! Mirrors `services/bootstrap/`. Runs once at host startup to bring
//! local state up to the current schema:
//!
//! 1. Resolve the data directory (memory + sessions + settings).
//! 2. Read the current `schema_version` marker (or 0 if absent).
//! 3. Apply migrations in order from `current → latest`.
//! 4. Write back the new `schema_version`.
//!
//! Migrations are pure-function-on-fs: each takes a directory path
//! and returns either Ok or an error; the runner applies them in
//! sequence and refuses to skip versions.

pub mod migrations;

pub use migrations::{Migration, MigrationError, MigrationKind};

use std::path::{Path, PathBuf};

/// Current schema version. Bumped when a new migration is added.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Path to the on-disk schema marker file inside `data_dir`.
pub fn schema_marker_path(data_dir: &Path) -> PathBuf {
    data_dir.join("SCHEMA_VERSION")
}

#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    #[error("io: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("schema marker malformed: {0}")]
    BadMarker(String),
    #[error("migration {version} failed: {source}")]
    Migration {
        version: u32,
        #[source]
        source: MigrationError,
    },
    #[error("data dir version {found} is newer than build {build} — refusing to downgrade")]
    NewerThanBuild { found: u32, build: u32 },
}

/// Read the schema-version marker, defaulting to 0 if the file
/// doesn't exist (fresh install).
pub fn read_schema_version(data_dir: &Path) -> Result<u32, BootstrapError> {
    let p = schema_marker_path(data_dir);
    match std::fs::read_to_string(&p) {
        Ok(s) => s
            .trim()
            .parse::<u32>()
            .map_err(|e| BootstrapError::BadMarker(e.to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(BootstrapError::Io { path: p, source: e }),
    }
}

/// Write the schema-version marker. Caller usually does this only
/// from [`run`] after successful migration.
pub fn write_schema_version(data_dir: &Path, version: u32) -> Result<(), BootstrapError> {
    let p = schema_marker_path(data_dir);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| BootstrapError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    std::fs::write(&p, format!("{version}\n"))
        .map_err(|e| BootstrapError::Io { path: p, source: e })
}

/// Run all pending migrations from `current+1 → CURRENT_SCHEMA_VERSION`.
/// Idempotent: a fully-up-to-date dir does nothing.
pub fn run(data_dir: &Path) -> Result<RunReport, BootstrapError> {
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir).map_err(|e| BootstrapError::Io {
            path: data_dir.to_path_buf(),
            source: e,
        })?;
    }
    let current = read_schema_version(data_dir)?;
    if current > CURRENT_SCHEMA_VERSION {
        return Err(BootstrapError::NewerThanBuild {
            found: current,
            build: CURRENT_SCHEMA_VERSION,
        });
    }
    let migrations = migrations::all();
    let mut applied: Vec<u32> = Vec::new();
    for m in migrations.iter().filter(|m| m.version() > current) {
        m.apply(data_dir)
            .map_err(|source| BootstrapError::Migration {
                version: m.version(),
                source,
            })?;
        applied.push(m.version());
    }
    if let Some(&last) = applied.last() {
        write_schema_version(data_dir, last)?;
    } else if current == 0 {
        // Fresh dir with no migrations defined: still write version 0
        // so the marker exists for future runs.
        write_schema_version(data_dir, 0)?;
    }
    Ok(RunReport {
        previous_version: current,
        new_version: applied.last().copied().unwrap_or(current),
        applied,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunReport {
    pub previous_version: u32,
    pub new_version: u32,
    pub applied: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn schema_marker_path_is_inside_dir() {
        let p = schema_marker_path(Path::new("/tmp/x"));
        assert_eq!(p, PathBuf::from("/tmp/x/SCHEMA_VERSION"));
    }

    #[test]
    fn read_missing_file_returns_zero() {
        let dir = tempdir().unwrap();
        assert_eq!(read_schema_version(dir.path()).unwrap(), 0);
    }

    #[test]
    fn write_then_read_roundtrip() {
        let dir = tempdir().unwrap();
        write_schema_version(dir.path(), 7).unwrap();
        assert_eq!(read_schema_version(dir.path()).unwrap(), 7);
    }

    #[test]
    fn malformed_marker_errors() {
        let dir = tempdir().unwrap();
        std::fs::write(schema_marker_path(dir.path()), "not-a-number").unwrap();
        match read_schema_version(dir.path()) {
            Err(BootstrapError::BadMarker(_)) => {}
            other => panic!("expected BadMarker, got {other:?}"),
        }
    }

    #[test]
    fn run_creates_dir_when_missing() {
        let parent = tempdir().unwrap();
        let nested = parent.path().join("agent-data");
        assert!(!nested.exists());
        run(&nested).unwrap();
        assert!(nested.exists());
    }

    #[test]
    fn run_on_fresh_dir_applies_all_migrations() {
        let dir = tempdir().unwrap();
        let report = run(dir.path()).unwrap();
        assert_eq!(report.previous_version, 0);
        assert_eq!(report.new_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(report.applied.len(), CURRENT_SCHEMA_VERSION as usize);
        // Marker reflects the version.
        assert_eq!(
            read_schema_version(dir.path()).unwrap(),
            CURRENT_SCHEMA_VERSION
        );
    }

    #[test]
    fn run_is_idempotent() {
        let dir = tempdir().unwrap();
        run(dir.path()).unwrap();
        let second = run(dir.path()).unwrap();
        assert!(second.applied.is_empty());
        assert_eq!(second.previous_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(second.new_version, CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn newer_than_build_errors() {
        let dir = tempdir().unwrap();
        write_schema_version(dir.path(), CURRENT_SCHEMA_VERSION + 5).unwrap();
        match run(dir.path()) {
            Err(BootstrapError::NewerThanBuild { found, build }) => {
                assert_eq!(found, CURRENT_SCHEMA_VERSION + 5);
                assert_eq!(build, CURRENT_SCHEMA_VERSION);
            }
            other => panic!("expected NewerThanBuild, got {other:?}"),
        }
    }
}
