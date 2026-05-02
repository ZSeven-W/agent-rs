//! Migration definitions (Tier 2 / claude-code parity).
//!
//! Each migration is a pure function-on-fs that the bootstrap runner
//! applies in order. New migrations append to [`all`] with the next
//! version number; bumping [`super::CURRENT_SCHEMA_VERSION`] is what
//! actually activates them.

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    #[error("io: {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid state during migration: {0}")]
    Invalid(String),
}

/// Stable identifier for what a migration does. Used by the runner
/// for telemetry — no behaviour-bearing semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MigrationKind {
    /// Initial layout: create memory/ + sessions/ subdirectories.
    InitialLayout,
}

/// One migration. Implementations live as small `Fn`-like structs.
pub trait Migration: std::fmt::Debug + Send + Sync {
    fn version(&self) -> u32;
    fn kind(&self) -> MigrationKind;
    fn apply(&self, data_dir: &Path) -> Result<(), MigrationError>;
}

/// All known migrations, in version order. Must be a strictly-
/// increasing sequence with no gaps.
pub fn all() -> Vec<Box<dyn Migration>> {
    vec![Box::new(M1InitialLayout)]
}

#[derive(Debug)]
struct M1InitialLayout;

impl Migration for M1InitialLayout {
    fn version(&self) -> u32 {
        1
    }
    fn kind(&self) -> MigrationKind {
        MigrationKind::InitialLayout
    }
    fn apply(&self, data_dir: &Path) -> Result<(), MigrationError> {
        for sub in ["memory", "sessions", "logs"] {
            let p = data_dir.join(sub);
            std::fs::create_dir_all(&p).map_err(|e| MigrationError::Io { path: p, source: e })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn versions_are_strictly_increasing_with_no_gaps() {
        let migrations = all();
        for (i, m) in migrations.iter().enumerate() {
            assert_eq!(
                m.version(),
                (i + 1) as u32,
                "migration #{i} version should be {}",
                i + 1
            );
        }
    }

    #[test]
    fn initial_layout_creates_subdirs() {
        let dir = tempdir().unwrap();
        M1InitialLayout.apply(dir.path()).unwrap();
        for sub in ["memory", "sessions", "logs"] {
            assert!(dir.path().join(sub).is_dir());
        }
    }

    #[test]
    fn initial_layout_is_idempotent() {
        let dir = tempdir().unwrap();
        M1InitialLayout.apply(dir.path()).unwrap();
        // Second apply must succeed (subdirs already exist).
        M1InitialLayout.apply(dir.path()).unwrap();
    }

    #[test]
    fn migration_kind_round_trip() {
        assert_eq!(M1InitialLayout.kind(), MigrationKind::InitialLayout);
    }

    #[test]
    fn apply_into_nonexistent_parent_creates_it() {
        let parent = tempdir().unwrap();
        let nested = parent.path().join("missing").join("data");
        // create_dir_all in M1 should walk parents.
        M1InitialLayout.apply(&nested).unwrap();
        assert!(nested.join("memory").is_dir());
    }
}
