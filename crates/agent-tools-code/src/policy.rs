//! Path safety policy shared by every fs/shell/search tool.
//!
//! The host configures a [`WorkspacePolicy`] once per session and
//! passes it (in an `Arc`) to each tool. The policy answers two
//! questions:
//!
//! 1. **Where can the tool operate?** Paths are resolved relative
//!    to the policy's `cwd`, then canonicalized, then checked
//!    against a list of allowed roots. Anything outside is
//!    rejected with [`PolicyError::OutsideWorkspace`].
//! 2. **How big a payload is allowed?** Read / write tools cap
//!    file sizes at `max_file_size_bytes`; exceeding that returns
//!    [`PolicyError::TooLarge`].
//!
//! Symlinks: by default they're resolved as part of canonicalization
//! and the resolved target must also be inside an allowed root. A
//! host can disable that resolution by setting
//! `follow_symlinks = false`, in which case the policy refuses any
//! path that contains a symlink along its prefix (defensive — keeps
//! a malicious symlink from punching out of the workspace).

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PolicyError {
    #[error("path '{path}' is outside the configured workspace roots")]
    OutsideWorkspace { path: String },
    #[error("path '{path}' contains a symlink and follow_symlinks is disabled")]
    SymlinkRejected { path: String },
    #[error("path '{path}' could not be canonicalized: {source}")]
    Canonicalize {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("payload too large: {size_bytes} bytes (limit {limit_bytes})")]
    TooLarge { size_bytes: u64, limit_bytes: u64 },
    #[error("path '{path}' has no parent (root path?)")]
    NoParent { path: String },
}

/// Workspace policy. Cheap to clone via `Arc`. Construct once per
/// session and pass to every tool.
#[derive(Debug, Clone)]
pub struct WorkspacePolicy {
    /// Working directory that bare relative paths resolve against.
    /// Must be canonical; [`Self::new`] canonicalizes for you.
    pub cwd: PathBuf,
    /// Allowed parent directories. A resolved path is accepted iff
    /// its canonical form starts with one of these. Empty means
    /// "anything under `cwd`".
    pub allowed_roots: Vec<PathBuf>,
    /// Hard cap on read / write payload size. Defaults to 8 MiB —
    /// matches the [`agent::file_cache::FileStateCache`] entry cap
    /// so tool reads don't blow past the cache's per-entry budget.
    pub max_file_size_bytes: u64,
    /// If `false`, paths containing a symlink anywhere in their
    /// prefix are rejected. If `true` (default), the symlink target
    /// is canonicalized and checked against `allowed_roots` like
    /// any other path.
    pub follow_symlinks: bool,
}

impl WorkspacePolicy {
    /// Construct a policy rooted at `cwd`. The path is canonicalized
    /// up-front so subsequent containment checks are O(1) prefix
    /// matches. `cwd` is added to `allowed_roots` automatically.
    ///
    /// Returns an error if `cwd` doesn't exist or can't be resolved.
    pub fn new(cwd: impl AsRef<Path>) -> Result<Self, PolicyError> {
        let raw = cwd.as_ref();
        let canonical = canonicalize(raw)?;
        Ok(Self {
            allowed_roots: vec![canonical.clone()],
            cwd: canonical,
            max_file_size_bytes: 8 * 1024 * 1024,
            follow_symlinks: true,
        })
    }

    /// Convenience: wrap in `Arc` for sharing across tool registrations.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Add an extra allowed root. Useful when the host wants a tool
    /// to read from a sibling directory (e.g., `~/.cache/agent`).
    pub fn with_allowed_root(mut self, root: impl AsRef<Path>) -> Result<Self, PolicyError> {
        self.allowed_roots.push(canonicalize(root.as_ref())?);
        Ok(self)
    }

    pub fn with_max_file_size(mut self, n: u64) -> Self {
        self.max_file_size_bytes = n;
        self
    }

    pub fn with_follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }

    /// Resolve a host-supplied path against the policy. Returns the
    /// canonical absolute form on success, or a `PolicyError` if
    /// the path violates any constraint.
    ///
    /// `must_exist`: when `true`, the path must already exist on
    /// disk. Use `false` for write/mkdir/move-target paths whose
    /// parent must exist but whose leaf may not.
    pub fn resolve(&self, raw: &str, must_exist: bool) -> Result<PathBuf, PolicyError> {
        let path = if Path::new(raw).is_absolute() {
            PathBuf::from(raw)
        } else {
            self.cwd.join(raw)
        };
        if must_exist {
            let canonical = canonicalize(&path)?;
            self.check_inside(&canonical)?;
            self.check_symlink_policy(&path)?;
            return Ok(canonical);
        }
        // Path may not exist — and for `mkdir -p`-style use the
        // parent may not exist either. Walk up to the first
        // ancestor that *does* exist, canonicalize it (so symlinks
        // along that prefix get resolved and containment-checked),
        // then reattach the missing tail lexically.
        let (existing_ancestor, missing_tail) =
            walk_up_to_existing(&path).ok_or_else(|| PolicyError::Canonicalize {
                path: raw.to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no ancestor of the path exists; cannot resolve",
                ),
            })?;
        // The reattached tail must contain only `Normal` components.
        // `Path::file_name` shouldn't ever return Some("..") per
        // std's contract, but defense in depth: reject anything but
        // a plain segment so `..` / `.` / prefix / root-dir can
        // never sneak through to a recursive `mkdir` and create a
        // path that looks inside via `starts_with` but resolves
        // outside.
        for c in missing_tail.components() {
            if !matches!(c, Component::Normal(_)) {
                return Err(PolicyError::OutsideWorkspace {
                    path: raw.to_string(),
                });
            }
        }
        let ancestor_canonical = canonicalize(&existing_ancestor)?;
        // Avoid `join("")` — `PathBuf::join` with an empty path
        // appends a trailing separator, which makes downstream
        // `tokio::fs::write` interpret the target as a directory.
        let resolved = if missing_tail.as_os_str().is_empty() {
            ancestor_canonical
        } else {
            ancestor_canonical.join(&missing_tail)
        };
        self.check_inside(&resolved)?;
        self.check_symlink_policy(&path)?;
        Ok(resolved)
    }

    /// Enforce the file-size cap. Call before reading bytes into
    /// memory.
    pub fn check_size(&self, size_bytes: u64) -> Result<(), PolicyError> {
        if size_bytes > self.max_file_size_bytes {
            return Err(PolicyError::TooLarge {
                size_bytes,
                limit_bytes: self.max_file_size_bytes,
            });
        }
        Ok(())
    }

    fn check_inside(&self, canonical: &Path) -> Result<(), PolicyError> {
        if self
            .allowed_roots
            .iter()
            .any(|root| canonical.starts_with(root))
        {
            return Ok(());
        }
        Err(PolicyError::OutsideWorkspace {
            path: canonical.display().to_string(),
        })
    }

    fn check_symlink_policy(&self, raw: &Path) -> Result<(), PolicyError> {
        if self.follow_symlinks {
            return Ok(());
        }
        // Walk every prefix component; if any is a symlink, reject.
        let mut acc = PathBuf::new();
        for c in raw.components() {
            match c {
                Component::Prefix(p) => acc.push(p.as_os_str()),
                Component::RootDir => acc.push(std::path::MAIN_SEPARATOR.to_string()),
                Component::Normal(seg) => {
                    acc.push(seg);
                    if let Ok(meta) = std::fs::symlink_metadata(&acc) {
                        if meta.file_type().is_symlink() {
                            return Err(PolicyError::SymlinkRejected {
                                path: raw.display().to_string(),
                            });
                        }
                    }
                }
                Component::CurDir | Component::ParentDir => acc.push(c.as_os_str()),
            }
        }
        Ok(())
    }
}

/// Walk up `p`'s ancestors until we find one that exists on disk.
/// Returns `Some((existing_ancestor, missing_tail))` where
/// `missing_tail` is the relative remainder, or `None` if no
/// ancestor exists (which would be a misconfigured root).
///
/// `missing_tail` is built by prepending each popped leaf so the
/// returned order matches the original path. We avoid
/// `Path::join("")` (which appends a trailing slash) by tracking
/// the leaves in reverse order and joining once at the end.
fn walk_up_to_existing(p: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut existing = p.to_path_buf();
    let mut leaves: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if existing.exists() {
            let mut tail = PathBuf::new();
            // Leaves were collected outermost-first (we pushed when
            // popping); join them in original path order.
            for name in leaves.iter().rev() {
                tail.push(name);
            }
            return Some((existing, tail));
        }
        let leaf = existing.file_name().map(|n| n.to_os_string());
        if !existing.pop() {
            return None;
        }
        if let Some(name) = leaf {
            leaves.push(name);
        }
    }
}

fn canonicalize(p: &Path) -> Result<PathBuf, PolicyError> {
    std::fs::canonicalize(p).map_err(|e| PolicyError::Canonicalize {
        path: p.display().to_string(),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_policy() -> (TempDir, Arc<WorkspacePolicy>) {
        let dir = TempDir::new().unwrap();
        let policy = WorkspacePolicy::new(dir.path()).unwrap().into_arc();
        (dir, policy)
    }

    #[test]
    fn new_canonicalizes_cwd_and_seeds_allowed_root() {
        let dir = TempDir::new().unwrap();
        let policy = WorkspacePolicy::new(dir.path()).unwrap();
        assert!(policy.cwd.is_absolute());
        assert!(policy.allowed_roots.iter().any(|r| r == &policy.cwd));
        assert_eq!(policy.max_file_size_bytes, 8 * 1024 * 1024);
        assert!(policy.follow_symlinks);
    }

    #[test]
    fn resolve_existing_inside_workspace() {
        let (dir, policy) = temp_policy();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, b"x").unwrap();
        let resolved = policy.resolve("a.txt", true).unwrap();
        assert!(resolved.ends_with("a.txt"));
    }

    #[test]
    fn resolve_outside_workspace_rejected() {
        let (_dir, policy) = temp_policy();
        let outside = TempDir::new().unwrap();
        let path = outside.path().join("x.txt");
        std::fs::write(&path, b"x").unwrap();
        let err = policy
            .resolve(path.to_str().unwrap(), true)
            .expect_err("should reject");
        assert!(matches!(err, PolicyError::OutsideWorkspace { .. }));
    }

    #[test]
    fn resolve_nonexistent_inside_for_write() {
        let (dir, policy) = temp_policy();
        let target = dir.path().join("new_file.txt");
        let resolved = policy.resolve(target.to_str().unwrap(), false).unwrap();
        assert!(resolved.ends_with("new_file.txt"));
    }

    #[test]
    fn resolve_must_exist_errors_on_missing() {
        let (_dir, policy) = temp_policy();
        let err = policy.resolve("nope.txt", true).expect_err("missing");
        assert!(matches!(err, PolicyError::Canonicalize { .. }));
    }

    #[test]
    fn check_size_enforces_cap() {
        let (_dir, policy) = temp_policy();
        assert!(policy.check_size(1024).is_ok());
        assert!(policy.check_size(policy.max_file_size_bytes).is_ok());
        let err = policy
            .check_size(policy.max_file_size_bytes + 1)
            .expect_err("too large");
        assert!(matches!(err, PolicyError::TooLarge { .. }));
    }

    #[test]
    fn with_allowed_root_extends_acceptance() {
        let (_dir, policy) = temp_policy();
        let extra = TempDir::new().unwrap();
        let f = extra.path().join("y.txt");
        std::fs::write(&f, b"y").unwrap();
        let extended = (*policy)
            .clone()
            .with_allowed_root(extra.path())
            .unwrap()
            .into_arc();
        let resolved = extended.resolve(f.to_str().unwrap(), true).unwrap();
        assert!(resolved.ends_with("y.txt"));
    }

    #[test]
    fn resolve_non_strict_normal_components_only() {
        // Sanity: a clean missing tail with only normal components
        // resolves to ancestor + tail.
        let (dir, policy) = temp_policy();
        let target = dir.path().join("a/b/c.txt");
        let resolved = policy.resolve(target.to_str().unwrap(), false).unwrap();
        assert!(resolved.ends_with("a/b/c.txt"));
    }

    #[cfg(unix)]
    #[test]
    fn follow_symlinks_false_rejects_symlink_path() {
        let (dir, _policy) = temp_policy();
        let target = dir.path().join("real.txt");
        std::fs::write(&target, b"r").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_follow_symlinks(false)
            .into_arc();
        let err = policy
            .resolve(link.to_str().unwrap(), true)
            .expect_err("symlink rejected");
        assert!(matches!(err, PolicyError::SymlinkRejected { .. }));
    }
}
