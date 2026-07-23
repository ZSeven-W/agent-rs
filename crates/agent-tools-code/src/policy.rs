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

use agent::abort::AbortController;
use async_trait::async_trait;

/// Performs the actual filesystem MUTATION on behalf of the fs tools, once a
/// path has already been resolved + policy-checked. The default
/// [`DirectFsSink`] writes in-process; a host can supply an implementation that
/// performs the mutating syscall inside an OS sandbox, so file writes become
/// kernel-enforced — not just shell commands. This mirrors Codex's
/// `ExecutorFileSystem` + `SandboxedFileSystem` split: the tool computes the
/// new bytes in-process (no read-before-write staleness problem), then hands
/// the final mutation to the sink.
#[async_trait]
pub trait FsSink: std::fmt::Debug + Send + Sync {
    /// Write (create or truncate) `path` with `bytes`.
    async fn write_file(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()>;
    /// Create `path` as a directory (`recursive` ⇒ create missing parents).
    async fn create_dir(&self, path: &Path, recursive: bool) -> std::io::Result<()>;
    /// Rename / move `from` to `to`.
    async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()>;
    /// Remove `path`. `is_dir` selects dir vs file removal; `recursive` removes
    /// a non-empty directory tree.
    async fn remove(&self, path: &Path, recursive: bool, is_dir: bool) -> std::io::Result<()>;

    /// Tracked mutation variants let hosts keep detached OS workers visible to
    /// the root turn after the calling tool future has been cancelled. Sinks
    /// without detached work inherit the direct implementations above.
    async fn write_file_tracked(
        &self,
        path: &Path,
        bytes: &[u8],
        _abort: &AbortController,
    ) -> std::io::Result<()> {
        self.write_file(path, bytes).await
    }
    async fn create_dir_tracked(
        &self,
        path: &Path,
        recursive: bool,
        _abort: &AbortController,
    ) -> std::io::Result<()> {
        self.create_dir(path, recursive).await
    }
    async fn rename_tracked(
        &self,
        from: &Path,
        to: &Path,
        _abort: &AbortController,
    ) -> std::io::Result<()> {
        self.rename(from, to).await
    }
    async fn remove_tracked(
        &self,
        path: &Path,
        recursive: bool,
        is_dir: bool,
        _abort: &AbortController,
    ) -> std::io::Result<()> {
        self.remove(path, recursive, is_dir).await
    }
}

/// Default sink: perform the mutation in-process via `tokio::fs`. Identical to
/// the behavior the fs tools had before the sink seam existed.
#[derive(Debug, Default)]
pub struct DirectFsSink;

#[async_trait]
impl FsSink for DirectFsSink {
    async fn write_file(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        tokio::fs::write(path, bytes).await
    }
    async fn create_dir(&self, path: &Path, recursive: bool) -> std::io::Result<()> {
        if recursive {
            tokio::fs::create_dir_all(path).await
        } else {
            tokio::fs::create_dir(path).await
        }
    }
    async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        tokio::fs::rename(from, to).await
    }
    async fn remove(&self, path: &Path, recursive: bool, is_dir: bool) -> std::io::Result<()> {
        if is_dir {
            if recursive {
                tokio::fs::remove_dir_all(path).await
            } else {
                tokio::fs::remove_dir(path).await
            }
        } else {
            tokio::fs::remove_file(path).await
        }
    }

    async fn write_file_tracked(
        &self,
        path: &Path,
        bytes: &[u8],
        abort: &AbortController,
    ) -> std::io::Result<()> {
        crate::fs_supervision::write_file(path, bytes, abort).await
    }
    async fn create_dir_tracked(
        &self,
        path: &Path,
        recursive: bool,
        abort: &AbortController,
    ) -> std::io::Result<()> {
        crate::fs_supervision::create_dir(path, recursive, abort).await
    }
    async fn rename_tracked(
        &self,
        from: &Path,
        to: &Path,
        abort: &AbortController,
    ) -> std::io::Result<()> {
        crate::fs_supervision::rename(from, to, abort).await
    }
    async fn remove_tracked(
        &self,
        path: &Path,
        recursive: bool,
        is_dir: bool,
        abort: &AbortController,
    ) -> std::io::Result<()> {
        crate::fs_supervision::remove(path, recursive, is_dir, abort).await
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PolicyError {
    #[error("path '{path}' is outside the configured workspace roots")]
    OutsideWorkspace { path: String },
    #[error("path '{path}' is a protected, read-only location")]
    Protected { path: String },
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
    /// Subtrees that are read-only even when inside an allowed root: a
    /// resolved WRITE target under any of these is rejected with
    /// [`PolicyError::Protected`]. Reads are unaffected. The host sets these to
    /// protect metadata like `.git` / `.zode` from being rewritten by file
    /// tools — the policy-layer twin of the OS sandbox's protected carveouts.
    /// Stored as absolute paths (not necessarily existing, so first-time
    /// creation is blocked too); matched by canonical-prefix.
    pub denied_subpaths: Vec<PathBuf>,
    /// Subtrees that are hidden from READS (resolve_read rejects them). Separate
    /// from `denied_subpaths` (which only blocks writes) so a host can opt into
    /// hiding credential dirs (`~/.ssh`, …) from the file tools WITHOUT making
    /// the write-protected `.git`/`.zode` unreadable. Empty by default → reads
    /// stay unconfined. Matched by canonical/lexical prefix, like writes.
    pub read_denied_subpaths: Vec<PathBuf>,
    /// Hard cap on read / write payload size. Defaults to 8 MiB —
    /// matches the [`agent::file_cache::FileStateCache`] entry cap
    /// so tool reads don't blow past the cache's per-entry budget.
    pub max_file_size_bytes: u64,
    /// If `false`, paths containing a symlink anywhere in their
    /// prefix are rejected. If `true` (default), the symlink target
    /// is canonicalized and checked against `allowed_roots` like
    /// any other path.
    pub follow_symlinks: bool,
    /// How the fs tools perform their mutating syscalls. Defaults to
    /// [`DirectFsSink`] (in-process); a host can swap in a sandboxed sink so
    /// writes are kernel-enforced. Cheap to clone (Arc).
    pub fs_sink: Arc<dyn FsSink>,
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
            denied_subpaths: Vec::new(),
            read_denied_subpaths: Vec::new(),
            cwd: canonical,
            max_file_size_bytes: 8 * 1024 * 1024,
            follow_symlinks: true,
            fs_sink: Arc::new(DirectFsSink),
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

    /// Mark a subtree read-only (writes under it are rejected). The lexical
    /// absolute path is stored WITHOUT requiring it to exist, so a not-yet-
    /// created `.zode` is still protected from first-time creation. If the path
    /// DOES exist, its canonical form is stored too: a write target is matched
    /// after canonicalization, so a denied dir that is itself a symlink would
    /// otherwise resolve to a different prefix and slip through.
    pub fn with_denied_subpath(mut self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        };
        if let Ok(canonical) = canonicalize(&abs) {
            if canonical != abs && !self.denied_subpaths.contains(&canonical) {
                self.denied_subpaths.push(canonical);
            }
        }
        self.denied_subpaths.push(abs);
        self
    }

    /// Hide a subtree from READS (`resolve_read` rejects it). Stores the lexical
    /// absolute path AND, if it exists, its canonical form — same dual-match as
    /// [`Self::with_denied_subpath`], so a symlinked credential dir can't slip
    /// through. Used by a host that opts into credential-read hiding.
    pub fn with_read_denied_subpath(mut self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        };
        if let Ok(canonical) = canonicalize(&abs) {
            if canonical != abs && !self.read_denied_subpaths.contains(&canonical) {
                self.read_denied_subpaths.push(canonical);
            }
        }
        self.read_denied_subpaths.push(abs);
        self
    }

    pub fn with_max_file_size(mut self, n: u64) -> Self {
        self.max_file_size_bytes = n;
        self
    }

    /// Swap the filesystem-mutation sink (e.g. a sandboxed writer). The fs
    /// tools route their final write/create/rename/remove through this.
    pub fn with_fs_sink(mut self, sink: Arc<dyn FsSink>) -> Self {
        self.fs_sink = sink;
        self
    }

    /// Mutation helpers the fs tools call instead of `tokio::fs` directly, so a
    /// host-supplied sink (sandboxed or not) handles the actual syscall.
    pub async fn write_file(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        self.fs_sink.write_file(path, bytes).await
    }
    pub async fn create_dir(&self, path: &Path, recursive: bool) -> std::io::Result<()> {
        self.fs_sink.create_dir(path, recursive).await
    }
    pub async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        self.fs_sink.rename(from, to).await
    }
    pub async fn remove(&self, path: &Path, recursive: bool, is_dir: bool) -> std::io::Result<()> {
        self.fs_sink.remove(path, recursive, is_dir).await
    }

    pub async fn write_file_tracked(
        &self,
        path: &Path,
        bytes: &[u8],
        abort: &AbortController,
    ) -> std::io::Result<()> {
        self.fs_sink.write_file_tracked(path, bytes, abort).await
    }
    pub async fn create_dir_tracked(
        &self,
        path: &Path,
        recursive: bool,
        abort: &AbortController,
    ) -> std::io::Result<()> {
        self.fs_sink
            .create_dir_tracked(path, recursive, abort)
            .await
    }
    pub async fn rename_tracked(
        &self,
        from: &Path,
        to: &Path,
        abort: &AbortController,
    ) -> std::io::Result<()> {
        self.fs_sink.rename_tracked(from, to, abort).await
    }
    pub async fn remove_tracked(
        &self,
        path: &Path,
        recursive: bool,
        is_dir: bool,
        abort: &AbortController,
    ) -> std::io::Result<()> {
        self.fs_sink
            .remove_tracked(path, recursive, is_dir, abort)
            .await
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
            // Check the deny-list against BOTH the canonical target and the
            // lexical path: a `.zode` symlink created mid-session would make the
            // canonical target escape the denied prefix, but the lexical
            // `cwd/.zode/...` still matches.
            self.check_not_denied(&canonical)?;
            self.check_not_denied(&path)?;
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
        self.check_not_denied(&resolved)?;
        // Also block the lexical path (see the must_exist branch above).
        self.check_not_denied(&path)?;
        self.check_symlink_policy(&path)?;
        Ok(resolved)
    }

    /// Resolve a path for READ access, WITHOUT confining it to
    /// `allowed_roots`. Reads are non-destructive, and a coding agent routinely
    /// needs to read files outside the working directory — the project source
    /// when launched elsewhere, sibling projects, `~/.zode`, configs. Only
    /// existence + canonicalization apply (the OS still enforces real file
    /// permissions). Write confinement stays in [`resolve`].
    pub fn resolve_read(&self, raw: &str) -> Result<PathBuf, PolicyError> {
        let path = if Path::new(raw).is_absolute() {
            PathBuf::from(raw)
        } else {
            self.cwd.join(raw)
        };
        let canonical = canonicalize(&path)?;
        // Credential-read hiding (opt-in): reject reads under a read-denied
        // subtree. Check BOTH the canonical and lexical path (a symlinked dir
        // created mid-session would canonicalize elsewhere; the lexical prefix
        // still matches), mirroring the write deny-list.
        if !self.read_denied_subpaths.is_empty() {
            self.check_not_read_denied(&canonical)?;
            self.check_not_read_denied(&path)?;
        }
        Ok(canonical)
    }

    fn check_not_read_denied(&self, resolved: &Path) -> Result<(), PolicyError> {
        if self.is_read_denied(resolved) {
            return Err(PolicyError::Protected {
                path: resolved.display().to_string(),
            });
        }
        Ok(())
    }

    /// Whether `path` is under a read-denied (strict-read credential) subtree.
    /// Public so directory-WALKING tools (`Grep`/`Glob`) can skip entries that a
    /// per-file `resolve_read` would reject — they only `resolve_read` the search
    /// root, then iterate, so without this a grep under an allowed root that is
    /// an ancestor of `~/.ssh` would still surface credentials.
    pub fn is_read_denied(&self, path: &Path) -> bool {
        self.read_denied_subpaths
            .iter()
            .any(|denied| path.starts_with(denied))
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

    /// Reject a resolved WRITE target that falls under a protected subtree.
    /// `resolved` is already canonical, so symlinks/`..` that point INTO a
    /// protected dir are caught here too.
    fn check_not_denied(&self, resolved: &Path) -> Result<(), PolicyError> {
        if self
            .denied_subpaths
            .iter()
            .any(|denied| resolved.starts_with(denied))
        {
            return Err(PolicyError::Protected {
                path: resolved.display().to_string(),
            });
        }
        Ok(())
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
    fn resolve_read_allows_outside_workspace() {
        // Reads are not confined to the workspace root: a write `resolve`
        // rejects an outside path, but `resolve_read` returns it.
        let (_dir, policy) = temp_policy();
        let outside = TempDir::new().unwrap();
        let path = outside.path().join("x.txt");
        std::fs::write(&path, b"x").unwrap();
        let raw = path.to_str().unwrap();
        assert!(policy.resolve(raw, true).is_err());
        let resolved = policy
            .resolve_read(raw)
            .expect("read resolves outside root");
        assert!(resolved.ends_with("x.txt"));
    }

    #[test]
    fn denied_subpath_blocks_writes_but_not_reads() {
        let dir = TempDir::new().unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_denied_subpath(".git");
        // A write under the protected subtree is rejected — even for a path
        // that does not exist yet (first-time creation blocked).
        let err = policy
            .resolve(".git/config", false)
            .expect_err("write into .git must be rejected");
        assert!(matches!(err, PolicyError::Protected { .. }), "{err:?}");
        // Writes elsewhere in the workspace still succeed.
        assert!(policy.resolve("src/main.rs", false).is_ok());
        // Reads of the protected subtree are still allowed.
        let f = dir.path().join(".git");
        std::fs::create_dir_all(&f).unwrap();
        std::fs::write(f.join("HEAD"), b"ref: x").unwrap();
        assert!(
            policy.resolve_read(".git/HEAD").is_ok(),
            "reads not confined"
        );
    }

    #[test]
    fn read_denied_subpath_blocks_reads_only() {
        let dir = TempDir::new().unwrap();
        let secrets = dir.path().join("secrets");
        std::fs::create_dir_all(&secrets).unwrap();
        std::fs::write(secrets.join("key"), b"shh").unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_read_denied_subpath(&secrets);
        // Reads under the read-denied subtree are rejected…
        let err = policy
            .resolve_read("secrets/key")
            .expect_err("read of a hidden dir must be rejected");
        assert!(matches!(err, PolicyError::Protected { .. }), "{err:?}");
        // …reads elsewhere are unaffected…
        std::fs::write(dir.path().join("ok.txt"), b"x").unwrap();
        assert!(policy.resolve_read("ok.txt").is_ok());
        // …and the read-deny does NOT also block writes into that dir (it is a
        // read-only hide, separate from the write deny-list).
        assert!(
            policy.resolve("secrets/new", false).is_ok(),
            "read-deny must not change write policy"
        );
    }

    #[test]
    fn denied_subpath_catches_symlink_into_protected_dir() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".zode")).unwrap();
        // A symlink that points into the protected dir is resolved and caught.
        let link = dir.path().join("sneaky");
        #[cfg(unix)]
        std::os::unix::fs::symlink(dir.path().join(".zode"), &link).unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_denied_subpath(".zode");
        #[cfg(unix)]
        {
            let err = policy
                .resolve("sneaky/state.json", false)
                .expect_err("symlink into .zode must be rejected");
            assert!(matches!(err, PolicyError::Protected { .. }), "{err:?}");
        }
    }

    #[test]
    #[cfg(unix)]
    fn denied_subpath_that_is_itself_a_symlink_is_still_blocked() {
        // `.zode` is a symlink to a real dir; a write through it canonicalizes
        // to the target, which the lexical deny path wouldn't catch — so the
        // canonical form must also be denied.
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real-zode");
        std::fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, dir.path().join(".zode")).unwrap();
        let policy = WorkspacePolicy::new(dir.path())
            .unwrap()
            .with_allowed_root(&real)
            .unwrap()
            .with_denied_subpath(".zode");
        let err = policy
            .resolve(".zode/state.json", false)
            .expect_err("write through the symlinked .zode must be rejected");
        assert!(matches!(err, PolicyError::Protected { .. }), "{err:?}");
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
