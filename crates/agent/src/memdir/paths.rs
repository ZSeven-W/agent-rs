//! Memory directory path resolution (Tier 1 / claude-code parity).
//!
//! Mirrors `memdir/paths.ts`. Determines where to look for memory
//! files when the host doesn't pass an explicit path. Resolution
//! order (first hit wins):
//!
//! 1. `AGENT_MEMORY_DIR` env override (test hook + power user).
//! 2. `<XDG_CONFIG_HOME>/agent/memory/` if `XDG_CONFIG_HOME` is set.
//! 3. `<APPDATA>/agent/memory/` if `APPDATA` is set. Canonically
//!    Windows-only but checked unconditionally so the same priority
//!    is observable on Linux/macOS hosts that opt in by setting it.
//! 4. `<HOME>/Library/Application Support/agent/memory/` on macOS.
//! 5. `<HOME>/.config/agent/memory/` on Unix.
//!
//! Per-project memory lives under
//! `<dir>/<project_id>/memory/` where `project_id` is a slug derived
//! from the working directory (`/Users/x/Workspace/Foo` →
//! `-Users-x-Workspace-Foo`, mirroring claude-code's slugifier).

use std::path::{Path, PathBuf};

/// Resolve the global memory root for the current host. Errors only
/// when no plausible directory exists (no HOME, no APPDATA, no env).
pub fn global_memory_root() -> Result<PathBuf, PathResolveError> {
    if let Ok(p) = std::env::var("AGENT_MEMORY_DIR") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(p).join("agent").join("memory"));
    }
    // APPDATA is canonically Windows-only but checked unconditionally
    // so the documented priority order is observable on every host.
    // A user (or test) that sets APPDATA on Linux/macOS gets the
    // documented behavior rather than silently-ignored env state.
    if let Ok(p) = std::env::var("APPDATA") {
        return Ok(PathBuf::from(p).join("agent").join("memory"));
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return Ok(PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("agent")
                .join("memory"));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(home)
            .join(".config")
            .join("agent")
            .join("memory"));
    }
    Err(PathResolveError::NoHome)
}

/// Slugify an absolute project path into a stable directory name.
///
/// Mirror of claude-code's slugifier: `/`, `\`, and `:` all collapse
/// to `-`, repeated separators collapse, and the trailing `-` is
/// stripped. Leading separators turn into a leading `-`.
///
/// **Slug stability vs. injectivity**: this function is deterministic
/// — same input gives same output — but it is NOT injective. Distinct
/// paths can collide (e.g., on Windows `C:\x` and `-C\x` both
/// slugify to `-C-x`). This matches claude-code's reference behavior;
/// hosts that need a guaranteed-unique per-project ID should append
/// their own hash of the original path.
pub fn project_slug(working_dir: &Path) -> String {
    let s = working_dir.to_string_lossy();
    let mut out = String::with_capacity(s.len());
    let mut last_dash = false;
    for ch in s.chars() {
        if matches!(ch, '/' | '\\' | ':') {
            if !last_dash {
                out.push('-');
                last_dash = true;
            }
        } else {
            out.push(ch);
            last_dash = false;
        }
    }
    out.trim_end_matches('-').to_string()
}

/// Compose the per-project memory directory: `<root>/<slug>/memory/`.
pub fn project_memory_dir(working_dir: &Path) -> Result<PathBuf, PathResolveError> {
    let root = global_memory_root()?;
    Ok(root.join(project_slug(working_dir)).join("memory"))
}

/// Like [`project_slug`] but guaranteed-unique per distinct input:
/// appends a 64-bit FNV-1a hash of the path's raw `OsStr` bytes as an
/// `.<hex>` suffix.
///
/// Crucially the hash operates on `Path::as_os_str().as_encoded_bytes()`
/// rather than `to_string_lossy()`, so two distinct non-UTF-8 Unix
/// paths that would otherwise collide under lossy decoding produce
/// different hashes and therefore different slugs. Hosts that ship
/// multi-tenant memory stores and can't tolerate slug collisions
/// should use this helper.
pub fn project_slug_unique(working_dir: &Path) -> String {
    const FNV_64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_64_PRIME: u64 = 0x100_0000_01b3;
    let raw = working_dir.as_os_str().as_encoded_bytes();
    let mut h = FNV_64_OFFSET;
    for &b in raw {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_64_PRIME);
    }
    format!("{}.{:016x}", project_slug(working_dir), h)
}

/// The root MEMORY.md index file inside a memory directory.
pub fn index_file(memory_dir: &Path) -> PathBuf {
    memory_dir.join("MEMORY.md")
}

#[derive(Debug, thiserror::Error)]
pub enum PathResolveError {
    #[error(
        "no $HOME / $APPDATA / $XDG_CONFIG_HOME / $AGENT_MEMORY_DIR — cannot resolve memory root"
    )]
    NoHome,
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Tests in this module mutate process-wide env vars
    /// (AGENT_MEMORY_DIR, XDG_CONFIG_HOME, APPDATA, HOME) and would
    /// race under cargo's default parallel runner. Serialize via a
    /// shared mutex so each env-touching test gets exclusive access.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn slug_replaces_separators() {
        let p = Path::new("/Users/x/Workspace/Foo");
        assert_eq!(project_slug(p), "-Users-x-Workspace-Foo");
    }

    #[test]
    fn slug_collapses_repeated_separators() {
        let p = Path::new("///foo//bar");
        assert_eq!(project_slug(p), "-foo-bar");
    }

    #[cfg(unix)]
    #[test]
    fn slug_unique_distinguishes_non_utf8_paths() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        // Two different invalid-UTF-8 byte sequences. Both decode to
        // the SAME replacement-character string under to_string_lossy,
        // so a slug derived from lossy bytes would collide. The
        // os-string-encoded slug must NOT.
        let a = Path::new(OsStr::from_bytes(b"/data/\xff\xfe"));
        let b = Path::new(OsStr::from_bytes(b"/data/\xff\xfd"));
        assert_eq!(a.to_string_lossy(), b.to_string_lossy());
        assert_ne!(project_slug_unique(a), project_slug_unique(b));
    }

    #[test]
    fn slug_unique_distinguishes_collisions() {
        // /Users/x and :Users:x both collapse to "-Users-x" under
        // the simple slug. `project_slug_unique` should still
        // distinguish them via the hash suffix.
        let slash = project_slug_unique(Path::new("/Users/x"));
        let colon = project_slug_unique(Path::new(":Users:x"));
        assert_ne!(slash, colon);
        assert!(slash.starts_with("-Users-x."));
        assert!(colon.starts_with("-Users-x."));
    }

    #[test]
    fn slug_path_with_colon_separator_collides() {
        // Documented limitation: claude-code-parity slug is NOT
        // injective. `/Users/x` and `:Users:x` both collapse to the
        // same slug after `/`, `\`, and `:` all map to `-`. Pin this
        // expectation so future "fix" attempts have to make a
        // deliberate choice + update the docstring.
        let slash = project_slug(Path::new("/Users/x"));
        let colon = project_slug(Path::new(":Users:x"));
        assert_eq!(slash, colon);
    }

    #[test]
    fn slug_strips_trailing_dash() {
        let p = Path::new("/foo/bar/");
        assert_eq!(project_slug(p), "-foo-bar");
    }

    #[test]
    fn env_override_wins() {
        let _guard = env_lock();
        std::env::set_var("AGENT_MEMORY_DIR", "/tmp/agent-memdir-test");
        let r = global_memory_root().unwrap();
        assert_eq!(r, PathBuf::from("/tmp/agent-memdir-test"));
        std::env::remove_var("AGENT_MEMORY_DIR");
    }

    #[test]
    fn xdg_config_home_used_when_no_override() {
        let _guard = env_lock();
        std::env::remove_var("AGENT_MEMORY_DIR");
        std::env::remove_var("APPDATA");
        std::env::set_var("XDG_CONFIG_HOME", "/tmp/xdg-test");
        let r = global_memory_root().unwrap();
        assert_eq!(r, PathBuf::from("/tmp/xdg-test/agent/memory"));
        std::env::remove_var("XDG_CONFIG_HOME");
    }

    #[test]
    fn project_dir_composes() {
        let _guard = env_lock();
        std::env::set_var("AGENT_MEMORY_DIR", "/tmp/foo");
        let p = project_memory_dir(Path::new("/Users/x/proj")).unwrap();
        assert_eq!(p, PathBuf::from("/tmp/foo/-Users-x-proj/memory"));
        std::env::remove_var("AGENT_MEMORY_DIR");
    }

    #[test]
    fn index_file_is_memory_md() {
        let p = Path::new("/tmp/some/memdir");
        assert_eq!(index_file(p), p.join("MEMORY.md"));
    }
}
