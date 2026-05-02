//! Memory file discovery (Tier 1 / claude-code parity).
//!
//! Mirrors `memdir/memoryScan.ts`. Walks a directory non-recursively
//! (memory files live flat — no nested layout) and yields each `.md`
//! file's path + bytes. The caller pairs this with [`super::frontmatter`]
//! and [`super::loader`] to produce strongly-typed [`super::Memory`]
//! entries.
//!
//! `MEMORY.md` (the index) is excluded from the scan because its
//! content is a one-line-per-file pointer list, not a memory body.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannedFile {
    pub path: PathBuf,
    pub content: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("memory directory `{0}` does not exist")]
    Missing(PathBuf),
    #[error("memory directory `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Read every `.md` file under `dir` (non-recursive), excluding the
/// index file `MEMORY.md`. Returns paths in stable lexicographic
/// order so the loader output is deterministic across runs.
pub fn scan_dir(dir: &Path) -> Result<Vec<ScannedFile>, ScanError> {
    if !dir.exists() {
        return Err(ScanError::Missing(dir.to_path_buf()));
    }
    let entries = fs::read_dir(dir).map_err(|source| ScanError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut out: Vec<ScannedFile> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| ScanError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name() else {
            continue;
        };
        // Match on raw OS bytes so non-UTF-8 filenames (Unix paths
        // with arbitrary byte sequences) are still considered. The
        // ASCII suffix check is byte-safe because `.md` is pure
        // ASCII; non-ASCII bytes can sit anywhere else in the name.
        let bytes = name.as_encoded_bytes();
        if !ends_with_md_ascii_ci(bytes) {
            continue;
        }
        // Case-insensitive index exclusion — `MEMORY.md`, `memory.md`,
        // `Memory.md` are all the index file.
        if bytes.eq_ignore_ascii_case(b"MEMORY.md") {
            continue;
        }
        let content = fs::read_to_string(&path).map_err(|source| ScanError::Io {
            path: path.clone(),
            source,
        })?;
        out.push(ScannedFile { path, content });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Byte-level case-insensitive check that `name` ends with `.md`.
/// Lets non-UTF-8 filenames survive while keeping the suffix match
/// pure ASCII (which the `.md` extension always is).
fn ends_with_md_ascii_ci(name: &[u8]) -> bool {
    if name.len() < 3 {
        return false;
    }
    let suf = &name[name.len() - 3..];
    suf.eq_ignore_ascii_case(b".md")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn scan_skips_index_and_non_md() {
        let dir = tempdir().unwrap();
        let p = dir.path();
        fs::write(p.join("MEMORY.md"), "index").unwrap();
        fs::write(p.join("a.md"), "alpha").unwrap();
        fs::write(p.join("b.md"), "bravo").unwrap();
        fs::write(p.join("notes.txt"), "ignored").unwrap();
        let files = scan_dir(p).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.md", "b.md"]);
    }

    #[test]
    fn scan_missing_dir_errors() {
        match scan_dir(Path::new("/tmp/this-does-not-exist-12345")) {
            Err(ScanError::Missing(_)) => {}
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn scan_empty_dir_returns_empty_vec() {
        let dir = tempdir().unwrap();
        let files = scan_dir(dir.path()).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn scan_returns_files_in_lex_order() {
        let dir = tempdir().unwrap();
        let p = dir.path();
        fs::write(p.join("z.md"), "z").unwrap();
        fs::write(p.join("a.md"), "a").unwrap();
        fs::write(p.join("m.md"), "m").unwrap();
        let files = scan_dir(p).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.md", "m.md", "z.md"]);
    }

    #[test]
    fn scan_index_exclusion_is_case_insensitive() {
        let dir = tempdir().unwrap();
        let p = dir.path();
        fs::write(p.join("memory.md"), "lower").unwrap();
        fs::write(p.join("Memory.md"), "title").unwrap();
        fs::write(p.join("a.md"), "alpha").unwrap();
        let files = scan_dir(p).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.md"]);
    }

    /// Linux-only: macOS APFS/HFS+ rejects non-UTF-8 filenames at the
    /// kernel layer, so we can't actually write the test fixture
    /// there. The byte-level matcher in `scan_dir` is platform-agnostic
    /// and Linux CI proves the integration works end-to-end.
    #[cfg(all(unix, target_os = "linux"))]
    #[test]
    fn scan_includes_non_utf8_md_filenames() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let dir = tempdir().unwrap();
        let p = dir.path();
        let bad = p.join(OsStr::from_bytes(b"\xff\xfe.md"));
        fs::write(&bad, "ok").unwrap();
        fs::write(p.join("clean.md"), "also ok").unwrap();
        let files = scan_dir(p).unwrap();
        assert_eq!(files.len(), 2, "non-UTF-8 .md must not be dropped");
    }

    #[test]
    fn ends_with_md_ascii_ci_byte_matcher() {
        // Goes through the same call chain scan_dir uses: build an
        // OsString, hand it `as_encoded_bytes()`, then call the
        // helper. Verifies the ASCII suffix check survives non-UTF-8
        // prefix bytes — the property that motivates the refactor.
        use std::ffi::OsStr;
        fn check(name: &OsStr) -> bool {
            super::ends_with_md_ascii_ci(name.as_encoded_bytes())
        }
        assert!(check(OsStr::new("x.md")));
        assert!(check(OsStr::new("X.MD")));
        assert!(!check(OsStr::new("x.txt")));
        assert!(!check(OsStr::new("md")));
        assert!(!check(OsStr::new("")));
        // Non-UTF-8 byte prefix (Unix-only — OsStr::from_bytes lives
        // in std::os::unix::ffi::OsStrExt).
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(check(OsStr::from_bytes(b"\xff\xfe.md")));
            assert!(!check(OsStr::from_bytes(b"\xff\xfe.txt")));
        }
    }

    #[test]
    fn scan_skips_subdirectories() {
        let dir = tempdir().unwrap();
        let p = dir.path();
        fs::create_dir(p.join("sub")).unwrap();
        fs::write(p.join("sub").join("x.md"), "nested").unwrap();
        fs::write(p.join("top.md"), "top").unwrap();
        let files = scan_dir(p).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.ends_with("top.md"));
    }
}
