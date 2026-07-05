//! RTK-style output compression. Recognized simple dev commands can shrink
//! stdout before it reaches the model; piped, redirected, or unknown commands
//! return `None` and fall through to the Bash byte cap.

/// Result of compressing a command's stdout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compressed {
    pub text: String,
    /// One-line summary of what was elided, appended so the model can re-run raw.
    pub note: String,
}

#[derive(Debug, PartialEq, Eq)]
enum Kind {
    GitStatus,
    Ls,
    CargoTest,
}

const UNSAFE_SHELL: &[&str] = &["|", ">", "<", "&&", ";", "$(", "`", "\n"];

fn detect(command: &str) -> Option<Kind> {
    let cmd = command.trim();
    if UNSAFE_SHELL.iter().any(|m| cmd.contains(m)) {
        return None;
    }
    if cmd == "git status" || cmd.starts_with("git status ") {
        return Some(Kind::GitStatus);
    }
    if cmd == "ls" || cmd.starts_with("ls ") {
        return Some(Kind::Ls);
    }
    if cmd == "cargo test" || cmd.starts_with("cargo test ") {
        return Some(Kind::CargoTest);
    }
    None
}

fn is_porcelain_status(status: &str) -> bool {
    status.len() == 2
        && status
            .chars()
            .all(|c| matches!(c, ' ' | 'M' | 'A' | 'D' | 'R' | 'C' | 'U' | '?' | '!'))
}

#[allow(dead_code)]
pub(crate) fn dedup_consecutive(text: &str) -> (String, usize) {
    let mut out: Vec<String> = Vec::new();
    let mut removed = 0usize;
    let mut prev: Option<(&str, usize)> = None;
    let flush = |out: &mut Vec<String>, line: &str, n: usize| {
        if n > 1 {
            out.push(format!("{line} (x{n})"));
        } else {
            out.push(line.to_string());
        }
    };
    for line in text.lines() {
        match prev {
            Some((p, n)) if p == line => {
                removed += 1;
                prev = Some((p, n + 1));
            }
            Some((p, n)) => {
                flush(&mut out, p, n);
                prev = Some((line, 1));
            }
            None => prev = Some((line, 1)),
        }
    }
    if let Some((p, n)) = prev {
        flush(&mut out, p, n);
    }
    (out.join("\n"), removed)
}

pub(crate) fn truncate_middle(
    lines: &[&str],
    keep_head: usize,
    keep_tail: usize,
) -> (String, usize) {
    if lines.len() <= keep_head + keep_tail {
        return (lines.join("\n"), 0);
    }
    let elided = lines.len() - keep_head - keep_tail;
    let mut out = String::new();
    out.push_str(&lines[..keep_head].join("\n"));
    out.push_str(&format!("\n... {elided} lines elided ...\n"));
    out.push_str(&lines[lines.len() - keep_tail..].join("\n"));
    (out, elided)
}

/// Compress `stdout` for `command`, or return `None` for unknown/unsafe shapes.
pub fn compress_command(command: &str, stdout: &str) -> Option<Compressed> {
    match detect(command)? {
        Kind::GitStatus => Some(compress_git_status(stdout)),
        Kind::Ls => Some(compress_ls(stdout)),
        Kind::CargoTest => Some(compress_cargo_test(stdout)),
    }
}

fn compress_git_status(stdout: &str) -> Compressed {
    let mut modified = 0usize;
    let mut untracked = 0usize;
    let mut staged = 0usize;
    let mut files: Vec<String> = Vec::new();
    let mut in_untracked = false;

    for line in stdout.lines() {
        let t = line.trim();
        if t.starts_with("Untracked files:") {
            in_untracked = true;
            continue;
        }
        if t.starts_with("Changes to be committed:") {
            in_untracked = false;
            continue;
        }
        if t.starts_with("Changes not staged") {
            in_untracked = false;
            continue;
        }
        if let Some(rest) = t.strip_prefix("modified:") {
            modified += 1;
            files.push(format!("M {}", rest.trim()));
        } else if t.starts_with("new file:")
            || t.starts_with("deleted:")
            || t.starts_with("renamed:")
        {
            staged += 1;
            files.push(t.to_string());
        } else if in_untracked && !t.is_empty() && !t.starts_with('(') {
            untracked += 1;
            files.push(format!("? {t}"));
        } else if let Some((status, path)) = t.split_once(' ') {
            if status == "??" && !path.trim().is_empty() {
                untracked += 1;
                files.push(format!("? {}", path.trim()));
            } else if is_porcelain_status(status) && !path.trim().is_empty() {
                let mut chars = status.chars();
                let index = chars.next().unwrap_or(' ');
                let worktree = chars.next().unwrap_or(' ');
                if index != ' ' {
                    staged += 1;
                    files.push(format!("{index} {}", path.trim()));
                }
                if worktree != ' ' {
                    modified += 1;
                    files.push(format!("{worktree} {}", path.trim()));
                }
            }
        }
    }

    let file_refs: Vec<&str> = files.iter().map(String::as_str).collect();
    let (list, elided) = truncate_middle(&file_refs, 20, 0);
    let text =
        format!("git status: {modified} modified, {staged} staged, {untracked} untracked\n{list}");
    Compressed {
        text,
        note: format!(
            "compressed git status ({elided} filenames elided; re-run raw `git status` for full detail)"
        ),
    }
}

fn compress_ls(s: &str) -> Compressed {
    Compressed {
        text: s.to_string(),
        note: String::new(),
    }
}

fn compress_cargo_test(s: &str) -> Compressed {
    Compressed {
        text: s.to_string(),
        note: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_skips_piped_commands() {
        assert!(detect("git status | grep foo").is_none());
        assert!(detect("ls > out.txt").is_none());
        assert!(detect("cargo test && echo done").is_none());
    }

    #[test]
    fn detect_recognizes_simple_invocations() {
        assert!(matches!(detect("git status"), Some(Kind::GitStatus)));
        assert!(matches!(
            detect("git status --porcelain"),
            Some(Kind::GitStatus)
        ));
        assert!(matches!(detect("ls -la /tmp"), Some(Kind::Ls)));
        assert!(matches!(
            detect("cargo test --workspace"),
            Some(Kind::CargoTest)
        ));
        assert!(detect("echo hi").is_none());
    }

    #[test]
    fn dedup_consecutive_collapses_repeats() {
        let (out, removed) = dedup_consecutive("a\na\na\nb\n");
        assert_eq!(out, "a (x3)\nb");
        assert_eq!(removed, 2);
    }

    #[test]
    fn truncate_middle_keeps_ends() {
        let lines: Vec<&str> = (0..100).map(|_| "x").collect();
        let (out, elided) = truncate_middle(&lines, 3, 2);
        assert_eq!(elided, 95);
        assert!(out.starts_with("x\nx\nx\n"));
        assert!(out.trim_end().ends_with("x"));
    }

    #[test]
    fn git_status_summarizes_counts() {
        let raw = "On branch main\n\
Changes not staged for commit:\n\
\tmodified:   a.rs\n\
\tmodified:   b.rs\n\
Untracked files:\n\
\tc.rs\n\
\td.rs\n\
\te.rs\n";
        let out = compress_command("git status", raw).unwrap();
        assert!(out.text.contains("2 modified"), "{}", out.text);
        assert!(out.text.contains("3 untracked"), "{}", out.text);
        assert!(out.note.contains("git status"));
    }
}
