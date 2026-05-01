//! Post-compaction file restoration (claude-code parity, Tier 1).
//!
//! After [`super::summarize::compact_conversation`] replaces the bulky
//! conversation with a summary, the assistant has lost direct
//! visibility into recently-touched files. `post_cleanup` builds a
//! single User message that re-attaches the top-N most-relevant files
//! as text content blocks so the next turn can reference them
//! verbatim without an extra Read tool round-trip.
//!
//! Mirror of `services/compact/postCompactCleanup.ts`.

use std::path::PathBuf;

use crate::message::{ContentBlock, Header, Message};

/// Constants from Claude Code:
/// `POST_COMPACT_MAX_FILES_TO_RESTORE`,
/// `POST_COMPACT_TOKEN_BUDGET`,
/// `POST_COMPACT_MAX_TOKENS_PER_FILE`,
/// `POST_COMPACT_MAX_TOKENS_PER_SKILL`,
/// `POST_COMPACT_SKILLS_TOKEN_BUDGET`.
#[derive(Debug, Clone, Copy)]
pub struct PostCompactConfig {
    pub max_files_to_restore: usize,
    pub token_budget: u32,
    pub max_tokens_per_file: u32,
    pub max_tokens_per_skill: u32,
    pub skills_token_budget: u32,
}

impl Default for PostCompactConfig {
    fn default() -> Self {
        Self {
            max_files_to_restore: 5,
            token_budget: 50_000,
            max_tokens_per_file: 5_000,
            max_tokens_per_skill: 5_000,
            skills_token_budget: 25_000,
        }
    }
}

/// One file the caller wants attached. The caller is responsible for
/// reading the bytes (the post-cleanup module does not touch the
/// filesystem) — this keeps storage / sandboxing decisions in the
/// product layer.
#[derive(Debug, Clone)]
pub struct FileAttachment {
    pub path: PathBuf,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct PostCompactResult {
    /// One synthetic User message carrying the file attachments as
    /// text content blocks. None if no files were restored.
    pub restored_message: Option<Message>,
    /// Paths of every file that ended up in `restored_message`.
    pub restored_paths: Vec<PathBuf>,
    /// Tokens consumed by the restored content. Approximate.
    pub tokens_used: u32,
    /// Files that were dropped because they didn't fit (over budget /
    /// over per-file cap / past the file-count cap).
    pub skipped_paths: Vec<PathBuf>,
}

/// Build the post-compaction restoration message.
///
/// Selection rules:
/// 1. Sort `files` by relevance score (caller-supplied via the
///    surrounding ranking — the input order IS the priority order).
/// 2. Truncate each file's content to `config.max_tokens_per_file`
///    estimated tokens.
/// 3. Accumulate into the synthetic message until either
///    `config.max_files_to_restore` or `config.token_budget` is hit.
/// 4. Return the message, or `None` if no files survived selection.
pub fn build_post_compact_message(
    files: Vec<FileAttachment>,
    config: &PostCompactConfig,
) -> PostCompactResult {
    let mut restored_paths = Vec::new();
    let mut skipped_paths = Vec::new();
    let mut content_blocks = Vec::new();
    let mut tokens_used: u32 = 0;

    for (idx, file) in files.into_iter().enumerate() {
        if idx >= config.max_files_to_restore {
            skipped_paths.push(file.path);
            continue;
        }

        let truncated = truncate_to_tokens(&file.content, config.max_tokens_per_file);
        let cost = super::estimate_text_tokens(&truncated);
        if tokens_used.saturating_add(cost) > config.token_budget {
            skipped_paths.push(file.path);
            continue;
        }

        let header_text = format!("--- {} ---\n", file.path.display());
        let block_text = format!("{header_text}{truncated}");
        content_blocks.push(ContentBlock::Text { text: block_text });
        tokens_used = tokens_used.saturating_add(cost).saturating_add(8); // +overhead per attachment header
        restored_paths.push(file.path);
    }

    let restored_message = if content_blocks.is_empty() {
        None
    } else {
        let preamble = ContentBlock::Text {
            text: format!(
                "[Post-compaction file restoration: {} file(s) attached below.]",
                content_blocks.len()
            ),
        };
        let mut blocks = Vec::with_capacity(content_blocks.len() + 1);
        blocks.push(preamble);
        blocks.extend(content_blocks);
        Some(Message::User {
            header: Header::new(),
            content: blocks,
        })
    };

    PostCompactResult {
        restored_message,
        restored_paths,
        tokens_used,
        skipped_paths,
    }
}

/// Truncate `text` so it fits roughly within `max_tokens` of our
/// heuristic budget. Cuts at a UTF-8 char boundary; appends an
/// elision suffix when truncated.
fn truncate_to_tokens(text: &str, max_tokens: u32) -> String {
    let approx_max_chars = max_tokens.saturating_mul(4) as usize; // 4 chars ≈ 1 token
    if text.len() <= approx_max_chars {
        return text.to_string();
    }
    let mut end = approx_max_chars.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + 64);
    out.push_str(&text[..end]);
    out.push_str("\n... [truncated for post-compact restoration] ...");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, content: &str) -> FileAttachment {
        FileAttachment {
            path: path.into(),
            content: content.into(),
        }
    }

    #[test]
    fn empty_input_produces_no_message() {
        let result = build_post_compact_message(vec![], &PostCompactConfig::default());
        assert!(result.restored_message.is_none());
        assert!(result.restored_paths.is_empty());
    }

    #[test]
    fn restores_files_in_input_order() {
        let result = build_post_compact_message(
            vec![file("/a.rs", "alpha"), file("/b.rs", "bravo"), file("/c.rs", "charlie")],
            &PostCompactConfig::default(),
        );
        let paths: Vec<String> = result
            .restored_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        assert_eq!(paths, vec!["/a.rs", "/b.rs", "/c.rs"]);
    }

    #[test]
    fn caps_at_max_files_to_restore() {
        let cfg = PostCompactConfig {
            max_files_to_restore: 2,
            ..Default::default()
        };
        let result = build_post_compact_message(
            vec![
                file("/1.rs", "a"),
                file("/2.rs", "b"),
                file("/3.rs", "c"),
                file("/4.rs", "d"),
            ],
            &cfg,
        );
        assert_eq!(result.restored_paths.len(), 2);
        assert_eq!(result.skipped_paths.len(), 2);
    }

    #[test]
    fn skips_when_token_budget_exhausted() {
        // Each file is 4_000 chars (~1_000 tokens). Budget = 1_500
        // tokens. Should fit one + skip the rest.
        let big = "x".repeat(4_000);
        let cfg = PostCompactConfig {
            max_files_to_restore: 5,
            token_budget: 1_500,
            max_tokens_per_file: 5_000,
            ..Default::default()
        };
        let result = build_post_compact_message(
            vec![file("/a.rs", &big), file("/b.rs", &big), file("/c.rs", &big)],
            &cfg,
        );
        assert!(result.restored_paths.len() < 3);
        assert!(!result.skipped_paths.is_empty());
    }

    #[test]
    fn truncates_overlong_file_per_max_tokens_per_file() {
        // 40_000 chars (~10_000 tokens) but cap is 1_000.
        let huge = "x".repeat(40_000);
        let cfg = PostCompactConfig {
            max_files_to_restore: 1,
            token_budget: 50_000,
            max_tokens_per_file: 1_000,
            ..Default::default()
        };
        let result = build_post_compact_message(vec![file("/big.rs", &huge)], &cfg);
        assert_eq!(result.restored_paths.len(), 1);
        // The token total should be ~1_000 + overhead, not ~10_000.
        assert!(result.tokens_used < 1_500);
        // Verify the truncation marker landed in the message.
        let msg = result.restored_message.unwrap();
        if let Message::User { content, .. } = msg {
            let combined: String = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            assert!(combined.contains("truncated"));
        } else {
            panic!("expected User message");
        }
    }

    #[test]
    fn message_includes_path_header_per_file() {
        let result = build_post_compact_message(
            vec![file("/path/one.rs", "alpha"), file("/path/two.rs", "bravo")],
            &PostCompactConfig::default(),
        );
        let msg = result.restored_message.unwrap();
        if let Message::User { content, .. } = msg {
            let combined: String = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            assert!(combined.contains("--- /path/one.rs ---"));
            assert!(combined.contains("--- /path/two.rs ---"));
            assert!(combined.contains("Post-compaction file restoration"));
        }
    }
}
