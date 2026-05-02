//! YAML-subset frontmatter parser (Tier 1 / claude-code parity).
//!
//! Memory files start with a fenced YAML block:
//!
//! ```text
//! ---
//! name: My Memory
//! description: One-line summary used during recall.
//! type: feedback
//! ---
//!
//! Body content goes here.
//! ```
//!
//! We deliberately do NOT pull in a full YAML parser — the subset we
//! need is `key: value` pairs (plus list-of-strings for tags) and we
//! gain a smaller dep tree. If callers ever need richer YAML (anchors,
//! complex maps), swap to `serde_yaml` behind a feature gate.
//!
//! ## Behavior
//!
//! - The frontmatter delimiter is exactly `---` on its own line at
//!   the start of the file (BOM-tolerant) and again to close.
//! - Keys are the substring before the first `:`; values are the rest
//!   of the line trimmed.
//! - Lists may be inline (`tags: [a, b, c]`) or block (`tags:` with
//!   `- a` on subsequent lines).
//! - Quotes (`"…"` or `'…'`) on values are stripped.
//! - Escape sequences are NOT decoded — values are literal.
//! - Files without frontmatter are valid and yield empty metadata
//!   plus the full content as `body`.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Frontmatter {
    pub fields: BTreeMap<String, FieldValue>,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldValue {
    Scalar(String),
    List(Vec<String>),
}

impl FieldValue {
    pub fn as_scalar(&self) -> Option<&str> {
        match self {
            Self::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[String]> {
        match self {
            Self::List(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FrontmatterError {
    #[error("frontmatter opened but never closed (missing trailing `---`)")]
    Unterminated,
    #[error("malformed list value at line {line}: {detail}")]
    BadList { line: usize, detail: String },
    #[error("duplicate key `{0}` in frontmatter")]
    DuplicateKey(String),
}

/// Parse a Markdown file with optional YAML-subset frontmatter.
///
/// Strips a UTF-8 BOM if present. If the file does not begin with the
/// `---` delimiter (after BOM stripping), the entire input is treated
/// as the body and `fields` is empty.
pub fn parse(input: &str) -> Result<Frontmatter, FrontmatterError> {
    let stripped = strip_bom(input);

    let mut lines = stripped.lines();
    let first = match lines.next() {
        Some(l) => l,
        None => {
            return Ok(Frontmatter {
                fields: BTreeMap::new(),
                body: String::new(),
            });
        }
    };
    if first.trim_end_matches('\r') != "---" {
        // No frontmatter — entire input is body.
        return Ok(Frontmatter {
            fields: BTreeMap::new(),
            body: stripped.to_string(),
        });
    }

    let mut fields: BTreeMap<String, FieldValue> = BTreeMap::new();
    let mut closed = false;
    let mut current_list_key: Option<String> = None;
    let mut current_list_acc: Vec<String> = Vec::new();
    let mut line_no: usize;

    let mut body_start: Option<usize> = None;

    // Precompute byte positions of each line start in stripped.
    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(stripped.match_indices('\n').map(|(i, _)| i + 1))
        .collect();

    let line_iter: Vec<&str> = stripped.lines().collect();

    // Skip the first delimiter we already consumed.
    for (idx_after_first, raw) in line_iter.iter().skip(1).enumerate() {
        let idx = idx_after_first + 1; // index in line_iter
        line_no = idx + 1;
        let line = raw.trim_end_matches('\r');
        if line == "---" {
            closed = true;
            // Body starts after this line.
            if let Some(next_start) = line_starts.get(idx + 1) {
                body_start = Some(*next_start);
            } else {
                body_start = Some(stripped.len());
            }
            // Flush any trailing block list. If no items were ever
            // collected, downgrade to an empty scalar (closer to YAML
            // literal semantics for a bare `tags:`).
            if let (Some(k), v) = (
                current_list_key.take(),
                std::mem::take(&mut current_list_acc),
            ) {
                let value = if v.is_empty() {
                    FieldValue::Scalar(String::new())
                } else {
                    FieldValue::List(v)
                };
                if fields.insert(k.clone(), value).is_some() {
                    return Err(FrontmatterError::DuplicateKey(k));
                }
            }
            break;
        }
        // Block list continuation: line starts with leading "- "
        let trimmed = line.trim_start();
        if let Some(item_str) = trimmed.strip_prefix("- ") {
            let key = current_list_key.clone().ok_or(FrontmatterError::BadList {
                line: line_no,
                detail: "list item with no preceding key".into(),
            })?;
            let item = strip_quotes(item_str.trim());
            current_list_acc.push(item);
            // Keep accumulating; this is the same `key`.
            let _ = key;
            continue;
        }
        if line.trim().is_empty() {
            // Blank lines inside frontmatter: ignored.
            continue;
        }
        // Comment lines (`#`-prefixed after stripping leading whitespace)
        // are dropped silently. Without this they'd be parsed as
        // unknown keys via the `key: value` arm below.
        if trimmed.starts_with('#') {
            continue;
        }
        // Not a list item — flush any pending block list.
        if let (Some(k), v) = (
            current_list_key.take(),
            std::mem::take(&mut current_list_acc),
        ) {
            // If the key was actually a list opener that received
            // zero items, store it as an empty scalar instead of an
            // empty list. Lets `tags:` (with nothing) round-trip as
            // an empty value rather than an empty array, which is
            // closer to YAML's literal interpretation.
            let value = if v.is_empty() {
                FieldValue::Scalar(String::new())
            } else {
                FieldValue::List(v)
            };
            if fields.insert(k.clone(), value).is_some() {
                return Err(FrontmatterError::DuplicateKey(k));
            }
        }

        // Key: value
        let Some((k, rest)) = line.split_once(':') else {
            // Non-conforming line — skip silently rather than failing.
            continue;
        };
        let key = k.trim().to_string();
        if key.is_empty() {
            continue;
        }
        let value = rest.trim();
        if value.is_empty() {
            // Could be a block-list opener (`tags:` with `- a` lines
            // following) OR an empty scalar (`tags:` with no items).
            // We can't disambiguate without lookahead — store the key
            // tentatively as a list collector. If no items follow
            // before the next key/EOF, the flush logic above (or in
            // the closing `---` handler below) will downgrade the
            // empty-list to an empty-scalar.
            current_list_key = Some(key);
            continue;
        }
        if value.starts_with('[') && value.ends_with(']') {
            let inner = &value[1..value.len() - 1];
            let items: Vec<String> = inner
                .split(',')
                .map(|s| strip_quotes(s.trim()))
                .filter(|s| !s.is_empty())
                .collect();
            if fields
                .insert(key.clone(), FieldValue::List(items))
                .is_some()
            {
                return Err(FrontmatterError::DuplicateKey(key));
            }
            continue;
        }
        let scalar = strip_quotes(value);
        if fields
            .insert(key.clone(), FieldValue::Scalar(scalar))
            .is_some()
        {
            return Err(FrontmatterError::DuplicateKey(key));
        }
    }

    if !closed {
        return Err(FrontmatterError::Unterminated);
    }

    let body = match body_start {
        Some(b) => stripped[b..].to_string(),
        None => String::new(),
    };

    Ok(Frontmatter { fields, body })
}

fn strip_bom(s: &str) -> &str {
    s.strip_prefix('\u{feff}').unwrap_or(s)
}

fn strip_quotes(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_frontmatter_returns_full_body() {
        let f = parse("Just plain content.\nMore.").unwrap();
        assert!(f.fields.is_empty());
        assert_eq!(f.body, "Just plain content.\nMore.");
    }

    #[test]
    fn simple_scalar_fields() {
        let input = "---\nname: Foo\ndescription: Bar baz\ntype: user\n---\nbody text";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("name").and_then(FieldValue::as_scalar),
            Some("Foo")
        );
        assert_eq!(
            f.fields.get("description").and_then(FieldValue::as_scalar),
            Some("Bar baz")
        );
        assert_eq!(
            f.fields.get("type").and_then(FieldValue::as_scalar),
            Some("user")
        );
        assert_eq!(f.body, "body text");
    }

    #[test]
    fn quoted_scalar_strips_quotes() {
        let input = "---\nname: \"My Name\"\ndesc: 'apostrophed'\n---\n";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("name").and_then(FieldValue::as_scalar),
            Some("My Name")
        );
        assert_eq!(
            f.fields.get("desc").and_then(FieldValue::as_scalar),
            Some("apostrophed")
        );
    }

    #[test]
    fn inline_list_value() {
        let input = "---\ntags: [a, b, c]\n---\nx";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("tags").and_then(FieldValue::as_list),
            Some(&["a".to_string(), "b".to_string(), "c".to_string()][..])
        );
    }

    #[test]
    fn block_list_value() {
        let input = "---\ntags:\n- a\n- b\n- c\nname: x\n---\nbody";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("tags").and_then(FieldValue::as_list),
            Some(&["a".to_string(), "b".to_string(), "c".to_string()][..])
        );
        assert_eq!(
            f.fields.get("name").and_then(FieldValue::as_scalar),
            Some("x")
        );
        assert_eq!(f.body, "body");
    }

    #[test]
    fn unterminated_frontmatter_errors() {
        let input = "---\nname: foo\nstill in frontmatter";
        match parse(input) {
            Err(FrontmatterError::Unterminated) => {}
            other => panic!("expected Unterminated, got {other:?}"),
        }
    }

    #[test]
    fn bom_is_tolerated() {
        let input = "\u{feff}---\nname: x\n---\nbody";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("name").and_then(FieldValue::as_scalar),
            Some("x")
        );
        assert_eq!(f.body, "body");
    }

    #[test]
    fn duplicate_key_errors() {
        let input = "---\nname: a\nname: b\n---\nbody";
        match parse(input) {
            Err(FrontmatterError::DuplicateKey(ref k)) if k == "name" => {}
            other => panic!("expected DuplicateKey, got {other:?}"),
        }
    }

    #[test]
    fn empty_body_after_frontmatter() {
        let input = "---\nname: x\n---\n";
        let f = parse(input).unwrap();
        assert_eq!(f.body, "");
    }

    #[test]
    fn empty_value_with_no_items_is_empty_scalar() {
        // `tags:` with no list items must NOT become an empty array;
        // store as empty scalar instead.
        let input = "---\ntags:\nname: x\n---\n";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("tags"),
            Some(&FieldValue::Scalar(String::new()))
        );
        assert_eq!(
            f.fields.get("name").and_then(FieldValue::as_scalar),
            Some("x")
        );
    }

    #[test]
    fn comment_lines_are_skipped() {
        // `# This is a comment` would otherwise parse as an unknown
        // key+value via split_once(':').
        let input = "---\n# A comment\nname: x\n# another: comment\n---\nbody";
        let f = parse(input).unwrap();
        assert_eq!(f.fields.len(), 1);
        assert_eq!(
            f.fields.get("name").and_then(FieldValue::as_scalar),
            Some("x")
        );
    }

    #[test]
    fn empty_value_followed_by_list_items_is_list() {
        let input = "---\ntags:\n- a\n- b\n---\n";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("tags").and_then(FieldValue::as_list),
            Some(&["a".to_string(), "b".to_string()][..])
        );
    }

    #[test]
    fn crlf_line_endings_handled() {
        let input = "---\r\nname: x\r\n---\r\nbody";
        let f = parse(input).unwrap();
        assert_eq!(
            f.fields.get("name").and_then(FieldValue::as_scalar),
            Some("x")
        );
        assert!(f.body.contains("body"));
    }
}
