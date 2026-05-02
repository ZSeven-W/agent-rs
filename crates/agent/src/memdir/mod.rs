//! Memory directory (Tier 1 / claude-code parity).
//!
//! Long-term, cross-conversation memory store. Mirrors `memdir/` from
//! Claude Code:
//!
//! - [`memory_type`] — the four-type taxonomy (User / Feedback /
//!   Project / Reference) plus body validation rules.
//! - [`frontmatter`] — YAML-subset parser for the `---` block at the
//!   top of each memory file.
//! - [`paths`] — directory resolution (env override, XDG, OS-specific
//!   defaults) and project-slug derivation.
//! - [`scan`] — non-recursive `.md` discovery with stable lex order.
//! - [`age`] — file-mtime → age-bucket mapping for relevance decay.
//! - [`relevance`] — mention-based scoring with type bias + length
//!   penalty + age decay.
//! - [`loader`] — top-level [`load_dir`] that produces strongly-typed
//!   [`Memory`] entries plus a warnings list.
//!
//! ## Known design choices
//!
//! - **Frontmatter parser is a YAML *subset***. Unsupported features
//!   include block scalars (`|`, `>`), folded multi-line scalars,
//!   anchors, and complex maps. Hosts that need full YAML should
//!   load files manually and call [`frontmatter::parse`] only on the
//!   key-value head.
//! - **Hard frontmatter errors drop the file**. A file with malformed
//!   frontmatter (e.g., unterminated, duplicate keys) is dropped
//!   from the loaded corpus rather than partially loaded with whatever
//!   fields parsed before the failure. Partial recovery is a future
//!   extension; for now, loaders prefer "load nothing" to "load a
//!   half-fact and surprise the model".
//! - **Scan order is byte-lex, not locale-aware**. Stable across
//!   runs, identical on every host, but accented filenames don't
//!   sort the way a human reader expects. Acceptable for v1
//!   because the order only affects diagnostic output, not relevance
//!   scoring.
//!
//! Typical use:
//!
//! ```no_run
//! use agent::memdir;
//! use std::time::SystemTime;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = memdir::project_memory_dir(std::path::Path::new("/path/to/proj"))?;
//! let outcome = memdir::load_dir(&dir, SystemTime::now())?;
//! let scored = memdir::find_relevant(
//!     "user wants to test database migration",
//!     &outcome.memories,
//!     &memdir::RelevanceConfig::default(),
//! );
//! for s in scored {
//!     println!("- {}: score {:.2}", s.memory.name, s.score);
//! }
//! # Ok(()) }
//! ```

pub mod age;
pub mod frontmatter;
pub mod loader;
pub mod memory_type;
pub mod paths;
pub mod relevance;
pub mod scan;

pub use age::{file_age, AgeBucket};
pub use frontmatter::{parse as parse_frontmatter, FieldValue, Frontmatter, FrontmatterError};
pub use loader::{load_dir, LoadError, LoadOutcome, LoadWarning, WarningKind};
pub use memory_type::{validate_body, MemoryType, ValidationWarning};
pub use paths::{
    global_memory_root, index_file, project_memory_dir, project_slug, project_slug_unique,
    PathResolveError,
};
pub use relevance::{find_relevant, RelevanceConfig, ScoredMemory};
pub use scan::{scan_dir, ScanError, ScannedFile};

use std::path::PathBuf;
use std::time::Duration;

/// One loaded memory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Memory {
    pub kind: MemoryType,
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
    /// Age relative to the wall-clock at load time. Drives the
    /// [`AgeBucket`] used in relevance scoring.
    pub age: Duration,
}
