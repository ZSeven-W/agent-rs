//! Skills system (Tier 2 / claude-code parity).
//!
//! Mirrors `services/skills/`. A "skill" is a prompt template +
//! metadata that the host can invoke as a callable behaviour:
//!
//! - Name, description (1-line), full prompt template.
//! - Optional model override (some skills want a smarter model).
//! - Optional tool allowlist (some skills need scoped tools only).
//! - Optional input schema (JSON-schema describing call parameters).
//!
//! Skills are loaded from a directory tree:
//!
//! ```text
//! skills/
//! ├── code-review/
//! │   └── SKILL.md
//! └── translate/
//!     └── SKILL.md
//! ```
//!
//! Each `SKILL.md` is a frontmatter document re-using
//! [`crate::memdir::frontmatter::parse`].

pub mod loader;
pub mod registry;
pub mod skill;

pub use loader::{load_dir, LoadError, LoadOutcome, LoadWarning};
pub use registry::SkillRegistry;
pub use skill::{Skill, SkillError, SkillInvocation};
