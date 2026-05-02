//! Memory age helpers (Tier 1 / claude-code parity).
//!
//! Mirrors `memdir/memoryAge.ts`. Eviction-scoring needs the age of
//! each memory file relative to "now"; this module exposes a single
//! helper plus the buckets used by the relevance scorer.

use std::path::Path;
use std::time::{Duration, SystemTime};

/// Age of `path`'s last modification, relative to `now`. Returns
/// [`Duration::ZERO`] if the modification time is in the future
/// (clock skew) and `Err` if the file metadata is unreadable.
pub fn file_age(path: &Path, now: SystemTime) -> Result<Duration, std::io::Error> {
    let mtime = std::fs::metadata(path)?.modified()?;
    Ok(now.duration_since(mtime).unwrap_or(Duration::ZERO))
}

/// Discrete age bucket — used by [`super::relevance`] to penalize
/// stale memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AgeBucket {
    /// Modified in the last 7 days.
    Fresh,
    /// 7 days–60 days.
    Recent,
    /// 60 days–365 days.
    Stale,
    /// >365 days.
    Ancient,
}

impl AgeBucket {
    /// Map a duration to its bucket.
    pub fn from_age(age: Duration) -> Self {
        const SECONDS_PER_DAY: u64 = 60 * 60 * 24;
        let days = age.as_secs() / SECONDS_PER_DAY;
        if days < 7 {
            Self::Fresh
        } else if days < 60 {
            Self::Recent
        } else if days < 365 {
            Self::Stale
        } else {
            Self::Ancient
        }
    }

    /// Relevance penalty multiplier — Fresh = 1.0, Ancient = 0.4.
    /// Used to multiply against the raw mention-match score.
    pub fn relevance_multiplier(self) -> f32 {
        match self {
            Self::Fresh => 1.00,
            Self::Recent => 0.85,
            Self::Stale => 0.60,
            Self::Ancient => 0.40,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn fresh_bucket_for_recent() {
        assert_eq!(
            AgeBucket::from_age(Duration::from_secs(60 * 60 * 24 * 3)),
            AgeBucket::Fresh
        );
    }

    #[test]
    fn recent_bucket_for_2_weeks() {
        assert_eq!(
            AgeBucket::from_age(Duration::from_secs(60 * 60 * 24 * 14)),
            AgeBucket::Recent
        );
    }

    #[test]
    fn stale_bucket_for_3_months() {
        assert_eq!(
            AgeBucket::from_age(Duration::from_secs(60 * 60 * 24 * 90)),
            AgeBucket::Stale
        );
    }

    #[test]
    fn ancient_bucket_for_2_years() {
        assert_eq!(
            AgeBucket::from_age(Duration::from_secs(60 * 60 * 24 * 730)),
            AgeBucket::Ancient
        );
    }

    #[test]
    fn multipliers_are_strictly_decreasing() {
        assert!(AgeBucket::Fresh.relevance_multiplier() > AgeBucket::Recent.relevance_multiplier());
        assert!(AgeBucket::Recent.relevance_multiplier() > AgeBucket::Stale.relevance_multiplier());
        assert!(
            AgeBucket::Stale.relevance_multiplier() > AgeBucket::Ancient.relevance_multiplier()
        );
    }

    #[test]
    fn file_age_for_freshly_written_file_is_small() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("x.md");
        fs::write(&p, "hi").unwrap();
        let age = file_age(&p, SystemTime::now()).unwrap();
        assert!(age < Duration::from_secs(60));
    }

    #[test]
    fn file_age_in_future_clamps_to_zero() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("x.md");
        fs::write(&p, "hi").unwrap();
        // "now" is the epoch, far before the file was written.
        let age = file_age(&p, SystemTime::UNIX_EPOCH).unwrap();
        assert_eq!(age, Duration::ZERO);
    }
}
