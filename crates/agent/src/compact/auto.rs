//! Auto-compaction trigger logic (claude-code parity, Tier 1).
//!
//! Mirrors `services/compact/autoCompact.ts`:
//!
//! - Effective context window = `model_max - reserved_for_summary`.
//! - Auto-compact threshold = `effective - AUTOCOMPACT_BUFFER_TOKENS`.
//! - Warning threshold = `effective - WARNING_THRESHOLD_BUFFER_TOKENS`.
//! - Circuit breaker: after [`MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES`]
//!   failures in a row, stop trying (matches Claude Code's BQ
//!   2026-03-10 finding: 1,279 sessions had 50+ consecutive failures
//!   wasting ~250K API calls/day globally).

use uuid::Uuid;

use super::summarize::MAX_OUTPUT_TOKENS_FOR_SUMMARY;

/// Reserved buffer between the auto-compact threshold and the
/// effective context window. Mirrors
/// `AUTOCOMPACT_BUFFER_TOKENS = 13_000`.
pub const AUTOCOMPACT_BUFFER_TOKENS: u32 = 13_000;

/// Reserved buffer for the warning threshold (one warning before the
/// hard auto-compact limit). Mirrors
/// `WARNING_THRESHOLD_BUFFER_TOKENS = 20_000`.
pub const WARNING_THRESHOLD_BUFFER_TOKENS: u32 = 20_000;

/// Reserved buffer for the error threshold.
pub const ERROR_THRESHOLD_BUFFER_TOKENS: u32 = 20_000;

/// Reserved buffer for manual compact (smaller — manual is opt-in).
pub const MANUAL_COMPACT_BUFFER_TOKENS: u32 = 3_000;

/// Stop retrying autocompact after this many consecutive failures.
pub const MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES: u32 = 3;

/// Effective usable context window for input. Equal to the model's
/// declared max minus the budget reserved for the compaction
/// summary's own output.
pub fn effective_context_window(model_max_tokens: u32) -> u32 {
    let reserved = MAX_OUTPUT_TOKENS_FOR_SUMMARY.min(model_max_tokens);
    model_max_tokens.saturating_sub(reserved)
}

/// Token count below which auto-compaction shouldn't fire.
pub fn auto_compact_threshold(model_max_tokens: u32) -> u32 {
    effective_context_window(model_max_tokens)
        .saturating_sub(AUTOCOMPACT_BUFFER_TOKENS)
}

/// Token count below which the user shouldn't see a warning yet.
pub fn warning_threshold(model_max_tokens: u32) -> u32 {
    effective_context_window(model_max_tokens)
        .saturating_sub(WARNING_THRESHOLD_BUFFER_TOKENS)
}

/// Token count below which manual compaction shouldn't be required.
pub fn manual_compact_threshold(model_max_tokens: u32) -> u32 {
    effective_context_window(model_max_tokens)
        .saturating_sub(MANUAL_COMPACT_BUFFER_TOKENS)
}

/// Why the auto-compact decision came out the way it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoCompactReason {
    UnderThreshold {
        current_tokens: u32,
        threshold: u32,
    },
    ThresholdHit {
        current_tokens: u32,
        threshold: u32,
    },
    CircuitBreakerOpen {
        consecutive_failures: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoCompactDecision {
    pub should_compact: bool,
    pub should_warn: bool,
    pub reason: AutoCompactReason,
}

/// Per-session tracker that survives across turns. Owners typically
/// keep this in their session-state struct alongside the
/// [`crate::message::MessageStore`].
#[derive(Debug, Clone)]
pub struct AutoCompactState {
    /// Whether the session has ever been compacted (set on first
    /// success; not cleared).
    pub compacted: bool,
    /// Strictly increasing per turn.
    pub turn_counter: u32,
    /// Unique id for the current turn (regenerated on `next_turn`).
    pub turn_id: String,
    /// Number of consecutive autocompact attempts that failed —
    /// reset to 0 on success. Exceeding
    /// [`MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES`] opens the circuit
    /// breaker.
    pub consecutive_failures: u32,
}

impl Default for AutoCompactState {
    fn default() -> Self {
        Self {
            compacted: false,
            turn_counter: 0,
            turn_id: Uuid::new_v4().to_string(),
            consecutive_failures: 0,
        }
    }
}

impl AutoCompactState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Roll the turn id forward (call this at the start of every
    /// new query loop turn).
    pub fn next_turn(&mut self) {
        self.turn_counter = self.turn_counter.saturating_add(1);
        self.turn_id = Uuid::new_v4().to_string();
    }

    pub fn record_success(&mut self) {
        self.compacted = true;
        self.consecutive_failures = 0;
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
    }

    pub fn circuit_open(&self) -> bool {
        self.consecutive_failures >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES
    }

    /// Evaluate whether the next turn should trigger compaction.
    /// Caller passes the current cumulative token count for messages
    /// that would be sent to the provider, plus the model's declared
    /// max context window.
    pub fn evaluate(&self, current_tokens: u32, model_max_tokens: u32) -> AutoCompactDecision {
        let threshold = auto_compact_threshold(model_max_tokens);
        let warn_threshold = warning_threshold(model_max_tokens);

        let should_warn = current_tokens >= warn_threshold;

        if self.circuit_open() {
            return AutoCompactDecision {
                should_compact: false,
                should_warn,
                reason: AutoCompactReason::CircuitBreakerOpen {
                    consecutive_failures: self.consecutive_failures,
                },
            };
        }

        if current_tokens >= threshold {
            AutoCompactDecision {
                should_compact: true,
                should_warn,
                reason: AutoCompactReason::ThresholdHit {
                    current_tokens,
                    threshold,
                },
            }
        } else {
            AutoCompactDecision {
                should_compact: false,
                should_warn,
                reason: AutoCompactReason::UnderThreshold {
                    current_tokens,
                    threshold,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_context_window_subtracts_summary_budget() {
        // 200K context, 20K reserved => 180K effective.
        assert_eq!(effective_context_window(200_000), 180_000);
    }

    #[test]
    fn effective_context_window_does_not_underflow() {
        // Smaller-than-summary model — saturate at 0.
        assert_eq!(effective_context_window(5_000), 0);
    }

    #[test]
    fn auto_threshold_subtracts_buffer() {
        // 200K -> 180K effective -> 167K threshold.
        assert_eq!(auto_compact_threshold(200_000), 167_000);
    }

    #[test]
    fn warning_threshold_lower_than_auto() {
        let auto = auto_compact_threshold(200_000);
        let warn = warning_threshold(200_000);
        assert!(warn < auto);
    }

    #[test]
    fn evaluate_under_threshold() {
        let s = AutoCompactState::new();
        let d = s.evaluate(50_000, 200_000);
        assert!(!d.should_compact);
        assert!(!d.should_warn);
        assert!(matches!(d.reason, AutoCompactReason::UnderThreshold { .. }));
    }

    #[test]
    fn evaluate_threshold_hit_triggers_compact() {
        let s = AutoCompactState::new();
        // 200K -> 167K threshold; 170K should fire.
        let d = s.evaluate(170_000, 200_000);
        assert!(d.should_compact);
        assert!(matches!(d.reason, AutoCompactReason::ThresholdHit { .. }));
    }

    #[test]
    fn evaluate_warning_emitted_independently() {
        let s = AutoCompactState::new();
        // 200K -> warn at 160K; auto at 167K.
        let d = s.evaluate(162_000, 200_000);
        assert!(d.should_warn);
        assert!(!d.should_compact);
    }

    #[test]
    fn circuit_breaker_blocks_compact_after_three_failures() {
        let mut s = AutoCompactState::new();
        s.record_failure();
        s.record_failure();
        s.record_failure();
        assert!(s.circuit_open());
        let d = s.evaluate(170_000, 200_000); // would otherwise fire
        assert!(!d.should_compact);
        assert!(matches!(
            d.reason,
            AutoCompactReason::CircuitBreakerOpen { consecutive_failures: 3 }
        ));
    }

    #[test]
    fn record_success_resets_failure_counter() {
        let mut s = AutoCompactState::new();
        s.record_failure();
        s.record_failure();
        s.record_success();
        assert_eq!(s.consecutive_failures, 0);
        assert!(s.compacted);
        assert!(!s.circuit_open());
    }

    #[test]
    fn next_turn_increments_counter_and_rotates_id() {
        let mut s = AutoCompactState::new();
        let id_a = s.turn_id.clone();
        let counter_a = s.turn_counter;
        s.next_turn();
        assert!(s.turn_counter > counter_a);
        assert_ne!(s.turn_id, id_a);
    }
}
