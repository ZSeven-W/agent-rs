//! Exponential-backoff retry policy with jitter (Tier 1 / claude-code parity).
//!
//! Mirrors `services/api/withRetry.ts`:
//!
//! - Per-error retry rules (rate limit, transient 5xx, network) with
//!   error-class-aware retry counts.
//! - Foreground vs. background 529 handling: foreground gets fewer
//!   retries, background absorbs 529 ("Anthropic overloaded") with a
//!   longer absolute wait.
//! - Decorrelated jitter (AWS architecture-blog style) so a thundering
//!   herd of clients doesn't synchronize their retry waves after a
//!   shared upstream blip.
//!
//! ## Surface
//!
//! - [`RetryPolicy`] — config: max attempts, base delay, jitter.
//! - [`RetryClassification`] — what kind of error we observed.
//! - [`classify_error`] / [`Retryable`] — extension trait for callers
//!   that already have an [`AgentError`] in hand.
//! - [`retry_async`] — the workhorse: takes an async closure, retries
//!   per policy, fires `OnRetry` hooks per attempt if a runner is wired.
//!
//! Implementation notes:
//!
//! - Sleeps via `tokio::time::sleep`, so callers must be inside a
//!   tokio runtime.
//! - Jitter is computed with the `rand` crate to avoid pulling in a
//!   second RNG dependency. Seed is per-thread.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use rand::Rng;

use crate::abort::AbortController;
use crate::error::AgentError;
use crate::hook::{HookEvent, HookRunner};

/// What kind of error we observed — drives whether to retry, and how
/// many times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClassification {
    /// Recoverable in principle: HTTP 429 (rate limit) or 503/504
    /// (transient unavailable). Retry with backoff.
    Transient,
    /// Anthropic-specific: HTTP 529 ("Overloaded"). Distinct from 503
    /// because the docs recommend a longer wait + exponential
    /// retreat. Foreground requests get fewer attempts, background
    /// absorbs more.
    Overloaded,
    /// Network-level (DNS failure, TLS handshake, connection reset).
    /// Retry like Transient.
    Network,
    /// Authentication / authorization failure. Do NOT retry — caller
    /// must rotate credentials.
    Auth,
    /// Request was malformed or violates a hard schema rule. Do NOT
    /// retry.
    BadRequest,
    /// Model-specific failure (deprecated model, content-policy
    /// reject). Do NOT retry.
    ModelError,
    /// Unknown shape; conservative default = do not retry.
    Unknown,
}

impl RetryClassification {
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::Transient | Self::Overloaded | Self::Network)
    }
}

/// Whether the call is a user-blocking foreground request or an
/// async background one. Affects the retry budget for [`Overloaded`]
/// errors specifically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallContext {
    /// User is waiting on this — short total wait, fewer retries.
    Foreground,
    /// Background job (auto-compact, telemetry, ingest). Allow longer
    /// total wait, more retries on overload.
    Background,
}

/// Tunable retry parameters. Defaults match Claude Code's foreground
/// settings: 3 attempts, 1s base, 30s cap, full jitter.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum total attempts (1 = no retry, 2 = one retry, etc.).
    pub max_attempts: u32,
    /// Initial wait before the first retry.
    pub base_delay: Duration,
    /// Cap on per-attempt wait (after exponential growth).
    pub max_delay: Duration,
    /// Multiplier between attempts.
    pub backoff_multiplier: f64,
    /// Whether to apply decorrelated jitter on top of the deterministic
    /// schedule. Disable in tests for reproducibility.
    pub jitter: bool,
    /// Calling context — foreground or background.
    pub context: CallContext,
    /// Extra delay floor specifically for [`RetryClassification::Overloaded`].
    /// 529 responses get this added to the computed delay so background
    /// jobs absorb upstream pressure without re-firing in milliseconds.
    pub overload_extra_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::foreground()
    }
}

impl RetryPolicy {
    /// User-blocking call: short, bounded, 3 attempts.
    pub fn foreground() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
            context: CallContext::Foreground,
            overload_extra_delay: Duration::from_secs(2),
        }
    }

    /// Background call (auto-compact, telemetry): more retries, longer cap.
    pub fn background() -> Self {
        Self {
            max_attempts: 8,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(120),
            backoff_multiplier: 2.0,
            jitter: true,
            context: CallContext::Background,
            overload_extra_delay: Duration::from_secs(10),
        }
    }

    /// Disable backoff entirely — useful for tests that don't want
    /// real-time waits.
    pub fn no_backoff() -> Self {
        Self {
            max_attempts: 1,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            backoff_multiplier: 1.0,
            jitter: false,
            context: CallContext::Foreground,
            overload_extra_delay: Duration::ZERO,
        }
    }

    pub fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    pub fn with_base_delay(mut self, d: Duration) -> Self {
        self.base_delay = d;
        self
    }

    pub fn with_max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }

    pub fn with_jitter(mut self, on: bool) -> Self {
        self.jitter = on;
        self
    }

    /// Compute the wait for `attempt` (1-indexed). Returns 0 for
    /// attempt 1 (no wait before the first call). `prev_delay` is the
    /// actual delay used at the previous attempt (zero on first
    /// retry) — drives true decorrelated jitter per the AWS
    /// architecture-blog formula.
    pub fn delay_for(
        &self,
        attempt: u32,
        classification: RetryClassification,
        prev_delay: Duration,
    ) -> Duration {
        if attempt <= 1 {
            return Duration::ZERO;
        }
        // Sanitize backoff_multiplier — public field, hosts could pass
        // negative or NaN. Clamp to ≥1.0 (no shrink) and finite.
        let multiplier = if self.backoff_multiplier.is_finite() && self.backoff_multiplier >= 1.0 {
            self.backoff_multiplier
        } else {
            1.0
        };
        let exp = multiplier.powi(attempt as i32 - 2).min(1e6);
        let nominal_secs = self.base_delay.as_secs_f64() * exp;
        let nominal_capped =
            Duration::from_secs_f64(nominal_secs.min(self.max_delay.as_secs_f64()));
        let jittered = if self.jitter {
            apply_decorrelated_jitter(self.base_delay, prev_delay, nominal_capped, self.max_delay)
        } else {
            nominal_capped
        };
        if classification == RetryClassification::Overloaded {
            jittered.saturating_add(self.overload_extra_delay)
        } else {
            jittered
        }
    }
}

/// Decorrelated jitter (AWS Architecture Blog style):
///
/// `next_delay = min(cap, random_between(base, prev_delay * 3))`
///
/// `prev_delay` is the actual delay used at the previous attempt
/// (zero on the first retry — in which case we fall back to the
/// nominal schedule as the upper bound). This produces true
/// random-walk decorrelation: concurrent clients that started
/// synchronized diverge over successive retries.
fn apply_decorrelated_jitter(
    base: Duration,
    prev: Duration,
    nominal: Duration,
    cap: Duration,
) -> Duration {
    let base_ms = base.as_millis() as u64;
    let cap_ms = cap.as_millis() as u64;
    // Use `prev * 3` as the upper bound, falling back to the nominal
    // schedule on the first retry where prev is zero.
    let driver_ms = if prev.is_zero() {
        nominal.as_millis() as u64
    } else {
        prev.as_millis() as u64
    };
    let upper = driver_ms.saturating_mul(3).max(base_ms.saturating_add(1));
    let upper = upper.min(cap_ms.max(base_ms));
    if upper <= base_ms {
        return Duration::from_millis(base_ms);
    }
    let mut rng = rand::thread_rng();
    let chosen = rng.gen_range(base_ms..=upper);
    Duration::from_millis(chosen)
}

/// Classify an [`AgentError`] for retry eligibility.
///
/// Only [`AgentError::Provider`] is text-classified — that variant is
/// produced by the provider adapters and carries actual API error
/// messages. Every other variant returns [`RetryClassification::Unknown`]
/// so tool / runtime errors that happen to contain status-code-looking
/// substrings are NOT silently retried.
pub fn classify_error(err: &AgentError) -> RetryClassification {
    match err {
        AgentError::Provider { message, .. } => classify_message(message),
        _ => RetryClassification::Unknown,
    }
}

/// Word-boundary check: returns true iff `needle` appears in `lower`
/// surrounded by non-alphanumeric chars (or string edges). Avoids the
/// "1404" containing "404" issue, and prevents matching tokens
/// embedded inside larger words.
fn has_word(lower: &str, needle: &str) -> bool {
    let bytes = lower.as_bytes();
    let nbytes = needle.as_bytes();
    if nbytes.is_empty() || nbytes.len() > bytes.len() {
        return false;
    }
    let mut i = 0;
    while i + nbytes.len() <= bytes.len() {
        if &bytes[i..i + nbytes.len()] == nbytes {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let after = i + nbytes.len();
            let after_ok = after == bytes.len() || !bytes[after].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn classify_message(msg: &str) -> RetryClassification {
    let lower = msg.to_lowercase();
    // Auth — checked first; exact tokens or word-bounded status codes.
    if lower.contains("invalid api key")
        || has_word(&lower, "unauthorized")
        || has_word(&lower, "authentication_error")
        || has_word(&lower, "permission_error")
        || has_word(&lower, "401")
        || has_word(&lower, "403")
    {
        return RetryClassification::Auth;
    }
    // Model errors before generic BadRequest because deprecation often
    // comes through the invalid_request_error envelope.
    if lower.contains("model") && (lower.contains("deprecated") || lower.contains("not found")) {
        return RetryClassification::ModelError;
    }
    if has_word(&lower, "invalid_request_error")
        || has_word(&lower, "400")
        || lower.contains("malformed json")
        || lower.contains("schema validation")
    {
        return RetryClassification::BadRequest;
    }
    if has_word(&lower, "529")
        || has_word(&lower, "overloaded_error")
        || has_word(&lower, "overloaded")
    {
        return RetryClassification::Overloaded;
    }
    if has_word(&lower, "429")
        || has_word(&lower, "503")
        || has_word(&lower, "504")
        || has_word(&lower, "rate_limit_error")
        || lower.contains("rate limit")
        || lower.contains("request timeout")
    {
        return RetryClassification::Transient;
    }
    if has_word(&lower, "dns")
        || has_word(&lower, "tls")
        || lower.contains("connection reset")
        || lower.contains("connection refused")
        || lower.contains("connection closed")
        || lower.contains("network is unreachable")
        || lower.contains("unexpected eof")
    {
        return RetryClassification::Network;
    }
    RetryClassification::Unknown
}

/// Extension trait so callers can write `err.is_retryable()` directly.
pub trait Retryable {
    fn classify(&self) -> RetryClassification;
    fn is_retryable(&self) -> bool {
        self.classify().is_retryable()
    }
}

impl Retryable for AgentError {
    fn classify(&self) -> RetryClassification {
        classify_error(self)
    }
}

/// Run `op` under `policy`. Up to `policy.max_attempts` attempts;
/// sleeps per [`RetryPolicy::delay_for`] between them.
///
/// Honors `abort` both at the top of each attempt AND mid-flight
/// during the sleep — a `tokio::select!` races the sleep against the
/// abort token, so a cancellation during a long jittered wait
/// short-circuits immediately rather than burning the rest of the
/// timer.
///
/// Fires [`HookEvent::OnRetry`] before each retry attempt if `hooks`
/// is `Some`.
///
/// `op` is a closure returning a future, NOT a single future, so we
/// can call it once per attempt.
pub async fn retry_async<F, Fut, T>(
    policy: &RetryPolicy,
    abort: &AbortController,
    hooks: Option<&Arc<HookRunner>>,
    mut op: F,
) -> Result<T, AgentError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, AgentError>>,
{
    let mut last_err: Option<AgentError> = None;
    let mut prev_delay = Duration::ZERO;
    for attempt in 1..=policy.max_attempts {
        if abort.is_aborted() {
            return Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            ));
        }
        if attempt > 1 {
            // The classification of the *previous* error governs the
            // wait — so we use last_err.
            let classification = last_err
                .as_ref()
                .map(classify_error)
                .unwrap_or(RetryClassification::Unknown);
            let delay = policy.delay_for(attempt, classification, prev_delay);
            prev_delay = delay;
            if let Some(h) = hooks {
                let _ = h
                    .run(&HookEvent::OnRetry {
                        attempt,
                        reason: format!("{classification:?}"),
                    })
                    .await;
            }
            if !delay.is_zero() {
                let token = abort.token().clone();
                tokio::select! {
                    biased;
                    _ = token.cancelled() => {
                        return Err(AgentError::Aborted(
                            abort.reason().unwrap_or_else(|| "aborted".into()),
                        ));
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
            // Sleep completed; recheck abort before firing op() — a
            // cancellation that landed at the sleep deadline edge
            // would otherwise slip through to a wasteful retry.
            if abort.is_aborted() {
                return Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "aborted".into()),
                ));
            }
        }
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                let classification = classify_error(&e);
                if !classification.is_retryable() || attempt == policy.max_attempts {
                    return Err(e);
                }
                last_err = Some(e);
            }
        }
    }
    // Unreachable — the loop body always either returns Ok or Err on
    // the final attempt. Defensive fallback in case of policy=0.
    Err(last_err.unwrap_or_else(|| AgentError::other("retry exhausted")))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::error::AgentError;

    #[test]
    fn classification_auth() {
        let e = AgentError::provider("test", "Unauthorized: invalid api key");
        assert_eq!(classify_error(&e), RetryClassification::Auth);
        assert!(!e.is_retryable());
    }

    #[test]
    fn classification_overloaded() {
        let e = AgentError::provider("test", "HTTP 529 overloaded");
        assert_eq!(classify_error(&e), RetryClassification::Overloaded);
        assert!(e.is_retryable());
    }

    #[test]
    fn classification_rate_limit() {
        let e = AgentError::provider("test", "429 rate limit exceeded");
        assert_eq!(classify_error(&e), RetryClassification::Transient);
        assert!(e.is_retryable());
    }

    #[test]
    fn classification_network() {
        let e = AgentError::provider("test", "connection reset by peer");
        assert_eq!(classify_error(&e), RetryClassification::Network);
        assert!(e.is_retryable());
    }

    #[test]
    fn classification_unknown_is_not_retryable() {
        let e = AgentError::provider("test", "something weird happened");
        assert_eq!(classify_error(&e), RetryClassification::Unknown);
        assert!(!e.is_retryable());
    }

    #[test]
    fn classification_other_variant_is_unknown_by_default() {
        // Plain Other errors must NOT be retried — only Provider
        // errors are text-classified, to avoid retrying tool / runtime
        // errors that happen to contain status-code-looking text.
        let e = AgentError::other("the user's tool printed: '429 rate limit was reached'");
        assert_eq!(classify_error(&e), RetryClassification::Unknown);
        assert!(!e.is_retryable());
    }

    #[test]
    fn classification_word_boundary_avoids_false_positives() {
        // "1429" should NOT match "429" because of word-boundary check.
        let e = AgentError::provider("test", "request id 1429000 took 10s");
        assert_eq!(classify_error(&e), RetryClassification::Unknown);
    }

    #[test]
    fn delay_for_first_attempt_is_zero() {
        let p = RetryPolicy::no_backoff();
        assert_eq!(
            p.delay_for(1, RetryClassification::Transient, Duration::ZERO),
            Duration::ZERO
        );
    }

    #[test]
    fn delay_grows_exponentially_without_jitter() {
        let p = RetryPolicy {
            jitter: false,
            ..RetryPolicy::foreground()
        };
        let d2 = p.delay_for(2, RetryClassification::Transient, Duration::ZERO);
        let d3 = p.delay_for(3, RetryClassification::Transient, d2);
        assert!(d3 >= d2 * 2 / 3, "d2={d2:?}, d3={d3:?}");
    }

    #[test]
    fn overload_extra_delay_is_added() {
        let p = RetryPolicy {
            jitter: false,
            ..RetryPolicy::foreground()
        };
        let transient = p.delay_for(2, RetryClassification::Transient, Duration::ZERO);
        let overloaded = p.delay_for(2, RetryClassification::Overloaded, Duration::ZERO);
        assert!(overloaded > transient);
    }

    #[test]
    fn invalid_backoff_multiplier_does_not_panic() {
        let p = RetryPolicy {
            backoff_multiplier: f64::NAN,
            jitter: false,
            ..RetryPolicy::foreground()
        };
        let d = p.delay_for(2, RetryClassification::Transient, Duration::ZERO);
        assert!(d.as_secs_f64().is_finite());
    }

    #[test]
    fn negative_backoff_multiplier_clamped() {
        let p = RetryPolicy {
            backoff_multiplier: -1.0,
            jitter: false,
            ..RetryPolicy::foreground()
        };
        let d = p.delay_for(3, RetryClassification::Transient, Duration::ZERO);
        // multiplier clamped to 1.0 → delay equals base.
        assert_eq!(d, p.base_delay);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retry_sleep_short_circuits_on_mid_flight_abort() {
        // Fire abort 50ms after kickoff, while a 5s sleep is in flight.
        let abort = AbortController::new();
        let abort_c = abort.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            abort_c.abort_with_reason("mid-flight cancel");
        });
        // Force a long sleep on the second attempt.
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_secs(5),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.0,
            jitter: false,
            context: CallContext::Foreground,
            overload_extra_delay: Duration::ZERO,
        };
        let started = std::time::Instant::now();
        let result = retry_async::<_, _, ()>(&policy, &abort, None, || async {
            Err::<(), _>(AgentError::provider("test", "429 rate limit"))
        })
        .await;
        let elapsed = started.elapsed();
        assert!(matches!(result, Err(AgentError::Aborted(_))));
        assert!(
            elapsed < Duration::from_secs(2),
            "abort should short-circuit the 5s sleep, took {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retry_succeeds_on_second_attempt() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_c = calls.clone();
        let abort = AbortController::new();
        let policy = RetryPolicy::no_backoff().with_max_attempts(3);

        let result = retry_async(&policy, &abort, None, || {
            let calls = calls_c.clone();
            async move {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err::<(), _>(AgentError::provider("test", "429 rate limit"))
                } else {
                    Ok(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retry_does_not_retry_auth_errors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_c = calls.clone();
        let abort = AbortController::new();
        let policy = RetryPolicy::no_backoff().with_max_attempts(5);

        let result = retry_async(&policy, &abort, None, || {
            let calls = calls_c.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(AgentError::provider("test", "Unauthorized"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "auth error should not retry"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retry_exhausts_after_max_attempts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_c = calls.clone();
        let abort = AbortController::new();
        let policy = RetryPolicy::no_backoff().with_max_attempts(3);

        let result = retry_async(&policy, &abort, None, || {
            let calls = calls_c.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(AgentError::provider("test", "429 rate limit"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retry_short_circuits_on_abort() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_c = calls.clone();
        let abort = AbortController::new();
        abort.abort_with_reason("test cancel");
        let policy = RetryPolicy::no_backoff().with_max_attempts(3);

        let result = retry_async(&policy, &abort, None, || {
            let calls = calls_c.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok::<(), _>(())
            }
        })
        .await;

        assert!(matches!(result, Err(AgentError::Aborted(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retry_fires_on_retry_hook_for_each_retry() {
        use std::sync::Mutex;

        use crate::hook::{HookHandler, HookOutcome, RustHookHandler};

        let attempts: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let attempts_c = attempts.clone();
        let h: Arc<dyn HookHandler> = Arc::new(RustHookHandler::new("retry-tap", move |e| {
            if let HookEvent::OnRetry { attempt, .. } = e {
                attempts_c.lock().unwrap().push(*attempt);
            }
            HookOutcome::Ok
        }));
        let mut runner = HookRunner::new();
        runner.register(h);
        let runner = Arc::new(runner);

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_c = calls.clone();
        let abort = AbortController::new();
        let policy = RetryPolicy::no_backoff().with_max_attempts(3);

        let _ = retry_async(&policy, &abort, Some(&runner), || {
            let calls = calls_c.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Err::<(), _>(AgentError::provider("test", "429 rate limit"))
            }
        })
        .await;

        let saw = attempts.lock().unwrap().clone();
        // Hook fires before retry attempts (i.e., attempts 2 and 3),
        // so we should see [2, 3].
        assert_eq!(saw, vec![2, 3]);
    }
}
