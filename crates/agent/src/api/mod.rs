//! API service layer (Tier 1 / claude-code parity).
//!
//! Mirrors `services/api/` from Claude Code. Provides cross-cutting
//! concerns that wrap [`crate::provider`] calls:
//!
//! - [`retry`] — exponential-backoff retry policy + error
//!   classification ([`services/api/withRetry.ts`]).
//! - [`errors`] — API error classification + parsing for rate limits,
//!   auth, model-deprecation, content-policy ([`services/api/errors.ts`]).
//! - [`cache`] — prompt cache break detection: tracks system hash +
//!   tool schema hash + beta header set, surfaces a hook event when a
//!   break is detected ([`services/api/promptCacheBreakDetection.ts`]).
//! - [`effort`] — provider-agnostic effort/thinking budget config
//!   that maps to `extended_thinking.budget_tokens` (Anthropic) and
//!   `reasoning_effort` (OpenAI-compatible).
//! - [`output`] — structured output config (JSON-schema mode for both
//!   providers).
//! - [`logging`] — request/response fingerprinting for cost tracking
//!   + trace IDs.
//!
//! [`services/api/withRetry.ts`]: https://github.com/anthropics/claude-code
//! [`services/api/errors.ts`]: https://github.com/anthropics/claude-code
//! [`services/api/promptCacheBreakDetection.ts`]: https://github.com/anthropics/claude-code

pub mod cache;
pub mod effort;
pub mod errors;
pub mod logging;
pub mod output;
pub mod retry;

pub use cache::{CacheBreakKind, CacheBreakObservation, PromptCacheState, PromptCacheTracker};
pub use effort::{EffortBudget, EffortLevel};
pub use errors::{parse_api_error, with_retry_after, ApiErrorKind, ParsedApiError};
pub use logging::{redact_secrets, request_fingerprint, RequestFingerprint};
pub use output::{OutputConfig, OutputMode, ValidationError};
pub use retry::{
    classify_error, retry_async, CallContext, RetryClassification, RetryPolicy, Retryable,
};
