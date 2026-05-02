//! Token-cost accounting (P1 #10 from `notes/2026-05-02-claude-code-non-tui-gaps.md`).
//!
//! agent-rs already emits [`crate::stream::Event::Usage`] (input /
//! output / cache-read / cache-write counts per turn). This module
//! converts those counts into a USD cost using a host-supplied
//! [`ModelPriceCatalog`], and aggregates them across turns into a
//! [`CostTracker`] / [`CostSnapshot`].
//!
//! # Why nanodollars
//!
//! All internal arithmetic uses an integer unit — **nanodollars**
//! (ND) — defined as `1e-9 USD`. Rationale:
//!
//! - Industry rates are quoted in `$/MTok` with up to three decimal
//!   places (e.g. `$15.00/MTok`, `$0.075/MTok` for some cached-input
//!   tiers). The conversion `1 USD/MTok = 1000 ND/token` is exact at
//!   3-decimal resolution; a coarser unit like microcents (`1e-8 USD`)
//!   would inflate `$0.075/MTok` from 75 ND/tok to 8 microcents
//!   (~6.7% over-charge).
//! - Multiplication of `u64` token counts by `u64` ND rates stays
//!   inside `u128` with ~21 orders of magnitude of headroom — even
//!   `1e12` tokens at `$1000/MTok` is only `1e18` ND, far below
//!   `u128::MAX ≈ 3.4e38`.
//! - The host can always export to `f64` for display, but we never
//!   accumulate in `f64` and therefore avoid the silent precision
//!   drift that makes "session total" disagree with the sum of its
//!   parts.
//!
//! # Wiring
//!
//! ```rust,ignore
//! use agent::cost::{CostTracker, ModelPriceCatalog};
//! use agent::stream::Event;
//!
//! let catalog = ModelPriceCatalog::with_anthropic_defaults();
//! let mut tracker = CostTracker::new(catalog.into());
//!
//! // For each Event::Usage observed on the stream:
//! tracker.observe_event("claude-opus-4-7", &event);
//!
//! let snap = tracker.snapshot();
//! println!("session total: {}", snap.format_total_usd());
//! ```

mod catalog;
mod prices;
mod tracker;

pub use catalog::ModelPriceCatalog;
pub use prices::{ModelPrices, NANODOLLARS_PER_USD};
pub use tracker::{CostSnapshot, CostTracker, ModelCost, UnknownModelTokens};
