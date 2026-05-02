//! Stateful accumulator that turns [`crate::stream::Event::Usage`]
//! observations into running USD costs.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::catalog::ModelPriceCatalog;
use super::prices::NANODOLLARS_PER_USD;
use crate::stream::Event;

/// Per-model accumulator. Tokens are `u64` so a long-running session
/// can't overflow even at provider-side maximums; cost is `u128`
/// nanodollars (1e-9 USD).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCost {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// Cumulative cost in nanodollars (1e-9 USD).
    pub cost_nd: u128,
}

impl ModelCost {
    /// USD as `f64`. Lossless for any session under ~`2^53` ND
    /// (~`$9_007_199`), display-only — keep aggregating in
    /// nanodollars.
    pub fn cost_usd(&self) -> f64 {
        nd_to_usd_f64(self.cost_nd)
    }
}

/// Stateful cost tracker. Cheap to construct (`Arc<ModelPriceCatalog>`
/// is shared); cheap to clone the snapshot.
///
/// **Thread-safety**: mutating methods take `&mut self`. Callers
/// sharing a tracker across tasks should wrap it in
/// `Arc<Mutex<CostTracker>>` (or equivalent) themselves; the catalog
/// pointer is already cheap to share via `Arc`.
#[derive(Debug, Clone)]
pub struct CostTracker {
    catalog: Arc<ModelPriceCatalog>,
    per_model: BTreeMap<String, ModelCost>,
    /// Running total in nanodollars, kept in sync with `per_model` to
    /// avoid an O(N) pass on every snapshot.
    total_nd: u128,
    /// Tokens charged to models the catalog doesn't know about,
    /// summed per model id. Surfaced verbatim in the snapshot so
    /// hosts notice missing rate entries instead of seeing a $0
    /// total.
    unknown_models: BTreeMap<String, UnknownModelTokens>,
    /// Per-model "last cumulative usage we observed via
    /// [`Self::observe_event`]". Lets `observe_event` apply the
    /// delta against the previous Usage in the same turn instead of
    /// double-counting cumulative reports. A new Usage whose any
    /// field is *less than* the cached value is treated as a new
    /// turn's first event and replaces the cache wholesale.
    event_baselines: BTreeMap<String, UsageDelta>,
}

/// Tokens charged to a model the catalog doesn't recognize. No `cost`
/// field — by definition we don't have a price.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnknownModelTokens {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

/// Internal type used by [`CostTracker::observe_event`] to track
/// last-seen cumulative counts per model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UsageDelta {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
}

impl CostTracker {
    pub fn new(catalog: Arc<ModelPriceCatalog>) -> Self {
        Self {
            catalog,
            per_model: BTreeMap::new(),
            total_nd: 0,
            unknown_models: BTreeMap::new(),
            event_baselines: BTreeMap::new(),
        }
    }

    /// Replace the catalog. Existing accumulated costs are preserved
    /// — hosts swapping rates mid-session see the new prices applied
    /// to subsequent observations only. To re-cost from scratch,
    /// call [`Self::reset`] after swapping.
    pub fn set_catalog(&mut self, catalog: Arc<ModelPriceCatalog>) {
        self.catalog = catalog;
    }

    /// Forget every accumulated number. Catalog stays.
    pub fn reset(&mut self) {
        self.per_model.clear();
        self.total_nd = 0;
        self.unknown_models.clear();
        self.event_baselines.clear();
    }

    /// **Explicit-delta API.** Observe one usage report against the
    /// named model. Token counts are added to the per-model
    /// accumulator; if the model is in the catalog, the cost is
    /// computed and added to both per-model and session totals. If
    /// not, the tokens land in the `unknown_models` bucket so the
    /// host can flag a config miss.
    ///
    /// Hosts that get raw incremental token counts use this; hosts
    /// that just forward provider [`Event::Usage`] frames want
    /// [`Self::observe_event`] instead, which handles the
    /// "cumulative-so-far" semantics those frames carry.
    pub fn observe(
        &mut self,
        model_id: &str,
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
    ) {
        match self.catalog.lookup(model_id) {
            Some(prices) => {
                let entry = self.per_model.entry(model_id.to_string()).or_default();
                entry.input_tokens = entry.input_tokens.saturating_add(input_tokens);
                entry.output_tokens = entry.output_tokens.saturating_add(output_tokens);
                entry.cache_read_tokens = entry.cache_read_tokens.saturating_add(cache_read_tokens);
                entry.cache_write_tokens =
                    entry.cache_write_tokens.saturating_add(cache_write_tokens);
                let delta = prices.cost_nd(
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                );
                entry.cost_nd = entry.cost_nd.saturating_add(delta);
                self.total_nd = self.total_nd.saturating_add(delta);
            }
            None => {
                let entry = self.unknown_models.entry(model_id.to_string()).or_default();
                entry.input_tokens = entry.input_tokens.saturating_add(input_tokens);
                entry.output_tokens = entry.output_tokens.saturating_add(output_tokens);
                entry.cache_read_tokens = entry.cache_read_tokens.saturating_add(cache_read_tokens);
                entry.cache_write_tokens =
                    entry.cache_write_tokens.saturating_add(cache_write_tokens);
            }
        }
    }

    /// Drop the per-model baseline used by [`Self::observe_event`].
    /// Call this between turns when a host knows a new turn is
    /// starting — the next [`Self::observe_event`] will then treat
    /// its full cumulative count as the turn's first delta. Without
    /// this signal, two turns whose cumulative-warm-cache counts
    /// happen to match (e.g., back-to-back identical prompt-cache
    /// hits) would compute a delta of zero and silently undercharge
    /// the second turn.
    ///
    /// Hosts that drive turns through [`crate::query::QueryLoop`]
    /// should call this in their per-turn finalize step (or before
    /// dispatching the next stream).
    pub fn clear_event_baseline(&mut self, model_id: &str) {
        self.event_baselines.remove(model_id);
    }

    /// **Cumulative-event API.** Convenience wrapper for callers
    /// that forward [`Event::Usage`] frames straight through to the
    /// tracker. The Usage event is documented as carrying
    /// "for-the-turn-so-far" counts and may be emitted multiple
    /// times per turn — naively adding each report would
    /// double-count.
    ///
    /// Heuristic:
    ///
    /// - Compute a per-field delta against the per-model baseline
    ///   cached from the prior `observe_event` call.
    /// - If every observed field is `>=` its cached value, the new
    ///   report is treated as the latest cumulative within the same
    ///   turn — only the delta is applied.
    /// - If any observed field decreased relative to the cache, the
    ///   tracker concludes a new turn has started: the cache is
    ///   replaced wholesale and the full new report is applied as
    ///   the turn's first delta.
    ///
    /// **Edge cases:**
    ///
    /// - *Warm-cache same-cumulative*: two consecutive turns that
    ///   end with identical token counts (cache served everything)
    ///   would each diff to zero. Call
    ///   [`Self::clear_event_baseline`] between turns to force the
    ///   next event to be charged in full.
    /// - *Concurrent same-model streams*: the baseline is keyed by
    ///   model id only. Two streams for the same model interleaved
    ///   on one tracker corrupt each other's deltas. Hosts running
    ///   parallel streams should either use a tracker per stream or
    ///   call [`Self::observe`] directly with explicit deltas.
    /// - *Mixed `observe` / `observe_event`*: [`Self::observe`]
    ///   does not touch the event baselines. Mixing the two for the
    ///   same model double-counts. Pick one API per model id, or
    ///   call [`Self::clear_event_baseline`] when switching.
    ///
    /// Returns `true` iff the event was a `Event::Usage` (so callers
    /// can route other events without an extra match).
    pub fn observe_event(&mut self, model_id: &str, event: &Event) -> bool {
        let Event::Usage {
            input_tokens,
            output_tokens,
            cache_read,
            cache_create,
        } = event
        else {
            return false;
        };
        let new_total = UsageDelta {
            input_tokens: *input_tokens as u64,
            output_tokens: *output_tokens as u64,
            cache_read_tokens: *cache_read as u64,
            cache_write_tokens: *cache_create as u64,
        };
        let baseline = self
            .event_baselines
            .get(model_id)
            .copied()
            .unwrap_or_default();
        let delta = if usage_progressed(&baseline, &new_total) {
            subtract(&new_total, &baseline)
        } else {
            // Any field decreased → new turn's cumulative restart.
            new_total
        };
        self.observe(
            model_id,
            delta.input_tokens,
            delta.output_tokens,
            delta.cache_read_tokens,
            delta.cache_write_tokens,
        );
        self.event_baselines.insert(model_id.to_string(), new_total);
        true
    }

    /// Snapshot the current totals. The snapshot is independent of
    /// the tracker — subsequent `observe` calls don't mutate it.
    pub fn snapshot(&self) -> CostSnapshot {
        CostSnapshot {
            total_nd: self.total_nd,
            per_model: self.per_model.clone(),
            unknown_models: self.unknown_models.clone(),
        }
    }

    /// Cumulative session total in nanodollars. Cheap O(1) read.
    pub fn total_nd(&self) -> u128 {
        self.total_nd
    }
}

fn usage_progressed(baseline: &UsageDelta, new: &UsageDelta) -> bool {
    new.input_tokens >= baseline.input_tokens
        && new.output_tokens >= baseline.output_tokens
        && new.cache_read_tokens >= baseline.cache_read_tokens
        && new.cache_write_tokens >= baseline.cache_write_tokens
}

fn subtract(new: &UsageDelta, baseline: &UsageDelta) -> UsageDelta {
    UsageDelta {
        input_tokens: new.input_tokens.saturating_sub(baseline.input_tokens),
        output_tokens: new.output_tokens.saturating_sub(baseline.output_tokens),
        cache_read_tokens: new
            .cache_read_tokens
            .saturating_sub(baseline.cache_read_tokens),
        cache_write_tokens: new
            .cache_write_tokens
            .saturating_sub(baseline.cache_write_tokens),
    }
}

/// Independent view of a tracker's state. Safe to serialize, log,
/// or render to a UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostSnapshot {
    pub total_nd: u128,
    pub per_model: BTreeMap<String, ModelCost>,
    pub unknown_models: BTreeMap<String, UnknownModelTokens>,
}

impl CostSnapshot {
    /// USD `f64` total. Display only — see the type-level note on
    /// [`ModelCost::cost_usd`].
    pub fn total_usd(&self) -> f64 {
        nd_to_usd_f64(self.total_nd)
    }

    /// Pretty-format the total. Resolution adapts to magnitude:
    /// sessions under one cent show four-decimal precision
    /// (`$0.0042`); larger totals show two-decimal cents (`$1.23`).
    pub fn format_total_usd(&self) -> String {
        format_nd_usd(self.total_nd)
    }

    /// `true` if any model's tokens fell through to the
    /// `unknown_models` bucket. Hosts should surface this as a
    /// warning so missing-rate misses don't silently produce a
    /// shrunken bill.
    pub fn has_unknown_models(&self) -> bool {
        !self.unknown_models.is_empty()
    }
}

fn nd_to_usd_f64(nd: u128) -> f64 {
    // 1 USD = 1e9 nanodollars. Cast through f64 — for accumulated
    // nanodollars below `2^53` (~9e15, i.e. $9e6) the cast is
    // lossless; above that f64 is approximate, which is why the
    // canonical accumulator stays in u128.
    (nd as f64) / (NANODOLLARS_PER_USD as f64)
}

/// Fixed-point format. Uses 4-decimal precision when the value is
/// strictly below `$0.01`, otherwise 2-decimal. No locale grouping —
/// easy to parse back. Zero is rendered as `$0.00`.
fn format_nd_usd(nd: u128) -> String {
    if nd == 0 {
        return "$0.00".to_string();
    }
    let usd = nd_to_usd_f64(nd);
    // $0.01 = 10_000_000 nanodollars.
    if nd < 10_000_000 {
        format!("${usd:.4}")
    } else {
        format!("${usd:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::ModelPrices;

    fn anthropic_catalog() -> Arc<ModelPriceCatalog> {
        Arc::new(ModelPriceCatalog::with_anthropic_defaults())
    }

    #[test]
    fn empty_tracker_has_zero_total() {
        let t = CostTracker::new(anthropic_catalog());
        assert_eq!(t.total_nd(), 0);
        let snap = t.snapshot();
        assert_eq!(snap.total_nd, 0);
        assert!(snap.per_model.is_empty());
        assert!(snap.unknown_models.is_empty());
        assert_eq!(snap.format_total_usd(), "$0.00");
    }

    #[test]
    fn observe_known_model_accumulates_tokens_and_cost() {
        let mut t = CostTracker::new(anthropic_catalog());
        // Opus rate: 15000/75000/1500/18750 ND/tok.
        // 1000*15000 + 500*75000 + 200*1500 + 100*18750
        // = 15_000_000 + 37_500_000 + 300_000 + 1_875_000 = 54_675_000 ND
        t.observe("claude-opus-4-7", 1000, 500, 200, 100);
        assert_eq!(t.total_nd(), 54_675_000);
        let snap = t.snapshot();
        assert_eq!(snap.total_nd, 54_675_000);
        let opus = snap.per_model.get("claude-opus-4-7").unwrap();
        assert_eq!(opus.input_tokens, 1000);
        assert_eq!(opus.output_tokens, 500);
        assert_eq!(opus.cost_nd, 54_675_000);
    }

    #[test]
    fn observe_repeats_accumulate() {
        let mut t = CostTracker::new(anthropic_catalog());
        t.observe("claude-opus-4-7", 1000, 0, 0, 0);
        t.observe("claude-opus-4-7", 500, 0, 0, 0);
        let snap = t.snapshot();
        let opus = snap.per_model.get("claude-opus-4-7").unwrap();
        assert_eq!(opus.input_tokens, 1500);
        assert_eq!(opus.cost_nd, 1500 * 15_000); // 22_500_000 ND
    }

    #[test]
    fn observe_unknown_model_lands_in_unknown_bucket() {
        let mut t = CostTracker::new(anthropic_catalog());
        t.observe("model-not-in-catalog", 1000, 500, 0, 0);
        let snap = t.snapshot();
        assert_eq!(snap.total_nd, 0);
        assert!(snap.per_model.is_empty());
        assert!(snap.has_unknown_models());
        let unk = snap.unknown_models.get("model-not-in-catalog").unwrap();
        assert_eq!(unk.input_tokens, 1000);
        assert_eq!(unk.output_tokens, 500);
    }

    #[test]
    fn observe_event_dispatches_usage_only() {
        let mut t = CostTracker::new(anthropic_catalog());
        let usage = Event::Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read: 0,
            cache_create: 0,
        };
        let text = Event::TextDelta { delta: "hi".into() };
        assert!(t.observe_event("claude-opus-4-7", &usage));
        assert!(!t.observe_event("claude-opus-4-7", &text));
        // 100*15000 + 50*75000 = 1_500_000 + 3_750_000 = 5_250_000 ND
        assert_eq!(t.total_nd(), 5_250_000);
    }

    #[test]
    fn observe_event_treats_cumulative_reports_as_deltas() {
        // Anthropic / OpenAI may emit Usage(100), Usage(200), Usage(300)
        // within one turn where each report is the running cumulative.
        // Tracker must apply the delta, not add each verbatim.
        let mut t = CostTracker::new(anthropic_catalog());
        let mk = |i: u32, o: u32| Event::Usage {
            input_tokens: i,
            output_tokens: o,
            cache_read: 0,
            cache_create: 0,
        };
        t.observe_event("claude-opus-4-7", &mk(100, 0));
        t.observe_event("claude-opus-4-7", &mk(200, 0));
        t.observe_event("claude-opus-4-7", &mk(300, 0));
        // Final cumulative = 300 input tokens, NOT 600.
        let snap = t.snapshot();
        let opus = snap.per_model.get("claude-opus-4-7").unwrap();
        assert_eq!(opus.input_tokens, 300);
        assert_eq!(opus.cost_nd, 300 * 15_000);
    }

    #[test]
    fn clear_event_baseline_charges_next_event_in_full() {
        // The "warm-cache same-cumulative" turn boundary: turn 1
        // ends at 100 input tokens, turn 2 starts and re-emits 100
        // (identical because the prompt cache satisfied the request
        // verbatim). Without the baseline reset the delta would be
        // zero — turn 2's input would silently fall off the bill.
        let mut t = CostTracker::new(anthropic_catalog());
        let mk = |i: u32| Event::Usage {
            input_tokens: i,
            output_tokens: 0,
            cache_read: 0,
            cache_create: 0,
        };
        t.observe_event("claude-opus-4-7", &mk(100));
        t.clear_event_baseline("claude-opus-4-7");
        t.observe_event("claude-opus-4-7", &mk(100));
        let snap = t.snapshot();
        let opus = snap.per_model.get("claude-opus-4-7").unwrap();
        assert_eq!(opus.input_tokens, 200);
    }

    #[test]
    fn observe_event_idempotent_duplicate_charges_zero_delta() {
        // Two identical Usage events in one turn must not
        // double-count.
        let mut t = CostTracker::new(anthropic_catalog());
        let usage = Event::Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read: 0,
            cache_create: 0,
        };
        t.observe_event("claude-opus-4-7", &usage);
        t.observe_event("claude-opus-4-7", &usage);
        let snap = t.snapshot();
        let opus = snap.per_model.get("claude-opus-4-7").unwrap();
        assert_eq!(opus.input_tokens, 100);
        assert_eq!(opus.output_tokens, 50);
    }

    #[test]
    fn observe_event_resets_cache_on_decreasing_usage() {
        // Turn 1: cumulative reaches 300. Turn 2: emits 50. The
        // decrease signals a new turn — apply 50 as a fresh delta,
        // not -250.
        let mut t = CostTracker::new(anthropic_catalog());
        let mk = |i: u32| Event::Usage {
            input_tokens: i,
            output_tokens: 0,
            cache_read: 0,
            cache_create: 0,
        };
        t.observe_event("claude-opus-4-7", &mk(300));
        t.observe_event("claude-opus-4-7", &mk(50));
        let snap = t.snapshot();
        let opus = snap.per_model.get("claude-opus-4-7").unwrap();
        assert_eq!(opus.input_tokens, 350);
        assert_eq!(opus.cost_nd, 350 * 15_000);
    }

    #[test]
    fn observe_event_per_model_baselines_dont_cross_contaminate() {
        // Two models, both emitting cumulative Usage. Each must
        // apply its own delta independently.
        let mut t = CostTracker::new(Arc::new(ModelPriceCatalog::with_defaults()));
        let mk = |i: u32| Event::Usage {
            input_tokens: i,
            output_tokens: 0,
            cache_read: 0,
            cache_create: 0,
        };
        t.observe_event("claude-opus-4-7", &mk(100));
        t.observe_event("gpt-4o-mini", &mk(500));
        t.observe_event("claude-opus-4-7", &mk(150));
        t.observe_event("gpt-4o-mini", &mk(700));
        let snap = t.snapshot();
        assert_eq!(
            snap.per_model.get("claude-opus-4-7").unwrap().input_tokens,
            150
        );
        assert_eq!(snap.per_model.get("gpt-4o-mini").unwrap().input_tokens, 700);
    }

    #[test]
    fn snapshot_is_independent_from_subsequent_observations() {
        let mut t = CostTracker::new(anthropic_catalog());
        t.observe("claude-opus-4-7", 1000, 0, 0, 0);
        let snap = t.snapshot();
        t.observe("claude-opus-4-7", 1000, 0, 0, 0);
        assert_eq!(snap.total_nd, 15_000 * 1000);
        assert_eq!(t.total_nd(), 15_000 * 2000);
    }

    #[test]
    fn reset_clears_state_but_preserves_catalog() {
        let mut t = CostTracker::new(anthropic_catalog());
        t.observe("claude-opus-4-7", 1000, 0, 0, 0);
        t.observe("model-not-in-catalog", 500, 0, 0, 0);
        // Also poison the event baselines so we know reset clears them.
        t.observe_event(
            "claude-opus-4-7",
            &Event::Usage {
                input_tokens: 100,
                output_tokens: 0,
                cache_read: 0,
                cache_create: 0,
            },
        );
        t.reset();
        let snap = t.snapshot();
        assert_eq!(snap.total_nd, 0);
        assert!(snap.per_model.is_empty());
        assert!(snap.unknown_models.is_empty());
        // After reset, observe_event should treat the next Usage as
        // a fresh cumulative (not subtract a stale baseline).
        t.observe_event(
            "claude-opus-4-7",
            &Event::Usage {
                input_tokens: 1000,
                output_tokens: 0,
                cache_read: 0,
                cache_create: 0,
            },
        );
        assert_eq!(t.total_nd(), 15_000 * 1000);
    }

    #[test]
    fn set_catalog_preserves_previous_totals() {
        let mut t = CostTracker::new(anthropic_catalog());
        t.observe("claude-opus-4-7", 1000, 0, 0, 0);
        let pre = t.total_nd();
        t.set_catalog(Arc::new(ModelPriceCatalog::new()));
        assert_eq!(t.total_nd(), pre);
        t.observe("claude-opus-4-7", 500, 0, 0, 0);
        assert_eq!(t.total_nd(), pre);
        let snap = t.snapshot();
        assert!(snap.has_unknown_models());
    }

    #[test]
    fn format_total_usd_adapts_precision() {
        let mut t = CostTracker::new(anthropic_catalog());
        assert_eq!(t.snapshot().format_total_usd(), "$0.00");
        // Sub-cent: 1000 input tokens at $0.80/MTok = 800 ND/tok × 1000 =
        // 800_000 ND = $0.0008.
        let mut catalog = ModelPriceCatalog::new();
        catalog.insert("tiny", ModelPrices::from_usd_per_mtok(0.80, 0.0, 0.0, 0.0));
        t.set_catalog(Arc::new(catalog));
        t.observe("tiny", 1_000, 0, 0, 0);
        assert_eq!(t.snapshot().format_total_usd(), "$0.0008");
        // Top up to 1M tokens total ⇒ $0.80, two-decimal.
        t.observe("tiny", 999_000, 0, 0, 0);
        assert_eq!(t.snapshot().format_total_usd(), "$0.80");
    }

    #[test]
    fn multi_model_session_aggregates_independently() {
        let mut t = CostTracker::new(Arc::new(ModelPriceCatalog::with_defaults()));
        t.observe("claude-opus-4-7", 1000, 500, 0, 0);
        t.observe("gpt-4o-mini", 10_000, 5_000, 0, 0);
        let snap = t.snapshot();
        // Opus: 1000*15000 + 500*75000 = 52_500_000 ND
        // gpt-4o-mini: 10000*150 + 5000*600 = 1_500_000 + 3_000_000 = 4_500_000 ND
        // total = 57_000_000 ND
        assert_eq!(snap.total_nd, 57_000_000);
        assert_eq!(
            snap.per_model.get("claude-opus-4-7").unwrap().cost_nd,
            52_500_000
        );
        assert_eq!(
            snap.per_model.get("gpt-4o-mini").unwrap().cost_nd,
            4_500_000
        );
    }

    #[test]
    fn saturating_arith_doesnt_panic_on_extreme_token_counts() {
        let mut t = CostTracker::new(anthropic_catalog());
        t.observe("claude-opus-4-7", u64::MAX, 0, 0, 0);
        assert!(t.total_nd() > 0);
    }

    #[test]
    fn snapshot_serde_round_trip() {
        let mut t = CostTracker::new(anthropic_catalog());
        t.observe("claude-opus-4-7", 1000, 500, 0, 0);
        t.observe("unknown-x", 100, 50, 0, 0);
        let snap = t.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let back: CostSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back, snap);
    }

    #[test]
    fn nd_per_usd_constant_is_1e9() {
        assert_eq!(NANODOLLARS_PER_USD, 1_000_000_000);
    }
}
