//! Per-model price tables in nanodollars per token.

use serde::{Deserialize, Serialize};

/// Per-model price table. Each field is **nanodollars per token**
/// (1 nanodollar = `1e-9 USD`). Construct via
/// [`Self::from_usd_per_mtok`] for the standard industry
/// representation.
///
/// Why nanodollars: `1 USD/MTok = 1000 nanodollars/token`, so every
/// 3-decimal published rate (e.g. `$0.075/MTok` cached-input on some
/// SKUs) round-trips losslessly. A coarser unit like microcents
/// (`1e-8 USD`) inflates `$0.075/MTok` from 75 ND/tok to 8 microcents
/// (≈ 6.7% over-charge).
///
/// Cache rates are optional in spirit: providers without prompt
/// caching set both `cache_read` and `cache_write` to zero. The token
/// counters in [`crate::stream::Event::Usage`] will already be zero
/// for those providers, so the multiplication is a no-op.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPrices {
    pub input_nd_per_token: u64,
    pub output_nd_per_token: u64,
    pub cache_read_nd_per_token: u64,
    pub cache_write_nd_per_token: u64,
}

/// Conversion factor: `$1 USD = 1_000_000_000 nanodollars`.
pub const NANODOLLARS_PER_USD: u128 = 1_000_000_000;

/// `$1 USD / MTok = 1000 ND/token`. Multiplier used by
/// [`ModelPrices::from_usd_per_mtok`].
const ND_PER_USD_PER_MTOK: f64 = 1_000.0;

impl ModelPrices {
    /// Construct from the industry-standard `$/MTok` representation.
    ///
    /// Conversion: `rate ($/MTok) × 1000 = ND/token`. The multiplier
    /// is exact for any rate with up to 3 decimal places (every
    /// published Anthropic / OpenAI / DeepSeek / Ollama rate fits).
    /// 4-decimal rates round to the nearest ND/token.
    ///
    /// Inputs that are NaN, infinite, negative, or zero clamp to
    /// zero. Finite-but-out-of-range values (e.g. `1e30`) saturate to
    /// `u64::MAX` per Rust's saturating `f64 as u64` cast.
    pub fn from_usd_per_mtok(input: f64, output: f64, cache_read: f64, cache_write: f64) -> Self {
        Self {
            input_nd_per_token: usd_per_mtok_to_nd(input),
            output_nd_per_token: usd_per_mtok_to_nd(output),
            cache_read_nd_per_token: usd_per_mtok_to_nd(cache_read),
            cache_write_nd_per_token: usd_per_mtok_to_nd(cache_write),
        }
    }

    /// Cost for a single observation (one turn or finer). Returns
    /// nanodollars to keep the caller in integer territory. Internal
    /// multiplications are in `u128`; the four-way sum uses
    /// `saturating_add` so adversarial inputs can't wrap.
    pub fn cost_nd(
        &self,
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
    ) -> u128 {
        let i = (input_tokens as u128).saturating_mul(self.input_nd_per_token as u128);
        let o = (output_tokens as u128).saturating_mul(self.output_nd_per_token as u128);
        let cr = (cache_read_tokens as u128).saturating_mul(self.cache_read_nd_per_token as u128);
        let cw = (cache_write_tokens as u128).saturating_mul(self.cache_write_nd_per_token as u128);
        i.saturating_add(o).saturating_add(cr).saturating_add(cw)
    }
}

fn usd_per_mtok_to_nd(rate: f64) -> u64 {
    if !rate.is_finite() || rate <= 0.0 {
        return 0;
    }
    (rate * ND_PER_USD_PER_MTOK).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_usd_per_mtok_rounds_to_nanodollars() {
        // Claude Opus 4.7 published: $15 input / $75 output / $1.50 cache_read / $18.75 cache_write
        let p = ModelPrices::from_usd_per_mtok(15.0, 75.0, 1.50, 18.75);
        assert_eq!(p.input_nd_per_token, 15_000);
        assert_eq!(p.output_nd_per_token, 75_000);
        assert_eq!(p.cache_read_nd_per_token, 1_500);
        assert_eq!(p.cache_write_nd_per_token, 18_750);
    }

    #[test]
    fn from_usd_per_mtok_handles_three_decimal_rates_exactly() {
        // $0.075/MTok is the canonical "near-zero precision miss" case.
        // Microcents would inflate it from 7.5 → 8 (6.7% over-charge).
        // Nanodollars hit it exactly: 0.075 * 1000 = 75 ND/tok.
        let p = ModelPrices::from_usd_per_mtok(0.075, 0.0, 0.0, 0.0);
        assert_eq!(p.input_nd_per_token, 75);
    }

    #[test]
    fn from_usd_per_mtok_handles_sub_dollar_rates() {
        let p = ModelPrices::from_usd_per_mtok(0.25, 1.25, 0.03, 0.30);
        assert_eq!(p.input_nd_per_token, 250);
        assert_eq!(p.output_nd_per_token, 1_250);
        assert_eq!(p.cache_read_nd_per_token, 30);
        assert_eq!(p.cache_write_nd_per_token, 300);
    }

    #[test]
    fn negative_or_nan_rates_clamp_to_zero() {
        let p = ModelPrices::from_usd_per_mtok(-1.0, f64::NAN, f64::INFINITY, -0.5);
        assert_eq!(p.input_nd_per_token, 0);
        assert_eq!(p.output_nd_per_token, 0);
        assert_eq!(p.cache_read_nd_per_token, 0);
        assert_eq!(p.cache_write_nd_per_token, 0);
    }

    #[test]
    fn cost_nd_is_sum_of_components() {
        let p = ModelPrices::from_usd_per_mtok(15.0, 75.0, 1.50, 18.75);
        // 1000*15000 + 500*75000 + 200*1500 + 100*18750
        // = 15_000_000 + 37_500_000 + 300_000 + 1_875_000
        // = 54_675_000 ND = $0.054675
        let nd = p.cost_nd(1_000, 500, 200, 100);
        assert_eq!(nd, 54_675_000);
    }

    #[test]
    fn cost_nd_handles_zero_token_components() {
        let p = ModelPrices::from_usd_per_mtok(15.0, 75.0, 1.50, 18.75);
        assert_eq!(p.cost_nd(0, 0, 0, 0), 0);
        assert_eq!(p.cost_nd(100, 0, 0, 0), 100 * 15_000);
    }

    #[test]
    fn cost_nd_million_tokens_at_15_per_mtok_is_15_dollars() {
        let p = ModelPrices::from_usd_per_mtok(15.0, 0.0, 0.0, 0.0);
        let nd = p.cost_nd(1_000_000, 0, 0, 0);
        // $15.00 = 15_000_000_000 ND
        assert_eq!(nd, 15 * NANODOLLARS_PER_USD);
    }

    #[test]
    fn cost_nd_saturates_on_max_inputs_no_panic() {
        let p = ModelPrices {
            input_nd_per_token: u64::MAX,
            output_nd_per_token: u64::MAX,
            cache_read_nd_per_token: u64::MAX,
            cache_write_nd_per_token: u64::MAX,
        };
        let nd = p.cost_nd(u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        assert_eq!(nd, u128::MAX);
    }
}
