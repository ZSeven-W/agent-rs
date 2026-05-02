//! Per-model price catalog.

use std::collections::BTreeMap;

use super::prices::ModelPrices;

/// Lookup table from model id → [`ModelPrices`].
///
/// The catalog is **not** authoritative — published rates change, and
/// new models ship every quarter. agent-rs ships a small set of
/// well-known defaults via [`Self::with_anthropic_defaults`] /
/// [`Self::with_openai_defaults`] so a host can run end-to-end
/// without configuring rates by hand, but production deployments
/// should override / refresh the catalog from their own settings.
///
/// Lookups are exact-match on the model id string. For provider
/// model id flavors (e.g. OpenRouter's `anthropic/claude-3-opus`),
/// hosts insert an entry under each flavor they care about.
#[derive(Debug, Clone, Default)]
pub struct ModelPriceCatalog {
    prices: BTreeMap<String, ModelPrices>,
}

impl ModelPriceCatalog {
    /// Empty catalog. All lookups return `None`; the
    /// [`crate::cost::CostTracker`] tracks tokens charged to unknown
    /// models in a dedicated bucket so missing-rate misses don't
    /// silently zero the bill.
    pub fn new() -> Self {
        Self::default()
    }

    /// Catalog seeded with the public Anthropic Claude lineup as of
    /// 2026-05-02. Rates are published at
    /// <https://www.anthropic.com/pricing> — verify before billing.
    ///
    /// Hosts can chain `.with_anthropic_defaults().insert(...)` to
    /// add their own entries.
    pub fn with_anthropic_defaults() -> Self {
        let mut c = Self::new();
        // Opus tier — top-of-line, $/MTok in: 15.00 / out: 75.00 /
        // cache-read: 1.50 / cache-write: 18.75.
        c.insert(
            "claude-opus-4-7",
            ModelPrices::from_usd_per_mtok(15.00, 75.00, 1.50, 18.75),
        );
        c.insert(
            "claude-opus-4-6",
            ModelPrices::from_usd_per_mtok(15.00, 75.00, 1.50, 18.75),
        );
        c.insert(
            "claude-opus-4",
            ModelPrices::from_usd_per_mtok(15.00, 75.00, 1.50, 18.75),
        );
        // Sonnet tier — balanced, $/MTok in: 3.00 / out: 15.00 /
        // cache-read: 0.30 / cache-write: 3.75.
        c.insert(
            "claude-sonnet-4-6",
            ModelPrices::from_usd_per_mtok(3.00, 15.00, 0.30, 3.75),
        );
        c.insert(
            "claude-sonnet-4",
            ModelPrices::from_usd_per_mtok(3.00, 15.00, 0.30, 3.75),
        );
        c.insert(
            "claude-3-5-sonnet-20241022",
            ModelPrices::from_usd_per_mtok(3.00, 15.00, 0.30, 3.75),
        );
        // Haiku tier — fastest, $/MTok in: 0.80 / out: 4.00 /
        // cache-read: 0.08 / cache-write: 1.00. Includes the
        // canonical 4.5 id used in agent-rs's real-API tests.
        c.insert(
            "claude-haiku-4-5-20251001",
            ModelPrices::from_usd_per_mtok(0.80, 4.00, 0.08, 1.00),
        );
        c.insert(
            "claude-3-5-haiku-20241022",
            ModelPrices::from_usd_per_mtok(0.80, 4.00, 0.08, 1.00),
        );
        c
    }

    /// Catalog seeded with a subset of OpenAI's public pricing as of
    /// 2026-05-02. OpenAI doesn't expose explicit prompt-cache rates
    /// at the API tier we hit, so cache-read mirrors the input rate
    /// (the de-facto cached-input price OpenAI publishes for some
    /// SKUs) and cache-write defaults to zero. Verify against
    /// <https://openai.com/pricing> before relying on this for
    /// billing.
    pub fn with_openai_defaults() -> Self {
        let mut c = Self::new();
        // gpt-5.4-codex — flagship coding model, $/MTok in: 7.50 /
        // out: 30.00. Cache values: best-effort.
        c.insert(
            "gpt-5.4-codex",
            ModelPrices::from_usd_per_mtok(7.50, 30.00, 1.50, 0.0),
        );
        c.insert(
            "gpt-5.3-codex-spark",
            ModelPrices::from_usd_per_mtok(2.00, 8.00, 0.50, 0.0),
        );
        c.insert(
            "gpt-4o",
            ModelPrices::from_usd_per_mtok(2.50, 10.00, 1.25, 0.0),
        );
        c.insert(
            "gpt-4o-mini",
            ModelPrices::from_usd_per_mtok(0.15, 0.60, 0.075, 0.0),
        );
        c
    }

    /// Combine the standard provider defaults. Cheapest path for
    /// hosts that want "show me a USD number" without curating a
    /// catalog themselves.
    pub fn with_defaults() -> Self {
        let mut c = Self::with_anthropic_defaults();
        for (k, v) in Self::with_openai_defaults().prices {
            c.prices.insert(k, v);
        }
        c
    }

    /// Insert / replace an entry. Returns the previously-registered
    /// price, if any.
    pub fn insert(
        &mut self,
        model_id: impl Into<String>,
        prices: ModelPrices,
    ) -> Option<ModelPrices> {
        self.prices.insert(model_id.into(), prices)
    }

    /// Look up an exact-match model id. Returns `None` if the
    /// catalog has no entry — hosts should treat this as a
    /// configuration miss rather than "free".
    pub fn lookup(&self, model_id: &str) -> Option<&ModelPrices> {
        self.prices.get(model_id)
    }

    /// Number of registered models.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Iterate registered (model_id, prices) pairs in lexicographic
    /// order — useful for diagnostics dumps.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ModelPrices)> {
        self.prices.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_catalog_misses() {
        let c = ModelPriceCatalog::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert!(c.lookup("claude-opus-4-7").is_none());
    }

    #[test]
    fn insert_and_lookup_round_trip() {
        let mut c = ModelPriceCatalog::new();
        let prices = ModelPrices::from_usd_per_mtok(1.0, 2.0, 0.0, 0.0);
        let prev = c.insert("custom-model", prices);
        assert!(prev.is_none());
        let got = c.lookup("custom-model").unwrap();
        assert_eq!(*got, prices);
    }

    #[test]
    fn anthropic_defaults_include_all_tiers() {
        let c = ModelPriceCatalog::with_anthropic_defaults();
        assert!(c.lookup("claude-opus-4-7").is_some());
        assert!(c.lookup("claude-sonnet-4-6").is_some());
        assert!(c.lookup("claude-haiku-4-5-20251001").is_some());
    }

    #[test]
    fn anthropic_defaults_match_published_rates() {
        let c = ModelPriceCatalog::with_anthropic_defaults();
        let opus = c.lookup("claude-opus-4-7").unwrap();
        assert_eq!(opus.input_nd_per_token, 15_000); // $15/MTok
        assert_eq!(opus.output_nd_per_token, 75_000); // $75/MTok
        let sonnet = c.lookup("claude-sonnet-4-6").unwrap();
        assert_eq!(sonnet.input_nd_per_token, 3_000); // $3/MTok
        let haiku = c.lookup("claude-haiku-4-5-20251001").unwrap();
        assert_eq!(haiku.input_nd_per_token, 800); // $0.80/MTok
    }

    #[test]
    fn with_defaults_combines_anthropic_and_openai() {
        let c = ModelPriceCatalog::with_defaults();
        assert!(c.lookup("claude-opus-4-7").is_some());
        assert!(c.lookup("gpt-5.4-codex").is_some());
        assert!(c.lookup("nonexistent-model").is_none());
    }

    #[test]
    fn iter_yields_lex_order() {
        let mut c = ModelPriceCatalog::new();
        c.insert("zeta", ModelPrices::default());
        c.insert("alpha", ModelPrices::default());
        c.insert("mu", ModelPrices::default());
        let names: Vec<&str> = c.iter().map(|(k, _)| k).collect();
        assert_eq!(names, vec!["alpha", "mu", "zeta"]);
    }

    #[test]
    fn insert_replace_returns_previous() {
        let mut c = ModelPriceCatalog::new();
        let p1 = ModelPrices::from_usd_per_mtok(1.0, 1.0, 0.0, 0.0);
        let p2 = ModelPrices::from_usd_per_mtok(2.0, 2.0, 0.0, 0.0);
        c.insert("m", p1);
        let prev = c.insert("m", p2);
        assert_eq!(prev, Some(p1));
        assert_eq!(*c.lookup("m").unwrap(), p2);
    }
}
