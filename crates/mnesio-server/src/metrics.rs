//! Search-time telemetry + rolling aggregator.
//!
//! Captures per-query metrics (timings, retrieval counts, score envelopes,
//! which BM25 fallback tier fired, whether source-boost reordered the top
//! result) and rolls them up into a sliding-window snapshot the UI polls
//! once every few seconds.
//!
//! This is the foundation for Phase 6's eval harness. Today's collector is
//! in-memory only; once `Outcome`-level metrics land (Phase 2), they'll
//! flow into the same shape and the rollup grows additional fields like
//! `task_success_rate` per artifact version.

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// How many recent searches the rolling aggregator keeps. ~200 is enough
/// for a meaningful p95 in a hand-driven demo and tiny in memory.
const WINDOW_CAPACITY: usize = 200;

/// Width of each history bucket in milliseconds. Lower for tighter
/// resolution at the cost of more buckets to scan. 60s gives a useful
/// minute-by-minute time series.
const BUCKET_WIDTH_MS: u64 = 60_000;

/// How many history buckets the collector keeps. 60 × 1-minute = 1 hour
/// of trailing history. Old buckets fall off the front as new ones arrive.
const HISTORY_CAPACITY: usize = 60;

/// Cap on latency samples retained per bucket so memory is bounded even
/// under heavy load. 1000 is plenty for representative percentiles within
/// a one-minute window.
const BUCKET_SAMPLE_CAP: usize = 1000;

/// Per-query metrics — every `/api/search` response includes one of these,
/// and a copy is pushed into the [`MetricsCollector`] for rollup.
#[derive(Debug, Clone, Serialize)]
pub struct SearchMetrics {
    /// Wall-clock timestamp (unix ms) when the search started. Lets the
    /// dashboard time-bucket queries even when the server's been up for
    /// days.
    pub timestamp_ms: u64,
    /// The literal query text the user typed. Stored so the recent-queries
    /// table can show it without a second lookup.
    pub query_text: String,
    pub total_ms: u64,
    pub phases: PhaseTimings,
    pub retrieval: RetrievalCounts,
    pub scores: ScoreEnvelope,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PhaseTimings {
    pub snapshot_replay_ms: u64,
    pub embed_query_ms: u64,
    pub vector_search_ms: u64,
    pub bm25_search_ms: u64,
    pub hybrid_fuse_ms: u64,
    pub source_boost_ms: u64,
    pub synthesize_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RetrievalCounts {
    pub vector_hits: usize,
    pub bm25_hits: usize,
    pub hybrid_hits: usize,
    /// Which BM25 fallback tier produced its hits (`strict_and`, `fuzzy_and`,
    /// `or_merge`, or `empty`).
    pub bm25_tier: &'static str,
    /// `true` when the source-boost reordered the top hybrid hit. A useful
    /// signal of how often the boost matters in practice.
    pub source_boost_changed_top: bool,
    /// Number of hits in `hybrid_hits` that received the source-boost lift.
    pub source_boost_lifts: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ScoreEnvelope {
    pub bm25_max: f32,
    pub bm25_min: f32,
    pub vector_max: f32,
    pub vector_min: f32,
    pub rrf_max: f32,
    pub rrf_min: f32,
}

/// Rolling aggregate over the last [`WINDOW_CAPACITY`] queries.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsRollup {
    /// Capacity of the underlying ring buffer.
    pub window_size: usize,
    /// How many queries are actually in the window right now.
    pub queries_in_window: usize,
    /// Lifetime query count since the server started (not just the window).
    pub queries_total: u64,
    pub latency_ms: LatencyStats,
    /// Average per-phase split as % of total time. Helps spot where
    /// budget goes.
    pub phase_share_pct: PhaseShare,
    /// How often each BM25 tier fired across the window.
    pub fallback_distribution: FallbackDistribution,
    /// Percent of queries where source-boost flipped the top hit.
    pub source_boost_change_rate_pct: u8,
    pub avg_hits: AvgHits,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LatencyStats {
    pub p50: u64,
    pub p95: u64,
    pub max: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PhaseShare {
    pub snapshot_replay: u8,
    pub embed: u8,
    pub vector: u8,
    pub bm25: u8,
    pub hybrid_fuse: u8,
    pub source_boost: u8,
    pub synthesize: u8,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FallbackDistribution {
    pub strict_and: u32,
    pub fuzzy_and: u32,
    pub or_merge: u32,
    pub empty: u32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct AvgHits {
    pub vector: f32,
    pub bm25: f32,
    pub hybrid: f32,
}

/// One minute-wide bucket in the metrics history. Aggregates everything
/// the dashboard needs to plot one tick of every time-series chart.
#[derive(Debug, Clone, Serialize)]
pub struct TimeBucket {
    pub start_ms: u64,
    pub width_ms: u64,
    pub query_count: u32,
    pub latency_ms: LatencyStats,
    pub mean_latency_ms: f32,
    /// Phase mean ms across the bucket — useful to see when embedding
    /// got faster, BM25 got slower, etc.
    pub phase_means_ms: PhaseMeansMs,
    pub fallback_distribution: FallbackDistribution,
    pub boost_flips: u32,
    pub mean_hits: AvgHits,
    /// Mean lifetime-of-bucket scores — useful to see if BM25 max scores
    /// are trending up or down.
    pub mean_scores: MeanScores,
    /// Internal: kept for percentile re-computation; not serialised.
    #[serde(skip)]
    latency_samples: Vec<u64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PhaseMeansMs {
    pub snapshot_replay: f32,
    pub embed: f32,
    pub vector: f32,
    pub bm25: f32,
    pub hybrid_fuse: f32,
    pub source_boost: f32,
    pub synthesize: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MeanScores {
    pub bm25_max: f32,
    pub vector_max: f32,
    pub rrf_max: f32,
}

/// Snapshot returned by `/api/metrics/history`. Buckets are oldest → newest.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsHistory {
    pub server_start_ms: u64,
    pub server_uptime_ms: u64,
    pub queries_total: u64,
    pub buckets: Vec<TimeBucket>,
    pub bucket_width_ms: u64,
    /// Quick lifetime summary — handy for the dashboard's KPI strip
    /// without re-aggregating bucket-by-bucket on the client.
    pub lifetime: LifetimeStats,
    /// The most recent N queries (latest first), for the dashboard's
    /// recent-queries table. Capped at `WINDOW_CAPACITY`.
    pub recent_queries: Vec<RecentQuery>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LifetimeStats {
    pub queries: u64,
    pub mean_latency_ms: f32,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub max_latency_ms: u64,
    pub fallback_distribution: FallbackDistribution,
    pub boost_flip_rate_pct: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentQuery {
    pub timestamp_ms: u64,
    pub query_text: String,
    pub total_ms: u64,
    pub bm25_tier: &'static str,
    pub hybrid_hits: usize,
    pub boost_flipped_top: bool,
}

/// In-memory ring buffer of recent searches + a separate time-bucketed
/// history for the long-view dashboard.
///
/// `RwLock` rather than `Mutex` because the rollup reader is independent
/// of writers — multiple readers can compute aggregates concurrently
/// while writers serialise on the push path.
pub struct MetricsCollector {
    window: RwLock<VecDeque<SearchMetrics>>,
    /// Time-bucket series, oldest at the front. Width per
    /// [`BUCKET_WIDTH_MS`], capped at [`HISTORY_CAPACITY`] entries.
    history: RwLock<VecDeque<TimeBucket>>,
    total: RwLock<u64>,
    /// Lifetime aggregates that don't fall off the window. The dashboard
    /// shows these directly.
    lifetime: RwLock<LifetimeAcc>,
    server_start_ms: u64,
}

#[derive(Default)]
struct LifetimeAcc {
    queries: u64,
    latency_sum_ms: u128,
    latency_samples: Vec<u64>,
    fallback: FallbackDistribution,
    boost_flips: u64,
}

impl LifetimeAcc {
    fn record(&mut self, m: &SearchMetrics) {
        self.queries += 1;
        self.latency_sum_ms += m.total_ms as u128;
        // Reservoir-like cap: keep at most 10k samples for lifetime
        // percentile calc. After that, sample 1-in-N to keep
        // representation flat. Approximation is fine for a demo.
        const CAP: usize = 10_000;
        if self.latency_samples.len() < CAP {
            self.latency_samples.push(m.total_ms);
        } else {
            // Replace a random slot ~1/queries of the time so older
            // samples slowly age out — Knuth reservoir style.
            let idx = (m.timestamp_ms as usize)
                .wrapping_mul(2654435761)
                .wrapping_rem(CAP);
            self.latency_samples[idx] = m.total_ms;
        }
        match m.retrieval.bm25_tier {
            "strict_and" => self.fallback.strict_and += 1,
            "fuzzy_and" => self.fallback.fuzzy_and += 1,
            "or_merge" => self.fallback.or_merge += 1,
            _ => self.fallback.empty += 1,
        }
        if m.retrieval.source_boost_changed_top {
            self.boost_flips += 1;
        }
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            window: RwLock::new(VecDeque::with_capacity(WINDOW_CAPACITY)),
            history: RwLock::new(VecDeque::with_capacity(HISTORY_CAPACITY)),
            total: RwLock::new(0),
            lifetime: RwLock::new(LifetimeAcc::default()),
            server_start_ms: now_ms(),
        }
    }

    /// Push the metrics from one completed search into the window. Drops
    /// the oldest entry when full.
    pub fn record(&self, m: SearchMetrics) {
        // Bucket assignment first — we need the timestamp before we move
        // the metrics into the window.
        self.push_to_history(&m);
        self.lifetime
            .write()
            .expect("metrics lifetime poisoned")
            .record(&m);
        let mut w = self.window.write().expect("metrics window poisoned");
        if w.len() == WINDOW_CAPACITY {
            w.pop_front();
        }
        w.push_back(m);
        drop(w);
        let mut t = self.total.write().expect("metrics total poisoned");
        *t += 1;
    }

    fn push_to_history(&self, m: &SearchMetrics) {
        let bucket_start = (m.timestamp_ms / BUCKET_WIDTH_MS) * BUCKET_WIDTH_MS;
        let mut h = self.history.write().expect("metrics history poisoned");

        // Fast path: same bucket as the most recent one.
        if let Some(last) = h.back_mut() {
            if last.start_ms == bucket_start {
                accumulate_into_bucket(last, m);
                return;
            }
            if bucket_start < last.start_ms {
                // Out-of-order timestamp — shouldn't happen with our
                // single-writer setup, but be defensive: drop on the floor.
                return;
            }
        }

        // New bucket. Fill any gap so the time series doesn't have
        // missing x-values (the dashboard then plots a clear "0 queries"
        // gap rather than just connecting across a silence).
        if let Some(last) = h.back() {
            let mut next = last.start_ms + BUCKET_WIDTH_MS;
            while next < bucket_start {
                if h.len() == HISTORY_CAPACITY {
                    h.pop_front();
                }
                h.push_back(empty_bucket(next));
                next += BUCKET_WIDTH_MS;
            }
        }

        if h.len() == HISTORY_CAPACITY {
            h.pop_front();
        }
        let mut fresh = empty_bucket(bucket_start);
        accumulate_into_bucket(&mut fresh, m);
        h.push_back(fresh);
    }

    /// Compute a rollup over the current window. Cheap — O(n log n) on the
    /// window size, which is bounded at [`WINDOW_CAPACITY`].
    pub fn rollup(&self) -> MetricsRollup {
        let w = self.window.read().expect("metrics window poisoned");
        let total = *self.total.read().expect("metrics total poisoned");
        let n = w.len();
        if n == 0 {
            return MetricsRollup {
                window_size: WINDOW_CAPACITY,
                queries_in_window: 0,
                queries_total: total,
                latency_ms: LatencyStats::default(),
                phase_share_pct: PhaseShare::default(),
                fallback_distribution: FallbackDistribution::default(),
                source_boost_change_rate_pct: 0,
                avg_hits: AvgHits::default(),
            };
        }

        let mut latencies: Vec<u64> = w.iter().map(|m| m.total_ms).collect();
        latencies.sort_unstable();
        let p50 = latencies[(n * 50 / 100).min(n - 1)];
        let p95 = latencies[(n * 95 / 100).min(n - 1)];
        let max = *latencies.last().unwrap();

        // Phase share — sum each phase across the window, divide by total
        // time across the window, render as percentages.
        let mut sum = PhaseSumMs::default();
        for m in w.iter() {
            sum.snapshot_replay += m.phases.snapshot_replay_ms;
            sum.embed += m.phases.embed_query_ms;
            sum.vector += m.phases.vector_search_ms;
            sum.bm25 += m.phases.bm25_search_ms;
            sum.hybrid_fuse += m.phases.hybrid_fuse_ms;
            sum.source_boost += m.phases.source_boost_ms;
            sum.synthesize += m.phases.synthesize_ms;
        }
        let phase_total = (sum.snapshot_replay
            + sum.embed
            + sum.vector
            + sum.bm25
            + sum.hybrid_fuse
            + sum.source_boost
            + sum.synthesize)
            .max(1);
        let pct = |x: u64| -> u8 { ((x * 100) / phase_total).min(100) as u8 };
        let phase_share_pct = PhaseShare {
            snapshot_replay: pct(sum.snapshot_replay),
            embed: pct(sum.embed),
            vector: pct(sum.vector),
            bm25: pct(sum.bm25),
            hybrid_fuse: pct(sum.hybrid_fuse),
            source_boost: pct(sum.source_boost),
            synthesize: pct(sum.synthesize),
        };

        // Tier distribution + boost change rate + avg hits.
        let mut tier = FallbackDistribution::default();
        let mut boost_changes = 0u32;
        let mut sum_vector = 0usize;
        let mut sum_bm25 = 0usize;
        let mut sum_hybrid = 0usize;
        for m in w.iter() {
            match m.retrieval.bm25_tier {
                "strict_and" => tier.strict_and += 1,
                "fuzzy_and" => tier.fuzzy_and += 1,
                "or_merge" => tier.or_merge += 1,
                _ => tier.empty += 1,
            }
            if m.retrieval.source_boost_changed_top {
                boost_changes += 1;
            }
            sum_vector += m.retrieval.vector_hits;
            sum_bm25 += m.retrieval.bm25_hits;
            sum_hybrid += m.retrieval.hybrid_hits;
        }
        let source_boost_change_rate_pct = ((boost_changes as u64 * 100) / n as u64) as u8;
        let avg_hits = AvgHits {
            vector: sum_vector as f32 / n as f32,
            bm25: sum_bm25 as f32 / n as f32,
            hybrid: sum_hybrid as f32 / n as f32,
        };

        MetricsRollup {
            window_size: WINDOW_CAPACITY,
            queries_in_window: n,
            queries_total: total,
            latency_ms: LatencyStats { p50, p95, max },
            phase_share_pct,
            fallback_distribution: tier,
            source_boost_change_rate_pct,
            avg_hits,
        }
    }

    /// Snapshot of the time-bucket history plus lifetime stats and the
    /// recent-queries log. Used by `/api/metrics/history` to feed the
    /// dashboard's time-series + table widgets in one round-trip.
    pub fn history(&self) -> MetricsHistory {
        let now = now_ms();
        let buckets = self
            .history
            .read()
            .expect("metrics history poisoned")
            .iter()
            .map(finalize_bucket)
            .collect::<Vec<_>>();
        let lifetime = self.lifetime_stats();
        let recent_queries = self
            .window
            .read()
            .expect("metrics window poisoned")
            .iter()
            .rev() // newest first
            .map(|m| RecentQuery {
                timestamp_ms: m.timestamp_ms,
                query_text: m.query_text.clone(),
                total_ms: m.total_ms,
                bm25_tier: m.retrieval.bm25_tier,
                hybrid_hits: m.retrieval.hybrid_hits,
                boost_flipped_top: m.retrieval.source_boost_changed_top,
            })
            .collect();
        MetricsHistory {
            server_start_ms: self.server_start_ms,
            server_uptime_ms: now.saturating_sub(self.server_start_ms),
            queries_total: *self.total.read().expect("metrics total poisoned"),
            buckets,
            bucket_width_ms: BUCKET_WIDTH_MS,
            lifetime,
            recent_queries,
        }
    }

    fn lifetime_stats(&self) -> LifetimeStats {
        let acc = self.lifetime.read().expect("metrics lifetime poisoned");
        if acc.queries == 0 {
            return LifetimeStats::default();
        }
        let mut samples = acc.latency_samples.clone();
        samples.sort_unstable();
        let n = samples.len().max(1);
        let p50 = samples[(n * 50 / 100).min(n - 1)];
        let p95 = samples[(n * 95 / 100).min(n - 1)];
        let max = *samples.last().unwrap_or(&0);
        let boost_flip_rate_pct = ((acc.boost_flips * 100) / acc.queries).min(100) as u8;
        LifetimeStats {
            queries: acc.queries,
            mean_latency_ms: (acc.latency_sum_ms as f64 / acc.queries as f64) as f32,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            max_latency_ms: max,
            fallback_distribution: acc.fallback.clone(),
            boost_flip_rate_pct,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Wall-clock now() in unix milliseconds. Falls back to 0 if the clock
/// went pre-epoch, which it shouldn't have on any reasonable system.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn empty_bucket(start_ms: u64) -> TimeBucket {
    TimeBucket {
        start_ms,
        width_ms: BUCKET_WIDTH_MS,
        query_count: 0,
        latency_ms: LatencyStats::default(),
        mean_latency_ms: 0.0,
        phase_means_ms: PhaseMeansMs::default(),
        fallback_distribution: FallbackDistribution::default(),
        boost_flips: 0,
        mean_hits: AvgHits::default(),
        mean_scores: MeanScores::default(),
        latency_samples: Vec::new(),
    }
}

fn accumulate_into_bucket(b: &mut TimeBucket, m: &SearchMetrics) {
    b.query_count += 1;
    if b.latency_samples.len() < BUCKET_SAMPLE_CAP {
        b.latency_samples.push(m.total_ms);
    }
    // Running sums kept in the corresponding mean fields temporarily;
    // we divide by query_count in `finalize_bucket`.
    b.mean_latency_ms += m.total_ms as f32;
    b.phase_means_ms.snapshot_replay += m.phases.snapshot_replay_ms as f32;
    b.phase_means_ms.embed += m.phases.embed_query_ms as f32;
    b.phase_means_ms.vector += m.phases.vector_search_ms as f32;
    b.phase_means_ms.bm25 += m.phases.bm25_search_ms as f32;
    b.phase_means_ms.hybrid_fuse += m.phases.hybrid_fuse_ms as f32;
    b.phase_means_ms.source_boost += m.phases.source_boost_ms as f32;
    b.phase_means_ms.synthesize += m.phases.synthesize_ms as f32;
    match m.retrieval.bm25_tier {
        "strict_and" => b.fallback_distribution.strict_and += 1,
        "fuzzy_and" => b.fallback_distribution.fuzzy_and += 1,
        "or_merge" => b.fallback_distribution.or_merge += 1,
        _ => b.fallback_distribution.empty += 1,
    }
    if m.retrieval.source_boost_changed_top {
        b.boost_flips += 1;
    }
    b.mean_hits.vector += m.retrieval.vector_hits as f32;
    b.mean_hits.bm25 += m.retrieval.bm25_hits as f32;
    b.mean_hits.hybrid += m.retrieval.hybrid_hits as f32;
    b.mean_scores.bm25_max += m.scores.bm25_max;
    b.mean_scores.vector_max += m.scores.vector_max;
    b.mean_scores.rrf_max += m.scores.rrf_max;
}

/// Divide running sums by `query_count` to produce means, compute the
/// percentiles from the bucket's latency samples, and return a clone for
/// the wire response.
fn finalize_bucket(b: &TimeBucket) -> TimeBucket {
    if b.query_count == 0 {
        return b.clone();
    }
    let n = b.query_count as f32;
    let mut out = b.clone();
    out.mean_latency_ms /= n;
    out.phase_means_ms.snapshot_replay /= n;
    out.phase_means_ms.embed /= n;
    out.phase_means_ms.vector /= n;
    out.phase_means_ms.bm25 /= n;
    out.phase_means_ms.hybrid_fuse /= n;
    out.phase_means_ms.source_boost /= n;
    out.phase_means_ms.synthesize /= n;
    out.mean_hits.vector /= n;
    out.mean_hits.bm25 /= n;
    out.mean_hits.hybrid /= n;
    out.mean_scores.bm25_max /= n;
    out.mean_scores.vector_max /= n;
    out.mean_scores.rrf_max /= n;

    let mut samples = b.latency_samples.clone();
    samples.sort_unstable();
    let n = samples.len();
    if n > 0 {
        out.latency_ms = LatencyStats {
            p50: samples[(n * 50 / 100).min(n - 1)],
            p95: samples[(n * 95 / 100).min(n - 1)],
            max: *samples.last().unwrap(),
        };
    }
    // Strip the samples — not part of the wire format.
    out.latency_samples.clear();
    out
}

#[derive(Default)]
struct PhaseSumMs {
    snapshot_replay: u64,
    embed: u64,
    vector: u64,
    bm25: u64,
    hybrid_fuse: u64,
    source_boost: u64,
    synthesize: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(total_ms: u64, tier: &'static str, hybrid: usize, boost_flip: bool) -> SearchMetrics {
        SearchMetrics {
            timestamp_ms: now_ms(),
            query_text: "test".into(),
            total_ms,
            phases: PhaseTimings {
                snapshot_replay_ms: 1,
                embed_query_ms: 1,
                vector_search_ms: 1,
                bm25_search_ms: 1,
                hybrid_fuse_ms: 0,
                source_boost_ms: 0,
                synthesize_ms: 0,
            },
            retrieval: RetrievalCounts {
                vector_hits: 5,
                bm25_hits: 4,
                hybrid_hits: hybrid,
                bm25_tier: tier,
                source_boost_changed_top: boost_flip,
                source_boost_lifts: 0,
            },
            scores: ScoreEnvelope::default(),
        }
    }

    fn mk_at(timestamp_ms: u64, total_ms: u64, tier: &'static str) -> SearchMetrics {
        SearchMetrics {
            timestamp_ms,
            query_text: "test".into(),
            total_ms,
            phases: PhaseTimings::default(),
            retrieval: RetrievalCounts {
                vector_hits: 1,
                bm25_hits: 1,
                hybrid_hits: 1,
                bm25_tier: tier,
                source_boost_changed_top: false,
                source_boost_lifts: 0,
            },
            scores: ScoreEnvelope::default(),
        }
    }

    #[test]
    fn empty_collector_returns_zeroed_rollup() {
        let c = MetricsCollector::new();
        let r = c.rollup();
        assert_eq!(r.queries_in_window, 0);
        assert_eq!(r.queries_total, 0);
        assert_eq!(r.latency_ms.p50, 0);
    }

    #[test]
    fn rollup_computes_percentiles() {
        let c = MetricsCollector::new();
        for ms in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
            c.record(mk(ms, "strict_and", 5, false));
        }
        let r = c.rollup();
        assert_eq!(r.queries_in_window, 10);
        assert_eq!(r.latency_ms.max, 10);
        // Indices: 50% of 10 = 5 → latencies[5] == 6
        assert_eq!(r.latency_ms.p50, 6);
        // 95% of 10 = 9 → latencies[9] == 10
        assert_eq!(r.latency_ms.p95, 10);
    }

    #[test]
    fn ring_buffer_caps_at_capacity() {
        let c = MetricsCollector::new();
        for i in 0..(WINDOW_CAPACITY + 50) {
            c.record(mk(i as u64, "strict_and", 1, false));
        }
        let r = c.rollup();
        assert_eq!(r.queries_in_window, WINDOW_CAPACITY);
        assert_eq!(r.queries_total, (WINDOW_CAPACITY + 50) as u64);
    }

    #[test]
    fn fallback_distribution_counts_tiers() {
        let c = MetricsCollector::new();
        c.record(mk(1, "strict_and", 1, false));
        c.record(mk(1, "strict_and", 1, false));
        c.record(mk(1, "fuzzy_and", 1, false));
        c.record(mk(1, "or_merge", 1, false));
        let r = c.rollup();
        assert_eq!(r.fallback_distribution.strict_and, 2);
        assert_eq!(r.fallback_distribution.fuzzy_and, 1);
        assert_eq!(r.fallback_distribution.or_merge, 1);
    }

    #[test]
    fn boost_change_rate_is_percentage() {
        let c = MetricsCollector::new();
        // 3 of 10 queries had boost flip the top — 30 %.
        for i in 0..10 {
            c.record(mk(1, "strict_and", 1, i < 3));
        }
        let r = c.rollup();
        assert_eq!(r.source_boost_change_rate_pct, 30);
    }

    #[test]
    fn avg_hits_per_signal() {
        let c = MetricsCollector::new();
        c.record(mk(1, "strict_and", 5, false));
        c.record(mk(1, "strict_and", 3, false));
        let r = c.rollup();
        assert!((r.avg_hits.hybrid - 4.0).abs() < 1e-6);
    }

    #[test]
    fn history_groups_queries_into_buckets_by_minute() {
        let c = MetricsCollector::new();
        // Align to a bucket boundary so the test's +0/+1s/+59.999s
        // arithmetic stays inside the first bucket deterministically.
        let t0 = (1_700_000_000_000u64 / 60_000) * 60_000;
        // Three in the first bucket, two in the next minute's bucket.
        c.record(mk_at(t0, 10, "strict_and"));
        c.record(mk_at(t0 + 1_000, 12, "strict_and"));
        c.record(mk_at(t0 + 59_999, 14, "fuzzy_and"));
        c.record(mk_at(t0 + 60_000, 8, "strict_and"));
        c.record(mk_at(t0 + 90_000, 20, "or_merge"));
        let h = c.history();
        assert_eq!(h.buckets.len(), 2);
        assert_eq!(h.buckets[0].query_count, 3);
        assert_eq!(h.buckets[1].query_count, 2);
        assert!((h.buckets[0].mean_latency_ms - 12.0).abs() < 1e-3);
    }

    #[test]
    fn history_fills_gaps_with_empty_buckets() {
        let c = MetricsCollector::new();
        let t0 = (1_700_000_000_000u64 / 60_000) * 60_000;
        c.record(mk_at(t0, 10, "strict_and"));
        // Skip ahead by 4 buckets — we expect 3 empty fillers + the new bucket.
        c.record(mk_at(t0 + 4 * 60_000, 20, "strict_and"));
        let h = c.history();
        assert_eq!(h.buckets.len(), 5);
        assert_eq!(h.buckets[0].query_count, 1);
        assert_eq!(h.buckets[1].query_count, 0);
        assert_eq!(h.buckets[2].query_count, 0);
        assert_eq!(h.buckets[3].query_count, 0);
        assert_eq!(h.buckets[4].query_count, 1);
    }

    #[test]
    fn lifetime_stats_track_all_queries() {
        let c = MetricsCollector::new();
        for i in 0..50 {
            c.record(mk((i + 1) as u64, "strict_and", 1, false));
        }
        let h = c.history();
        assert_eq!(h.lifetime.queries, 50);
        assert!(h.lifetime.p95_latency_ms >= h.lifetime.p50_latency_ms);
        assert!(h.lifetime.max_latency_ms >= h.lifetime.p95_latency_ms);
    }

    #[test]
    fn recent_queries_are_newest_first() {
        let c = MetricsCollector::new();
        let t0 = 1_700_000_000_000u64;
        c.record(mk_at(t0, 1, "strict_and"));
        c.record(mk_at(t0 + 1, 2, "fuzzy_and"));
        c.record(mk_at(t0 + 2, 3, "or_merge"));
        let h = c.history();
        assert_eq!(h.recent_queries.len(), 3);
        // newest first
        assert_eq!(h.recent_queries[0].timestamp_ms, t0 + 2);
        assert_eq!(h.recent_queries[2].timestamp_ms, t0);
    }
}
