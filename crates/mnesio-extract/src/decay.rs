//! Forgetting — bounded, biologically-inspired decay (FadeMem style).
//!
//! A memory's *effective* importance falls off with age and is
//! reinforced by access. The consolidation/forgetting worker (host) uses
//! [`DecayModel::decayed`] to rank memories and [`forgettable`] to pick
//! which ones to retire (soft-delete via `MemoryInvalidated`, never a
//! hard drop — Hard Rule #2 keeps the history).
//!
//! Pure math, no I/O, fully deterministic.

/// Exponential-decay model with access reinforcement.
#[derive(Debug, Clone, Copy)]
pub struct DecayModel {
    /// Time for an un-accessed memory's decay factor to halve, in
    /// seconds. Larger = stickier memory.
    pub half_life_secs: f64,
    /// How much each access slows decay, as an additive bonus to the
    /// decayed score (capped so a hot memory never exceeds its base).
    pub access_boost: f32,
}

impl Default for DecayModel {
    fn default() -> Self {
        Self {
            // 30 days — a reasonable default "working memory" horizon.
            half_life_secs: 30.0 * 24.0 * 3600.0,
            access_boost: 0.05,
        }
    }
}

impl DecayModel {
    /// Pure time-decay factor in `(0.0, 1.0]`: `0.5^(age / half_life)`.
    /// Age 0 → 1.0; one half-life → 0.5; etc.
    pub fn factor(&self, age_secs: u64) -> f32 {
        if self.half_life_secs <= 0.0 {
            return 1.0;
        }
        let exponent = age_secs as f64 / self.half_life_secs;
        0.5f64.powf(exponent) as f32
    }

    /// Effective importance = `base * decay_factor + access_reinforcement`,
    /// clamped to `[0.0, base]` (access can slow decay but never make a
    /// memory more important than it originally was).
    pub fn decayed(&self, base: f32, age_secs: u64, access_count: u32) -> f32 {
        let decayed = base * self.factor(age_secs);
        let reinforced = decayed + self.access_boost * (access_count as f32).min(10.0);
        reinforced.clamp(0.0, base)
    }
}

/// One memory's decay inputs, as the host would supply them.
#[derive(Debug, Clone, Copy)]
pub struct DecayInput<K> {
    pub key: K,
    pub base_importance: f32,
    pub age_secs: u64,
    pub access_count: u32,
}

/// Select the memories whose *effective* (decayed) importance has fallen
/// below `threshold` — the forgetting candidates. Returns `(key,
/// effective_score)` pairs, weakest first, so the host can retire up to
/// its budget (Hard Rule #6: forgetting is bounded too).
pub fn forgettable<K: Copy>(
    model: &DecayModel,
    items: &[DecayInput<K>],
    threshold: f32,
) -> Vec<(K, f32)> {
    let mut out: Vec<(K, f32)> = items
        .iter()
        .map(|it| {
            (
                it.key,
                model.decayed(it.base_importance, it.age_secs, it.access_count),
            )
        })
        .filter(|(_, score)| *score < threshold)
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_halves_at_half_life() {
        let m = DecayModel {
            half_life_secs: 100.0,
            access_boost: 0.0,
        };
        assert!((m.factor(0) - 1.0).abs() < 1e-6);
        assert!((m.factor(100) - 0.5).abs() < 1e-3);
        assert!((m.factor(200) - 0.25).abs() < 1e-3);
    }

    #[test]
    fn decayed_never_exceeds_base() {
        let m = DecayModel {
            half_life_secs: 100.0,
            access_boost: 0.5,
        };
        // Fresh + many accesses: reinforcement would overshoot, but it's
        // clamped to base.
        let d = m.decayed(0.6, 0, 10);
        assert!(d <= 0.6 + 1e-6, "decayed {d} must not exceed base 0.6");
    }

    #[test]
    fn access_slows_decay() {
        let m = DecayModel {
            half_life_secs: 100.0,
            access_boost: 0.05,
        };
        let cold = m.decayed(0.8, 100, 0);
        let hot = m.decayed(0.8, 100, 5);
        assert!(hot > cold, "accessed memory should decay slower");
    }

    #[test]
    fn forgettable_picks_weakest_below_threshold() {
        let m = DecayModel {
            half_life_secs: 100.0,
            access_boost: 0.0,
        };
        let items = vec![
            // fresh, important — keep
            DecayInput {
                key: 1usize,
                base_importance: 0.9,
                age_secs: 0,
                access_count: 0,
            },
            // old, was important — now decayed below threshold
            DecayInput {
                key: 2usize,
                base_importance: 0.9,
                age_secs: 400,
                access_count: 0,
            },
            // old + low base — weakest
            DecayInput {
                key: 3usize,
                base_importance: 0.4,
                age_secs: 400,
                access_count: 0,
            },
        ];
        let f = forgettable(&m, &items, 0.2);
        let keys: Vec<usize> = f.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
        assert!(!keys.contains(&1), "fresh important memory must be kept");
        // Weakest first.
        assert_eq!(f[0].0, 3);
    }

    #[test]
    fn zero_half_life_never_decays() {
        let m = DecayModel {
            half_life_secs: 0.0,
            access_boost: 0.0,
        };
        assert_eq!(m.factor(10_000), 1.0);
    }
}
