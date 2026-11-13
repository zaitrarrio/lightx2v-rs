//! MemRoPE: the History-Forcing training-free long-video memory
//! (arXiv:2611.02782, `dmd/long_video/memrope.py` + `inference_memrope.py`).
//!
//! The KV cache holds at most `cache_frames` latent frames, laid out as
//!
//! ```text
//! [Sink 3] + [Long EMA 1] + [Short EMA 1] + [Recent 4] + [New 3]
//! ```
//!
//! * **Sink** frames are the first frames of the video, pinned forever
//!   (attention sinks).
//! * **Recent** frames are the newest history, kept verbatim.
//! * When the cache is full, the oldest recent frame is folded into the two
//!   EMA slots — `ema ← (1−α)·ema + α·frame` with `α_long = 0.01` and
//!   `α_short = 0.1` — and then dropped. The long EMA remembers the whole
//!   past faintly; the short EMA tracks the recent past.
//! * **New** frames are the 3-frame chunk currently being generated.
//! * The cache update runs **after each attention layer** (upstream
//!   `MemRoPEAttention.forward` calls `_apply_cache_updates` per layer), and
//!   the KV cache also carries each token's temporal and spatial indices so
//!   block RoPE can rotate a cached frame by its *original* position.
//!
//! This module is pure `Vec<f32>` math, so the layout, EMA update and
//! eviction are unit-testable without a backend. A backend's KV cache mirrors
//! it through [`crate::backend::Backend::kv_cache_memrope_update`].

use crate::config::MemRopeConfig;

/// Which slot of the cache a frame occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    Sink,
    LongEma,
    ShortEma,
    Recent,
    New,
}

/// The slot layout for a config, in cache order.
pub fn slot_kinds(cfg: &MemRopeConfig) -> Vec<SlotKind> {
    let mut kinds = Vec::with_capacity(cfg.cache_frames);
    kinds.extend(std::iter::repeat_n(SlotKind::Sink, cfg.sink_frames));
    kinds.extend(std::iter::repeat_n(
        SlotKind::LongEma,
        cfg.long_memory_frames,
    ));
    kinds.extend(std::iter::repeat_n(
        SlotKind::ShortEma,
        cfg.short_memory_frames,
    ));
    kinds.extend(std::iter::repeat_n(SlotKind::Recent, cfg.recent_frames));
    kinds.extend(std::iter::repeat_n(SlotKind::New, cfg.new_frames()));
    kinds
}

/// One EMA step: `ema ← (1−α)·ema + α·frame`.
pub fn ema_update(ema: &[f32], frame: &[f32], alpha: f32) -> Vec<f32> {
    debug_assert_eq!(ema.len(), frame.len());
    ema.iter()
        .zip(frame)
        .map(|(&e, &f)| (1.0 - alpha) * e + alpha * f)
        .collect()
}

/// One cached latent frame: its feature vector plus the position it was
/// generated at (block RoPE rotates by this, not by its cache slot).
#[derive(Debug, Clone)]
pub struct CachedFrame {
    pub temporal_index: usize,
    pub kind: SlotKind,
    /// One feature vector per token of the frame is stored flat by the
    /// backend; here a frame is one `Vec<f32>` (the caller chooses the
    /// flattening — e.g. K or V of the frame's tokens).
    pub data: Vec<f32>,
}

/// The MemRoPE cache as pure host-side math. A backend KV cache mirrors the
/// same slot/kind bookkeeping; see `Backend::kv_cache_memrope_update`.
#[derive(Debug, Clone)]
pub struct MemRopeCache {
    pub cfg: MemRopeConfig,
    pub frames: Vec<CachedFrame>,
}

impl MemRopeCache {
    pub fn new(cfg: MemRopeConfig) -> Self {
        Self {
            cfg,
            frames: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.cfg.cache_frames
    }

    /// Temporal indices of the cached frames, in cache order (what block
    /// RoPE needs).
    pub fn temporal_indices(&self) -> Vec<usize> {
        self.frames.iter().map(|f| f.temporal_index).collect()
    }

    /// Append a freshly generated frame. Frames are classified on arrival:
    /// the first `sink_frames` are sinks, the rest are new; once the cache
    /// is full, the oldest recent frame is folded into both EMA slots and
    /// evicted, and the oldest new frame graduates to recent.
    pub fn append(&mut self, temporal_index: usize, data: Vec<f32>) {
        // The EMA slots are part of the layout from the start: seed them
        // (zeros) right after the sinks on the first non-sink frame, so an
        // eviction never has to grow the cache to make room for them.
        let sink_count = self
            .frames
            .iter()
            .filter(|f| f.kind == SlotKind::Sink)
            .count();
        if sink_count >= self.cfg.sink_frames
            && !self.frames.iter().any(|f| f.kind == SlotKind::LongEma)
        {
            let insert_at = self
                .frames
                .iter()
                .position(|f| f.kind != SlotKind::Sink)
                .unwrap_or(self.frames.len());
            for kind in [SlotKind::ShortEma, SlotKind::LongEma] {
                self.frames.insert(
                    insert_at,
                    CachedFrame {
                        temporal_index,
                        kind,
                        data: vec![0.0; data.len()],
                    },
                );
            }
        }
        if self.frames.len() >= self.cfg.cache_frames {
            self.evict_oldest_recent();
        }
        let kind = if sink_count < self.cfg.sink_frames {
            SlotKind::Sink
        } else {
            SlotKind::New
        };
        self.frames.push(CachedFrame {
            temporal_index,
            kind,
            data,
        });
        self.graduate_new();
    }

    /// New frames beyond the 3-frame new window become recent.
    fn graduate_new(&mut self) {
        let new_count = self
            .frames
            .iter()
            .filter(|f| f.kind == SlotKind::New)
            .count();
        if new_count > self.cfg.new_frames() {
            let excess = new_count - self.cfg.new_frames();
            let mut graduated = 0;
            for f in self.frames.iter_mut() {
                if graduated == excess {
                    break;
                }
                if f.kind == SlotKind::New {
                    f.kind = SlotKind::Recent;
                    graduated += 1;
                }
            }
        }
    }

    /// Fold the oldest recent frame into the EMA slots and drop it. If there
    /// is no recent frame yet (early in the rollout), the oldest new frame
    /// past the sink is folded instead. The EMA slots are seeded on the
    /// first non-sink append (see [`MemRopeCache::append`]), so they always
    /// exist by the time an eviction runs.
    fn evict_oldest_recent(&mut self) {
        let evict_pos = self
            .frames
            .iter()
            .position(|f| f.kind == SlotKind::Recent)
            .or_else(|| self.frames.iter().position(|f| f.kind == SlotKind::New));
        let Some(pos) = evict_pos else {
            return; // only sinks (and EMA slots) present: nothing to evict
        };
        let evicted = self.frames.remove(pos);
        // Fold into both EMA slots.
        for slot in self.frames.iter_mut() {
            match slot.kind {
                SlotKind::LongEma => {
                    slot.data = ema_update(&slot.data, &evicted.data, self.cfg.ema_alpha_long);
                }
                SlotKind::ShortEma => {
                    slot.data = ema_update(&slot.data, &evicted.data, self.cfg.ema_alpha_short);
                }
                _ => {}
            }
        }
    }

    /// Causal block mask for attention from the new frames to the whole
    /// cache: rows are the new query frames, columns are cache frames plus
    /// the new frames themselves. A query frame attends to every cached
    /// frame and to new frames up to and including itself — never to later
    /// new frames (causal within the chunk).
    ///
    /// Returns a `[new_frames × (cache_frames + new_frames)]` row-major mask.
    pub fn causal_block_mask(cache_frames: usize, new_frames: usize) -> Vec<bool> {
        let mut mask = Vec::with_capacity(new_frames * (cache_frames + new_frames));
        for q in 0..new_frames {
            mask.extend(std::iter::repeat_n(true, cache_frames));
            for k in 0..new_frames {
                mask.push(k <= q);
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MemRopeConfig {
        MemRopeConfig::default_memrope()
    }

    #[test]
    fn layout_is_sink_long_short_recent_new() {
        let kinds = slot_kinds(&cfg());
        assert_eq!(kinds.len(), 12);
        assert_eq!(kinds[..3], [SlotKind::Sink; 3]);
        assert_eq!(kinds[3], SlotKind::LongEma);
        assert_eq!(kinds[4], SlotKind::ShortEma);
        assert_eq!(kinds[5..9], [SlotKind::Recent; 4]);
        assert_eq!(kinds[9..12], [SlotKind::New; 3]);
    }

    #[test]
    fn ema_update_math() {
        let ema = vec![1.0f32, 2.0, 3.0];
        let frame = vec![11.0f32, 12.0, 13.0];
        let out = ema_update(&ema, &frame, 0.1);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
        // alpha = 1 replaces; alpha -> 0 keeps.
        assert_eq!(ema_update(&ema, &frame, 1.0), frame);
        assert_eq!(ema_update(&ema, &frame, 0.0), ema);
    }

    #[test]
    fn cache_never_exceeds_capacity_and_keeps_sinks() {
        let mut cache = MemRopeCache::new(cfg());
        for t in 0..40 {
            cache.append(t, vec![t as f32; 4]);
            assert!(cache.len() <= 12, "len {} at t={t}", cache.len());
        }
        // The three sinks are frames 0, 1, 2, pinned forever.
        let sinks: Vec<usize> = cache
            .frames
            .iter()
            .filter(|f| f.kind == SlotKind::Sink)
            .map(|f| f.temporal_index)
            .collect();
        assert_eq!(sinks, vec![0, 1, 2]);
        // The newest frame is always present.
        assert_eq!(cache.frames.last().unwrap().temporal_index, 39);
        // Exactly one long and one short EMA slot.
        assert_eq!(
            cache
                .frames
                .iter()
                .filter(|f| f.kind == SlotKind::LongEma)
                .count(),
            1
        );
        assert_eq!(
            cache
                .frames
                .iter()
                .filter(|f| f.kind == SlotKind::ShortEma)
                .count(),
            1
        );
        // Temporal indices stay strictly increasing in cache order, ignoring
        // the EMA slots (both are seeded from the first non-sink frame, so
        // they share its index).
        let idx: Vec<usize> = cache
            .frames
            .iter()
            .filter(|f| !matches!(f.kind, SlotKind::LongEma | SlotKind::ShortEma))
            .map(|f| f.temporal_index)
            .collect();
        assert!(idx.windows(2).all(|w| w[0] < w[1]), "{idx:?}");
    }

    #[test]
    fn causal_block_mask_is_causal_within_the_chunk() {
        let m = MemRopeCache::causal_block_mask(9, 3);
        assert_eq!(m.len(), 3 * 12);
        // Every query attends to all 9 cached frames.
        assert!(m[0..9].iter().all(|&b| b));
        // Query 0 sees only new frame 0; query 2 sees all three new frames.
        assert_eq!(&m[9..12], &[true, false, false]);
        assert_eq!(&m[33..36], &[true, true, true]);
    }
}
