//! Memory plan: what a generation keeps resident, for a given device,
//! before anything is allocated.
//!
//! Two knobs, as in fastwan-rs: the FFN token chunk (the only activation
//! that scales with `ffn_dim`) and the TAEHV frame chunk. Two additions for
//! LightX2V / History-Forcing:
//!
//! * the **MemRoPE KV cache**: `layers · cache_frames · frame_tokens · dim`
//!   K/V elements, bounded by `cache_frames` (12) no matter how long the
//!   rollout is — that bound is what makes a 60 s generation fit;
//! * **offload granularity**: `Model`/`Block`/`Disk` keep only part of the
//!   weights resident, at the cost of re-uploads the plan does not time.

use crate::{
    backend::OffloadGranularity,
    config::{GenerationConfig, MemRopeConfig, TaeConfig, WanConfig},
    geometry::Geometry,
};

#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub budget_bytes: u64,
    /// Largest single buffer binding the device allows.
    pub max_binding_bytes: u64,
    /// Bytes per stored weight element (2 for f16/bf16).
    pub weight_bytes: u64,
    /// Bytes per activation element (4 for f32, 2 for bf16).
    pub act_bytes: u64,
}

impl Device {
    pub fn gib(budget: f64, weight_bytes: u64, act_bytes: u64) -> Self {
        Self {
            budget_bytes: (budget * (1u64 << 30) as f64) as u64,
            max_binding_bytes: u64::MAX,
            weight_bytes,
            act_bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub tokens: usize,
    /// Latent frames in the rollout and 3-frame chunks.
    pub latent_frames: usize,
    pub chunks: usize,
    pub ffn_chunk: usize,
    pub tae_chunk: usize,
    pub offload: OffloadGranularity,
    /// Weights resident on the device (all of them, or one model / one block).
    pub weights_resident: u64,
    pub weights_total: u64,
    pub dit_activations: u64,
    pub cross_kv: u64,
    /// The MemRoPE KV cache, all layers, bounded by `cache_frames`.
    pub kv_cache: u64,
    pub decode_peak: u64,
    /// Largest single buffer the plan binds.
    pub largest_binding: u64,
    pub peak: u64,
    pub fits: bool,
    pub notes: Vec<String>,
}

/// Activation bytes of the DiT at `ffn_chunk`, excluding weights and the
/// cross-attention cache.
fn dit_acts(cfg: &WanConfig, s: u64, latent: u64, chunk: u64, a: u64) -> u64 {
    let d = cfg.dim() as u64;
    let elems = s * d // residual stream h
        + s * d // normed
        + s * 3 * d // fused qkv projection
        + 3 * s * d // q, k, v as [heads, S, d]
        + s * d // attention output (merged heads)
        + s * d // to_out / cross / ffn result
        + chunk * cfg.ffn_dim as u64 // FFN hidden, one chunk
        + s * cfg.patch_features_out().max(cfg.patch_features_in()) as u64
        + 3 * latent; // x, v, noise
    let rope = s * cfg.head_dim as u64 * 4; // cos+sin, f32, d/2 each
    elems * a + rope
}

/// Resident weights for an offload granularity: `None` keeps everything;
/// `Model` keeps the DiT resident while the text encoder / TAEHV are out;
/// `Block` keeps one transformer block plus the stem/head; `Disk` keeps
/// nothing (weights stream per op — the plan counts one block as the
/// working set).
fn resident_weights(cfg: &WanConfig, offload: OffloadGranularity, weight_bytes: u64) -> u64 {
    let total = cfg.params() as u64 * weight_bytes;
    match offload {
        OffloadGranularity::None | OffloadGranularity::Model => total,
        OffloadGranularity::Block | OffloadGranularity::Disk => {
            let per_block = total / cfg.layers.max(1) as u64;
            // Stem + head + one block, roughly.
            total / cfg.layers.max(1) as u64 + per_block
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plan(
    cfg: &WanConfig,
    tae: &TaeConfig,
    gen: &GenerationConfig,
    memrope: &MemRopeConfig,
    g: &Geometry,
    dev: &Device,
    offload: OffloadGranularity,
) -> Plan {
    let s = g.tokens(cfg) as u64;
    let latent = g.latent_len(cfg) as u64;
    let a = dev.act_bytes;
    let d = cfg.dim() as u64;
    let weights_total = cfg.params() as u64 * dev.weight_bytes;
    let weights_resident = resident_weights(cfg, offload, dev.weight_bytes);
    let cross_kv = cfg.layers as u64 * 2 * cfg.text_len as u64 * d * a;
    // MemRoPE KV cache: every layer caches cache_frames frames of K and V.
    let frame_tokens = g.chunk_tokens(cfg, 1) as u64;
    let kv_cache = cfg.layers as u64 * 2 * memrope.cache_frames as u64 * frame_tokens * d * a;
    let mut notes = Vec::new();
    if offload != OffloadGranularity::None {
        notes.push(format!(
            "offload {offload}: {weights_resident} of {weights_total} weight bytes resident"
        ));
    }
    notes.push(format!(
        "MemRoPE cache is bounded at {} frames ({} chunks of 3)",
        memrope.cache_frames,
        gen.latent_frames / 3
    ));

    // Largest FFN chunk (multiple of 256 tokens, or all of S) that fits both
    // the binding cap and the budget.
    let fixed = weights_resident + cross_kv + kv_cache;
    let mut chunk = s;
    let fits_chunk = |c: u64| {
        c * cfg.ffn_dim as u64 * a <= dev.max_binding_bytes
            && fixed + dit_acts(cfg, s, latent, c, a) <= dev.budget_bytes
    };
    while chunk > 256 && !fits_chunk(chunk) {
        chunk = (chunk / 2).div_ceil(256) * 256;
    }
    let dit_activations = dit_acts(cfg, s, latent, chunk, a);
    if chunk < s {
        notes.push(format!(
            "FFN runs in {} chunks of {chunk} tokens",
            s.div_ceil(chunk)
        ));
    }

    // TAEHV: the last stage runs at 4 frames per latent frame, 8× spatial,
    // `n_f[3]` channels; a MemBlock holds x, past, two temporaries and the
    // output. Choose the largest latent chunk that fits next to the weights
    // (the DiT activations are freed before decoding).
    let (t, lh, lw) = g.latent();
    let px = (lh * lw) as u64;
    let nf = tae.widths.map(|w| w as u64);
    let per_latent_peak = |c: u64| {
        let s0 = c * nf[0] * px;
        let s1 = c * tae.tgrow[0] as u64 * nf[1] * 4 * px;
        let s2 = c * (tae.tgrow[0] * tae.tgrow[1]) as u64 * nf[2] * 16 * px;
        let out = c * tae.t_upscale() as u64 * nf[3] * 64 * px;
        (5 * s0).max(5 * s1).max(5 * s2).max(s2 + 2 * out) * a
    };
    let mut tae_chunk = (t as u64).min(4);
    while tae_chunk > 1 && weights_resident + per_latent_peak(tae_chunk) > dev.budget_bytes {
        tae_chunk -= 1;
    }
    let decode_peak = per_latent_peak(tae_chunk);
    let largest_binding = (s * 3 * d * a)
        .max(chunk * cfg.ffn_dim as u64 * a)
        .max(per_latent_peak(tae_chunk) / 2);
    if largest_binding > dev.max_binding_bytes {
        notes.push(format!(
            "largest buffer {} MiB exceeds the device binding cap {} MiB",
            largest_binding >> 20,
            dev.max_binding_bytes >> 20
        ));
    }
    let peak = (fixed + dit_activations).max(weights_resident + decode_peak);
    Plan {
        tokens: s as usize,
        latent_frames: gen.latent_frames,
        chunks: gen.latent_frames / 3,
        ffn_chunk: chunk as usize,
        tae_chunk: tae_chunk as usize,
        offload,
        weights_resident,
        weights_total,
        dit_activations,
        cross_kv,
        kv_cache,
        decode_peak,
        largest_binding,
        peak,
        fits: peak <= dev.budget_bytes && largest_binding <= dev.max_binding_bytes,
        notes,
    }
}

impl std::fmt::Display for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        writeln!(f, "tokens          {}", self.tokens)?;
        writeln!(
            f,
            "latent frames   {} ({} chunks of 3)",
            self.latent_frames, self.chunks
        )?;
        writeln!(
            f,
            "weights         {:>9.1} MiB resident / {:>9.1} MiB total ({})",
            mib(self.weights_resident),
            mib(self.weights_total),
            self.offload
        )?;
        writeln!(f, "cross-attn K/V  {:>9.1} MiB", mib(self.cross_kv))?;
        writeln!(f, "MemRoPE KV cache{:>9.1} MiB", mib(self.kv_cache))?;
        writeln!(
            f,
            "DiT activations {:>9.1} MiB (FFN chunk {})",
            mib(self.dit_activations),
            self.ffn_chunk
        )?;
        writeln!(
            f,
            "decode peak     {:>9.1} MiB (TAEHV chunk {} latent frames)",
            mib(self.decode_peak),
            self.tae_chunk
        )?;
        writeln!(f, "largest buffer  {:>9.1} MiB", mib(self.largest_binding))?;
        writeln!(
            f,
            "peak            {:>9.1} MiB  fits: {}",
            mib(self.peak),
            self.fits
        )?;
        for n in &self.notes {
            writeln!(f, "note: {n}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::gib(16.0, 2, 2)
    }

    #[test]
    fn bf16_480p_short_fits_16gb() {
        let p = plan(
            &WanConfig::wan_1_3b(),
            &TaeConfig::taew2_1(),
            &GenerationConfig::short_clip(),
            &MemRopeConfig::default_memrope(),
            &Geometry::SHORT_480P,
            &dev(),
            OffloadGranularity::None,
        );
        assert!(p.fits, "{p}");
        assert_eq!(p.ffn_chunk, p.tokens);
        assert_eq!(p.chunks, 7);
    }

    #[test]
    fn long_rollout_cache_is_bounded() {
        let short = plan(
            &WanConfig::wan_1_3b(),
            &TaeConfig::taew2_1(),
            &GenerationConfig::short_clip(),
            &MemRopeConfig::default_memrope(),
            &Geometry::SHORT_480P,
            &dev(),
            OffloadGranularity::None,
        );
        let long = plan(
            &WanConfig::wan_1_3b(),
            &TaeConfig::taew2_1(),
            &GenerationConfig::demo(),
            &MemRopeConfig::default_memrope(),
            &Geometry::LONG_60S,
            &dev(),
            OffloadGranularity::None,
        );
        // 84 chunks instead of 7, but the KV cache is the same: bounded by
        // cache_frames, not by the rollout length.
        assert_eq!(long.chunks, 84);
        assert_eq!(long.kv_cache, short.kv_cache);
        // And the per-frame cache math: layers · 2 · 12 frames · tokens/frame · dim · 2B.
        let frame_tokens = Geometry::SHORT_480P.chunk_tokens(&WanConfig::wan_1_3b(), 1) as u64;
        assert_eq!(short.kv_cache, 30 * 2 * 12 * frame_tokens * 1536 * 2);
    }

    #[test]
    fn block_offload_shrinks_resident_weights() {
        let p = plan(
            &WanConfig::wan_1_3b(),
            &TaeConfig::taew2_1(),
            &GenerationConfig::short_clip(),
            &MemRopeConfig::default_memrope(),
            &Geometry::SHORT_480P,
            &dev(),
            OffloadGranularity::Block,
        );
        assert!(p.weights_resident < p.weights_total);
        assert!(p.weights_resident * 10 < p.weights_total);
    }
}
