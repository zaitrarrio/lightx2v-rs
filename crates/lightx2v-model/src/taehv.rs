//! TAEHV decoder (madebyollin/taehv) — Phase 1 loads and validates the
//! checkpoint; the decode itself lands in Phase 2.
//!
//! The interface is real: `TaeHv::load` checks the positional
//! `decoder.N` manifest against the checkpoint (a missing key is an error),
//! and `decode` takes a denoised latent and returns RGB frames. Until the
//! conv/MemBlock kernels are ported, `decode` returns
//! [`Error::NotImplemented`] naming exactly what is missing, so no caller
//! can mistake a stub for a picture.

use crate::pipeline::Video;
use lightx2v_core::{
    backend::Backend,
    config::{GenerationConfig, TaeConfig},
    weights::{missing, tae_manifest, WeightSource},
    Error, Result,
};

pub struct TaeHv<B: Backend> {
    pub cfg: TaeConfig,
    // The loaded weights stay on the backend in Phase 2; Phase 1 only
    // validates the manifest, so nothing is stored yet.
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> TaeHv<B> {
    pub fn load(_be: &B, cfg: TaeConfig, src: &dyn WeightSource) -> Result<Self> {
        let manifest = tae_manifest(&cfg);
        let miss = missing(src, &manifest);
        if !miss.is_empty() {
            return Err(Error::Weights(format!(
                "TAEHV checkpoint is missing {} tensor(s), first: {}",
                miss.len(),
                miss[..miss.len().min(3)].join(", ")
            )));
        }
        Ok(Self {
            cfg,
            _marker: std::marker::PhantomData,
        })
    }

    /// Decode a denoised latent `[C, T, H, W]` to RGB frames.
    pub fn decode(&self, _be: &B, _latent: &[f32], gen: &GenerationConfig) -> Result<Video> {
        Err(Error::NotImplemented(format!(
            "TAEHV decode (taew2_1, {} latent frames -> {} video frames) is Phase 2 work: \
             the MemBlock conv kernels and chunked decode state are not ported yet. \
             The checkpoint validated; the latent is on the host.",
            gen.latent_frames,
            gen.video_frames()
        )))
    }

    /// Output frames for a latent count (`4T − 3`).
    pub fn out_frames(&self, latent_frames: usize) -> usize {
        latent_frames * self.cfg.t_upscale() - self.cfg.frames_to_trim()
    }
}
