//! Prompt embeddings → causal DMD denoise → latent.
//!
//! The rollout is autoregressive over 3-latent-frame chunks
//! (`num_frame_per_block = 3`): chunk `i` is denoised with the 4-step DMD
//! schedule (`[1000, 750, 500, 250]`, shift 5.0) while attending to the
//! MemRoPE cache, then its latent is written out and the next chunk starts
//! from fresh noise. The first chunk uses the same step list
//! (`denoising_step_list_first_chunk = null` in the upstream config).
//!
//! Progress, in order: `Prepared`, one `Step` per DMD step of each chunk,
//! then `Chunk` once per finished chunk.

use crate::{dit::LightDit, taehv::TaeHv};
use lightx2v_core::{
    backend::Backend,
    config::{DistillConfig, GenerationConfig, WanConfig},
    geometry::Geometry,
    rng::Rng,
    schedule::DmdSchedule,
    Error, Result,
};
use std::time::Instant;

pub struct CausalPipeline<B: Backend> {
    pub dit: LightDit<B>,
    pub tae: TaeHv<B>,
    pub schedule: DmdSchedule,
    pub distill: DistillConfig,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub geometry: Geometry,
    pub seed: u64,
    /// Tokens per FFN chunk; `usize::MAX` = all.
    pub ffn_chunk: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub prompt_s: f64,
    pub steps_s: Vec<f64>,
    pub decode_s: f64,
}

impl Timings {
    pub fn denoise_s(&self) -> f64 {
        self.steps_s.iter().sum()
    }
}

/// Progress of a generation.
pub enum Event {
    Prepared {
        secs: f64,
    },
    Step {
        chunk: usize,
        index: usize,
        total: usize,
        secs: f64,
    },
    Chunk {
        index: usize,
        total: usize,
        secs: f64,
    },
    /// The finished latent is ready for the (Phase 2) decoder.
    Latent {
        frames: usize,
    },
}

/// `frames × height × width × RGB`, 8-bit — produced by the decoder in Phase 2.
#[derive(Debug, Clone)]
pub struct Video {
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl<B: Backend> CausalPipeline<B> {
    pub fn new(dit: LightDit<B>, tae: TaeHv<B>, distill: DistillConfig) -> Self {
        let schedule = DmdSchedule::new(&distill.denoising_step_list, distill.timestep_shift);
        Self {
            dit,
            tae,
            schedule,
            distill,
        }
    }

    /// Denoised latent `[C, T, H, W]` on the host, as f32.
    pub fn denoise(
        &self,
        be: &B,
        embeds: &[f32],
        gen: &GenerationConfig,
        opt: &Options,
        t: &mut Timings,
    ) -> Result<Vec<f32>> {
        self.denoise_with(be, embeds, gen, opt, t, &mut |_| {})
    }

    pub fn denoise_with(
        &self,
        be: &B,
        embeds: &[f32],
        gen: &GenerationConfig,
        opt: &Options,
        t: &mut Timings,
        on: &mut dyn FnMut(Event),
    ) -> Result<Vec<f32>> {
        gen.validate()?;
        self.distill.validate()?;
        let cfg = &self.dit.cfg;
        opt.geometry.validate(cfg)?;
        let (lt, lh, lw) = opt.geometry.latent();
        if lt != gen.latent_frames {
            return Err(Error::Config(format!(
                "geometry has {lt} latent frames, generation asks for {}",
                gen.latent_frames
            )));
        }
        let chunk_frames = self.distill.num_frame_per_block;
        if lt % chunk_frames != 0 {
            return Err(Error::Config(format!(
                "latent_frames {lt} is not a multiple of chunk size {chunk_frames}"
            )));
        }
        let chunks = lt / chunk_frames;
        let frame_tokens = (lh / cfg.patch) * (lw / cfg.patch);
        let chunk_tokens = chunk_frames * frame_tokens;
        let latent_len = opt.geometry.latent_len(cfg);
        let mut rng = Rng::new(opt.seed);

        let t0 = Instant::now();
        let prompt = self.dit.prepare_prompt(be, embeds);
        let mods: Vec<_> = self
            .schedule
            .timesteps
            .iter()
            .map(|&ts| self.dit.step_mods(be, ts as f32))
            .collect();
        // One workspace for the whole rollout: the caches persist across
        // chunks (that is the MemRoPE memory), only the RoPE table is
        // rebuilt per chunk inside `workspace_for_chunk`.
        let mut ws = self.dit.workspace(
            be,
            chunk_frames,
            (lh, lw),
            0,
            if opt.ffn_chunk == usize::MAX {
                chunk_tokens
            } else {
                opt.ffn_chunk
            },
        );
        be.sync();
        t.prompt_s = t0.elapsed().as_secs_f64();
        on(Event::Prepared { secs: t.prompt_s });

        let mut latent = vec![0.0f32; latent_len];
        let chunk_latent_len = cfg.in_channels * chunk_frames * lh * lw;
        for ci in 0..chunks {
            let ct0 = Instant::now();
            let start_frame = ci * chunk_frames;
            // Fresh noise for this chunk; the history comes only through
            // the MemRoPE cache.
            let x = be.upload(&rng.normal(chunk_latent_len));
            let v = be.alloc(chunk_latent_len);
            let noise = be.alloc(chunk_latent_len);
            // Rebuild the chunk's RoPE table (block RoPE: the chunk's own
            // temporal indices).
            let mut chunk_ws =
                self.dit
                    .workspace(be, chunk_frames, (lh, lw), start_frame, ws.ffn_chunk);
            // Move the persistent caches into the chunk workspace.
            std::mem::swap(&mut chunk_ws.caches, &mut ws.caches);
            let v_tokens = be.alloc(chunk_tokens * cfg.patch_features_out());
            for (i, m) in mods.iter().enumerate() {
                let st0 = Instant::now();
                // Patch-embed the chunk's current latent, run the DiT, and
                // project back to patch features — all per chunk.
                self.dit
                    .patch_embed(be, &chunk_ws, &x, cfg.in_channels, chunk_frames, lh, lw);
                // patch_embed wrote tokens into ws.h; forward_chunk reads
                // them from there and writes patch features to v_tokens.
                // (The DiT's `h` buffer is the chunk workspace's.)
                self.dit.forward_chunk(
                    be,
                    &mut chunk_ws,
                    m,
                    &prompt,
                    start_frame,
                    frame_tokens,
                    &v_tokens,
                );
                // v_tokens are patch features; unpatchify them into v (the
                // latent-shaped velocity) in place of the chunk's v buffer.
                be.unpatchify(
                    &v_tokens,
                    cfg.out_channels,
                    chunk_frames,
                    lh,
                    lw,
                    cfg.patch,
                    &v,
                );
                let (st, next) = self.schedule.coeffs(i);
                if next.is_some() {
                    be.write(&noise, &rng.normal(chunk_latent_len));
                }
                be.dmd_step(
                    &x,
                    &v,
                    next.map(|_| &noise),
                    st,
                    next.unwrap_or(0.0),
                    chunk_latent_len,
                );
                be.sync();
                t.steps_s.push(st0.elapsed().as_secs_f64());
                on(Event::Step {
                    chunk: ci,
                    index: i,
                    total: self.schedule.len(),
                    secs: t.steps_s.last().copied().unwrap_or(0.0),
                });
            }
            // Return the caches to the persistent workspace.
            std::mem::swap(&mut chunk_ws.caches, &mut ws.caches);
            let chunk_latent = be.download(&x);
            let off = ci * chunk_latent_len;
            latent[off..off + chunk_latent_len].copy_from_slice(&chunk_latent);
            on(Event::Chunk {
                index: ci,
                total: chunks,
                secs: ct0.elapsed().as_secs_f64(),
            });
        }
        on(Event::Latent {
            frames: gen.video_frames(),
        });
        Ok(latent)
    }

    /// Decode a denoised latent to RGB. Phase 1: the TAEHV weights load,
    /// the decode itself returns [`Error::NotImplemented`].
    pub fn decode(&self, be: &B, latent: &[f32], gen: &GenerationConfig) -> Result<Video> {
        self.tae.decode(be, latent, gen)
    }
}

/// The Wan config helper the CLI needs for a geometry check.
pub fn check_geometry(cfg: &WanConfig, g: &Geometry) -> Result<()> {
    g.validate(cfg)
}
