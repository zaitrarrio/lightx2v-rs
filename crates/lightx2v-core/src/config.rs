//! Architecture and generation constants.
//!
//! Numbers come from two upstream sources, both kept under `research/` and
//! NOT part of this workspace:
//!
//! * LightX2V `configs/wan/wan_t2v.json` (base, 50 steps, CFG on, shift 8)
//!   and `configs/wan22/wan_moe_t2v_distill.json` (DMD2 distill: 4 steps
//!   `[1000, 750, 500, 250]`, `sample_shift` 5.0, `enable_cfg` false,
//!   `guide [4.0, 3.0]`, `boundary_step_index` 2, `cpu_offload` true,
//!   `offload_granularity` "model", high/low-noise checkpoints).
//! * History-Forcing `configs/long_video/inference.yaml` +
//!   `dmd/long_video/config.py` (the demo profile and the MemRoPE defaults;
//!   see `MemRopeConfig::validate` for the exact rules ported from
//!   `validate_memrope_options` / `load_settings` / `validate_model_config`).

use crate::{cfg_err, Result};
use serde::{Deserialize, Serialize};

/// Wan DiT architecture. Batch is always 1: the distilled models run without
/// classifier-free guidance, so a second batch row never exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WanConfig {
    /// Spatial patch (the temporal patch is 1 for every Wan checkpoint).
    pub patch: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    /// Prompt tokens the text encoder pads to.
    pub text_len: usize,
    pub freq_dim: usize,
    pub ffn_dim: usize,
    pub layers: usize,
    pub eps: f32,
    pub rope_max_seq_len: usize,
    pub rope_theta: f64,
}

impl WanConfig {
    pub fn dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn patch_features_out(&self) -> usize {
        self.patch * self.patch * self.out_channels
    }

    pub fn patch_features_in(&self) -> usize {
        self.patch * self.patch * self.in_channels
    }

    /// Wan 2.1 T2V 1.3B (also the shape of the DMD student in
    /// History-Forcing's `dmd_ar10s_bi1p3b` / `dmd_ar5s_bi1p3b` configs).
    pub fn wan_1_3b() -> Self {
        Self {
            patch: 2,
            heads: 12,
            head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            text_len: 512,
            freq_dim: 256,
            ffn_dim: 8960,
            layers: 30,
            eps: 1e-6,
            rope_max_seq_len: 1024,
            rope_theta: 10000.0,
        }
    }

    /// A graph small enough to run on the CPU reference in a unit test. Head
    /// dim stays a multiple of 6 so the 3-axis RoPE split is non-trivial.
    pub fn tiny() -> Self {
        Self {
            patch: 2,
            heads: 2,
            head_dim: 12,
            in_channels: 4,
            out_channels: 4,
            text_dim: 16,
            text_len: 8,
            freq_dim: 16,
            ffn_dim: 40,
            layers: 2,
            eps: 1e-6,
            rope_max_seq_len: 64,
            rope_theta: 10000.0,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "wan-1.3b" | "wan21-1.3b" | "Wan2.1-T2V-1.3B" => Some(Self::wan_1_3b()),
            "tiny" => Some(Self::tiny()),
            _ => None,
        }
    }

    /// RoPE split of one head: `(t, h, w)` widths. Diffusers
    /// `WanRotaryPosEmbed`: `h = w = 2·(d/6)`, `t = d − h − w` (44/42/42).
    pub fn rope_split(&self) -> (usize, usize, usize) {
        let hw = 2 * (self.head_dim / 6);
        (self.head_dim - 2 * hw, hw, hw)
    }

    /// Parameter count of the DiT (dense; VSA gates excluded).
    pub fn params(&self) -> usize {
        let d = self.dim();
        let lin = |i: usize, o: usize| i * o + o;
        let block = 4 * lin(d, d) // self q k v o
            + 4 * lin(d, d) // cross q k v o
            + 4 * d // norm_q/norm_k ×2
            + 2 * d // norm2
            + lin(d, self.ffn_dim)
            + lin(self.ffn_dim, d)
            + 6 * d;
        self.layers * block
            + lin(self.patch_features_in(), d)
            + lin(self.freq_dim, d)
            + lin(d, d)
            + lin(d, 6 * d)
            + lin(self.text_dim, d)
            + lin(d, d)
            + lin(d, self.patch_features_out())
            + 2 * d
    }
}

/// TAEHV decoder (madebyollin/taehv) for Wan latents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaeConfig {
    pub latent_channels: usize,
    pub widths: [usize; 4],
    pub tgrow: [usize; 3],
    pub mem_blocks: usize,
    /// Pixel-shuffle factor after the last conv (1 for taew2_1, 2 for taew2_2).
    pub patch: usize,
}

impl TaeConfig {
    pub fn taew2_1() -> Self {
        Self {
            latent_channels: 16,
            widths: [256, 128, 64, 64],
            tgrow: [1, 2, 2],
            mem_blocks: 3,
            patch: 1,
        }
    }

    pub fn tiny() -> Self {
        Self {
            latent_channels: 4,
            widths: [8, 6, 4, 4],
            tgrow: [1, 2, 2],
            mem_blocks: 3,
            patch: 1,
        }
    }

    /// Output frames per latent frame.
    pub fn t_upscale(&self) -> usize {
        self.tgrow.iter().product()
    }

    /// Priming frames emitted first and trimmed: `4T − 3` frames for Wan.
    pub fn frames_to_trim(&self) -> usize {
        self.t_upscale() - 1
    }

    pub fn spatial_scale(&self) -> usize {
        8 * self.patch
    }

    pub fn out_channels(&self) -> usize {
        3 * self.patch * self.patch
    }
}

/// Generation settings. The validation rules are a port of History-Forcing's
/// `load_settings` in `dmd/long_video/config.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Latent frames. Must be a positive multiple of 3 (one chunk per block).
    pub latent_frames: usize,
    pub height: usize,
    pub width: usize,
    pub fps: usize,
    pub batch_size: usize,
    /// Only `"bfloat16"` is supported by the demo profile.
    pub dtype: String,
    /// `"ema"` or `"raw"` weights.
    pub weights: String,
    pub seed: u32,
    pub num_threads: usize,
    /// x264 CRF, 0..=51.
    pub crf: usize,
}

impl GenerationConfig {
    /// The History-Forcing long-video demo profile: 252 latent frames
    /// (1005 video frames ≈ 63 s at 16 fps), 832×480, bf16, batch 1, seed 0.
    pub fn demo() -> Self {
        Self {
            latent_frames: 252,
            height: 480,
            width: 832,
            fps: 16,
            batch_size: 1,
            dtype: "bfloat16".into(),
            weights: "ema".into(),
            seed: 0,
            num_threads: 4,
            crf: 23,
        }
    }

    /// A short clip: 21 latent frames → 81 video frames, same profile.
    pub fn short_clip() -> Self {
        Self {
            latent_frames: 21,
            ..Self::demo()
        }
    }

    /// Video frames this latent count decodes to (`4T − 3` for Wan/TAEHV).
    pub fn video_frames(&self) -> usize {
        self.latent_frames * 4 - 3
    }

    /// Duration in seconds at `fps`.
    pub fn seconds(&self) -> f64 {
        self.video_frames() as f64 / self.fps as f64
    }

    pub fn validate(&self) -> Result<()> {
        if self.latent_frames == 0 || !self.latent_frames.is_multiple_of(3) {
            return Err(cfg_err("latent_frames must be a positive multiple of 3"));
        }
        if (self.height, self.width, self.fps, self.batch_size) != (480, 832, 16, 1)
            || self.dtype != "bfloat16"
        {
            return Err(cfg_err(
                "this demo profile uses 832x480, 16 FPS, batch 1 and bfloat16",
            ));
        }
        if self.weights != "ema" && self.weights != "raw" {
            return Err(cfg_err("weights must be \"ema\" or \"raw\""));
        }
        if self.num_threads == 0 {
            return Err(cfg_err("num_threads must be positive"));
        }
        if self.crf > 51 {
            return Err(cfg_err("crf must be in [0, 51]"));
        }
        Ok(())
    }
}

/// MemRoPE memory options. Port of `DEFAULT_MEMROPE_OPTIONS` and
/// `validate_memrope_options` in History-Forcing `dmd/long_video/config.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemRopeConfig {
    /// Total cache frames: sink + long + short + recent + 3 new.
    pub cache_frames: usize,
    pub sink_frames: usize,
    pub recent_frames: usize,
    pub long_memory_frames: usize,
    pub short_memory_frames: usize,
    /// Only `"ema"` is supported.
    pub compression_method: String,
    pub ema_alpha_long: f32,
    pub ema_alpha_short: f32,
    pub ema_adaptive: bool,
    pub use_block_rope: bool,
}

impl MemRopeConfig {
    /// `[Sink 3] + [Long EMA 1] + [Short EMA 1] + [Recent 4] + [New 3]`.
    pub fn default_memrope() -> Self {
        Self {
            cache_frames: 12,
            sink_frames: 3,
            recent_frames: 4,
            long_memory_frames: 1,
            short_memory_frames: 1,
            compression_method: "ema".into(),
            ema_alpha_long: 0.01,
            ema_alpha_short: 0.1,
            ema_adaptive: false,
            use_block_rope: true,
        }
    }

    /// New frames appended per autoregressive chunk (`num_frame_per_block`).
    pub fn new_frames(&self) -> usize {
        3
    }

    pub fn validate(&self) -> Result<()> {
        if self.long_memory_frames != 1 || self.short_memory_frames != 1 {
            return Err(cfg_err(
                "EMA memory uses one long and one short memory frame",
            ));
        }
        let expected = self.sink_frames
            + self.recent_frames
            + self.long_memory_frames
            + self.short_memory_frames
            + self.new_frames();
        if self.cache_frames != expected {
            return Err(cfg_err(
                "cache_frames must equal sink + long memory + short memory + recent + 3 new frames",
            ));
        }
        if self.compression_method != "ema" {
            return Err(cfg_err("compression_method must be ema"));
        }
        for (name, a) in [
            ("ema_alpha_long", self.ema_alpha_long),
            ("ema_alpha_short", self.ema_alpha_short),
        ] {
            if !(a.is_finite() && a > 0.0 && a <= 1.0) {
                return Err(cfg_err(format!("{name} must be finite and in (0, 1]")));
            }
        }
        Ok(())
    }
}

/// The few-step distillation settings. Port of `validate_model_config` in
/// History-Forcing `dmd/long_video/config.py` plus the LightX2V distill
/// config `configs/wan22/wan_moe_t2v_distill.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistillConfig {
    /// Denoising timesteps, in order. DMD2: `[1000, 750, 500, 250]`.
    pub denoising_step_list: Vec<i32>,
    /// `None`: the first chunk uses the same list (`denoising_step_list_first_chunk = null`).
    pub denoising_step_list_first_chunk: Option<Vec<i32>>,
    /// Flow-match timestep shift (`sample_shift` / `model_kwargs.timestep_shift`).
    pub timestep_shift: f64,
    pub infer_steps: usize,
    pub enable_cfg: bool,
    /// LightX2V distill guide scales `[4.0, 3.0]` (high/low-noise models).
    pub guide: Vec<f32>,
    /// LightX2V `boundary_step_index`: steps before it use the high-noise
    /// checkpoint, after it the low-noise checkpoint.
    pub boundary_step_index: usize,
    /// Latent frames per autoregressive chunk.
    pub num_frame_per_block: usize,
    pub warp_denoising_step: bool,
    pub independent_first_frame: bool,
    pub context_noise: i32,
    /// LightX2V distill method name, e.g. `"dmd2"`.
    pub distill_method: String,
}

impl DistillConfig {
    /// History-Forcing DMD student on Wan2.1-T2V-1.3B + LightX2V DMD2
    /// distill numbers (they agree on the step list and shift).
    pub fn dmd2() -> Self {
        Self {
            denoising_step_list: vec![1000, 750, 500, 250],
            denoising_step_list_first_chunk: None,
            timestep_shift: 5.0,
            infer_steps: 4,
            enable_cfg: false,
            guide: vec![4.0, 3.0],
            boundary_step_index: 2,
            num_frame_per_block: 3,
            warp_denoising_step: true,
            independent_first_frame: false,
            context_noise: 0,
            distill_method: "dmd2".into(),
        }
    }

    /// The undistilled LightX2V base profile (`configs/wan/wan_t2v.json`):
    /// 50 steps, CFG on, guide 6, shift 8. Kept for reference; the
    /// History-Forcing path always uses [`DistillConfig::dmd2`].
    pub fn base_wan_t2v() -> Self {
        Self {
            denoising_step_list: (0..50).map(|i| 1000 - i * 20).collect(),
            denoising_step_list_first_chunk: None,
            timestep_shift: 8.0,
            infer_steps: 50,
            enable_cfg: true,
            guide: vec![6.0],
            boundary_step_index: 0,
            num_frame_per_block: 3,
            warp_denoising_step: false,
            independent_first_frame: false,
            context_noise: 0,
            distill_method: "none".into(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_frame_per_block != 3 {
            return Err(cfg_err("set num_frame_per_block=3"));
        }
        if !self.warp_denoising_step {
            return Err(cfg_err("set warp_denoising_step=true"));
        }
        if self.independent_first_frame {
            return Err(cfg_err("set independent_first_frame=false"));
        }
        if self.context_noise != 0 {
            return Err(cfg_err("set context_noise=0"));
        }
        if self.denoising_step_list != [1000, 750, 500, 250] {
            return Err(cfg_err("set denoising_step_list=[1000, 750, 500, 250]"));
        }
        if self.denoising_step_list_first_chunk.is_some() {
            return Err(cfg_err("set denoising_step_list_first_chunk=null"));
        }
        if self.timestep_shift != 5.0 {
            return Err(cfg_err("set model_kwargs.timestep_shift=5.0"));
        }
        if self.infer_steps != self.denoising_step_list.len() {
            return Err(cfg_err("infer_steps must equal len(denoising_step_list)"));
        }
        if self.enable_cfg {
            return Err(cfg_err("enable_cfg must be false for the DMD student"));
        }
        Ok(())
    }

    /// Which checkpoint a step uses: steps before `boundary_step_index` use
    /// the high-noise model, the rest the low-noise model.
    pub fn checkpoint_for_step(&self, step: usize) -> &'static str {
        if step < self.boundary_step_index {
            "high-noise"
        } else {
            "low-noise"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_profile_validates() {
        GenerationConfig::demo().validate().unwrap();
        GenerationConfig::short_clip().validate().unwrap();
        assert_eq!(GenerationConfig::demo().video_frames(), 1005);
        assert_eq!(GenerationConfig::short_clip().video_frames(), 81);
        MemRopeConfig::default_memrope().validate().unwrap();
        DistillConfig::dmd2().validate().unwrap();
    }

    #[test]
    fn memrope_cache_rule_is_exact() {
        let mut m = MemRopeConfig::default_memrope();
        m.cache_frames = 11;
        assert!(m.validate().is_err());
        m.cache_frames = 12;
        m.long_memory_frames = 2;
        assert!(m.validate().is_err());
        m.long_memory_frames = 1;
        m.ema_alpha_long = 0.0;
        assert!(m.validate().is_err());
        m.ema_alpha_long = 0.01;
        m.validate().unwrap();
    }

    #[test]
    fn generation_rules() {
        let mut g = GenerationConfig::demo();
        g.latent_frames = 250;
        assert!(g.validate().is_err());
        g.latent_frames = 252;
        g.crf = 52;
        assert!(g.validate().is_err());
        g.crf = 23;
        g.width = 640;
        assert!(g.validate().is_err());
    }

    #[test]
    fn distill_rules_and_boundary() {
        let d = DistillConfig::dmd2();
        assert_eq!(d.checkpoint_for_step(0), "high-noise");
        assert_eq!(d.checkpoint_for_step(1), "high-noise");
        assert_eq!(d.checkpoint_for_step(2), "low-noise");
        assert_eq!(d.checkpoint_for_step(3), "low-noise");
        let mut bad = d.clone();
        bad.timestep_shift = 8.0;
        assert!(bad.validate().is_err());
        let mut bad = d.clone();
        bad.enable_cfg = true;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn wan_1_3b_param_count() {
        let p = WanConfig::wan_1_3b().params();
        assert!((1_400_000_000..1_440_000_000).contains(&p), "{p}");
        assert_eq!(WanConfig::wan_1_3b().rope_split(), (44, 42, 42));
        assert_eq!(WanConfig::tiny().rope_split(), (4, 4, 4));
    }
}
