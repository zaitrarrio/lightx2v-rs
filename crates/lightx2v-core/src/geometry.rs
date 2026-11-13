//! Clip geometry: pixel frames ↔ latent frames ↔ tokens.
//!
//! Wan latents are `4×` temporally compressed and `8×` spatially compressed:
//! `video_frames = 4 · latent_frames − 3` (the TAEHV priming frames), and a
//! latent grid is `(T, H/8, W/8)` before the spatial patch `p = 2`.

use crate::cfg_err;
use crate::{config::WanConfig, Result};

/// A clip in pixel space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Video frames (not latent frames).
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl Geometry {
    /// 832×480×81, the LightX2V / History-Forcing short-clip profile.
    pub const SHORT_480P: Geometry = Geometry {
        frames: 81,
        height: 480,
        width: 832,
    };

    /// The 60 s demo: 252 latent frames → 1005 video frames at 16 fps.
    pub const LONG_60S: Geometry = Geometry {
        frames: 1005,
        height: 480,
        width: 832,
    };

    /// `(latent_frames, latent_h, latent_w)`.
    pub fn latent(&self) -> (usize, usize, usize) {
        (self.frames.div_ceil(4), self.height / 8, self.width / 8)
    }

    /// Latent frames for a video-frame count.
    pub fn latent_frames(frames: usize) -> usize {
        frames.div_ceil(4)
    }

    /// Video frames for a latent-frame count.
    pub fn video_frames(latent_frames: usize) -> usize {
        latent_frames * 4 - 3
    }

    /// Token count after patchify with the config's patch size.
    pub fn tokens(&self, cfg: &WanConfig) -> usize {
        let (t, h, w) = self.latent();
        t * (h / cfg.patch) * (w / cfg.patch)
    }

    /// Token grid `(t, h/p, w/p)`.
    pub fn token_grid(&self, cfg: &WanConfig) -> (usize, usize, usize) {
        let (t, h, w) = self.latent();
        (t, h / cfg.patch, w / cfg.patch)
    }

    /// Flat latent length `C · T · H · W`.
    pub fn latent_len(&self, cfg: &WanConfig) -> usize {
        let (t, h, w) = self.latent();
        cfg.in_channels * t * h * w
    }

    /// Tokens in one 3-latent-frame autoregressive chunk.
    pub fn chunk_tokens(&self, cfg: &WanConfig, chunk_latent_frames: usize) -> usize {
        let (_, h, w) = self.latent();
        chunk_latent_frames * (h / cfg.patch) * (w / cfg.patch)
    }

    pub fn validate(&self, cfg: &WanConfig) -> Result<()> {
        let (t, h, w) = self.latent();
        if t == 0 || h == 0 || w == 0 {
            return Err(cfg_err("geometry collapses to an empty latent grid"));
        }
        if h % cfg.patch != 0 || w % cfg.patch != 0 {
            return Err(cfg_err(format!(
                "latent grid ({h}x{w}) is not divisible by patch {}",
                cfg.patch
            )));
        }
        if self.frames != Self::video_frames(t) {
            return Err(cfg_err(format!(
                "frames {} is not 4T-3 for any latent T (got T={t})",
                self.frames
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WanConfig;

    #[test]
    fn short_and_long_geometry() {
        let cfg = WanConfig::wan_1_3b();
        let g = Geometry::SHORT_480P;
        assert_eq!(g.latent(), (21, 60, 104));
        assert_eq!(g.tokens(&cfg), 21 * 30 * 52);
        assert_eq!(g.latent_len(&cfg), 16 * 21 * 60 * 104);
        assert_eq!(Geometry::LONG_60S.latent(), (252, 60, 104));
        assert_eq!(Geometry::video_frames(252), 1005);
        g.validate(&cfg).unwrap();
        Geometry::LONG_60S.validate(&cfg).unwrap();
        assert_eq!(g.chunk_tokens(&cfg, 3), 3 * 30 * 52);
    }
}
