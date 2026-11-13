//! 3-axis RoPE tables (Diffusers `WanRotaryPosEmbed`) plus block RoPE for
//! MemRoPE.
//!
//! Each head of width `d` is split `(t, h, w)` (44/42/42 at d = 128) and each
//! axis rotates interleaved pairs `(x[2i], x[2i+1])` by `pos · θ^(−2i/axis)`.
//! Both members of a pair share one angle, so the table stores `d/2` angles
//! per token instead of the reference's repeat-interleaved `d`.
//!
//! MemRoPE (`use_block_rope = true`) keeps each cached frame's *original*
//! temporal index, so a frame that has been compressed into an EMA slot still
//! rotates by the position it was generated at. [`rope_table_for_indices`]
//! builds a table for an arbitrary list of temporal indices — the cached
//! frames' indices — instead of a contiguous range.

use crate::config::WanConfig;

/// `cos`/`sin` of shape `[tokens, head_dim / 2]`, tokens in `(t, h, w)` order.
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

fn axis_freqs(dim: usize, theta: f64) -> Vec<f64> {
    (0..dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / dim as f64))
        .collect()
}

/// Table for a `(t, h, w)` token grid whose first latent frame is `start_frame`.
pub fn rope_table(cfg: &WanConfig, grid: (usize, usize, usize), start_frame: usize) -> RopeTable {
    let (gt, gh, gw) = grid;
    let indices: Vec<usize> = (start_frame..start_frame + gt).collect();
    rope_table_for_indices(cfg, (gh, gw), &indices)
}

/// Table for a token grid whose latent frames sit at the given temporal
/// indices (one index per frame, in token order). With `use_block_rope` the
/// indices are the frames' original positions, not their cache slots.
pub fn rope_table_for_indices(
    cfg: &WanConfig,
    grid_hw: (usize, usize),
    temporal_indices: &[usize],
) -> RopeTable {
    let (dt, dh, dw) = cfg.rope_split();
    let (ft, fh, fw) = (
        axis_freqs(dt, cfg.rope_theta),
        axis_freqs(dh, cfg.rope_theta),
        axis_freqs(dw, cfg.rope_theta),
    );
    let half = cfg.head_dim / 2;
    let (gh, gw) = grid_hw;
    let n = temporal_indices.len() * gh * gw;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for &t in temporal_indices {
        for h in 0..gh {
            for w in 0..gw {
                for (freqs, pos) in [(&ft, t), (&fh, h), (&fw, w)] {
                    for &f in freqs.iter() {
                        // Diffusers computes the outer product in float64 and
                        // casts the angle's cos/sin to float32.
                        let a = pos as f64 * f;
                        cos.push(a.cos() as f32);
                        sin.push(a.sin() as f32);
                    }
                }
            }
        }
    }
    RopeTable { cos, sin, half }
}

/// Rotate one interleaved-pair row of a head by the table row: for each pair
/// `(x[2i], x[2i+1])`, `x' = (x·cos − y·sin, x·sin + y·cos)`.
pub fn apply_rope_row(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    debug_assert_eq!(x.len(), 2 * cos.len());
    for i in 0..cos.len() {
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        let (c, s) = (cos[i], sin[i]);
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_identity_and_shapes_add_up() {
        let cfg = WanConfig::wan_1_3b();
        let t = rope_table(&cfg, (2, 3, 4), 0);
        assert_eq!(t.cos.len(), 24 * 64);
        assert!(t.cos[..64].iter().all(|&c| c == 1.0));
        assert!(t.sin[..64].iter().all(|&s| s == 0.0));
        // Token (t=1,0,0): the first temporal pair rotates by 1 rad.
        let row = 12 * 64;
        assert!((t.cos[row] - 1f32.cos()).abs() < 1e-7);
        // ...and the spatial pairs do not.
        assert_eq!(t.cos[row + 22], 1.0);
    }

    #[test]
    fn arbitrary_indices_match_contiguous() {
        let cfg = WanConfig::tiny();
        let a = rope_table(&cfg, (3, 2, 2), 5);
        let b = rope_table_for_indices(&cfg, (2, 2), &[5, 6, 7]);
        assert_eq!(a.cos, b.cos);
        assert_eq!(a.sin, b.sin);
        // A cached frame keeps its original index after eviction of earlier
        // frames: index 9 alone must equal the third row of a [7,8,9] table.
        let c = rope_table_for_indices(&cfg, (2, 2), &[9]);
        let d = rope_table_for_indices(&cfg, (2, 2), &[7, 8, 9]);
        let per_frame = 2 * 2 * (cfg.head_dim / 2);
        let row = 2 * per_frame;
        assert_eq!(&c.cos, &d.cos[row..row + per_frame]);
    }

    #[test]
    fn apply_rope_row_rotates_pairs() {
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0];
        let (c, s) = (0.0f32, 1.0f32); // 90 degrees for both pairs
        apply_rope_row(&mut x, &[c, c], &[s, s]);
        assert!((x[0] - 0.0).abs() < 1e-7);
        assert!((x[1] - 1.0).abs() < 1e-7);
        assert!((x[2] + 1.0).abs() < 1e-7);
        assert!((x[3] - 0.0).abs() < 1e-7);
    }
}
