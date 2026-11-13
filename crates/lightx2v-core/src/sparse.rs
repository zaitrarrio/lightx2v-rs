//! Video Sparse Attention geometry (LightX2V sparse attention; the same
//! tiling fastvideo-rs / fastwan-rs use for VSA).
//!
//! 1. **Tile** the `(T, H, W)` token grid into `(4, 4, 4)` cubes, 64 slots
//!    each; a partial cube fills its first slots, the rest is padding.
//! 2. **Coarse**: mean-pool Q, K, V per tile and run dense attention over tiles.
//! 3. **Select** the top-k key tiles per query tile by those coarse scores
//!    (ties to the lower tile index).
//! 4. **Fine**: full-resolution attention from each query tile's tokens to
//!    the selected tiles' tokens only.
//! 5. **Combine**: `out = coarse · gate + fine`, where `gate` is the block's
//!    `to_gate_compress` output, split into heads.
//!
//! At sparsity 0.8 and 480p (624 tiles) each query tile reads 125 tiles: the
//! fine stage does a fifth of dense attention's work. The CPU backend
//! implements [`crate::backend::Backend::sparse_attention`] as dense
//! attention — the exact reference the sparse kernels are checked against;
//! a GPU backend skips the unselected tiles.

/// Upstream's tile: 4×4×4 = 64 tokens.
pub const TILE: (usize, usize, usize) = (4, 4, 4);
pub const TILE_ELEMS: usize = 64;
/// Slot value for padding in [`TilePlan::slot_src`].
pub const PAD: u32 = u32::MAX;
/// LightX2V / FastVideo default sparsity.
pub const DEFAULT_SPARSITY: f64 = 0.8;

/// Tiles to keep per query tile.
pub fn topk_for(sparsity: f64, num_tiles: usize) -> usize {
    (((1.0 - sparsity) * num_tiles as f64).ceil() as usize).clamp(1, num_tiles.max(1))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    pub grid: (usize, usize, usize),
    pub tiles: (usize, usize, usize),
    pub seq: usize,
    /// Source token of each padded slot, [`PAD`] for padding. `num_tiles · 64`.
    pub slot_src: Vec<u32>,
    /// Real tokens per tile.
    pub block_sizes: Vec<u32>,
}

impl TilePlan {
    pub fn new(grid: (usize, usize, usize)) -> Self {
        let (t, h, w) = grid;
        let (ts, hs, ws) = TILE;
        let tiles = (t.div_ceil(ts), h.div_ceil(hs), w.div_ceil(ws));
        let nb = tiles.0 * tiles.1 * tiles.2;
        let mut slot_src = vec![PAD; nb * TILE_ELEMS];
        let mut block_sizes = vec![0u32; nb];
        let mut tile = 0;
        for tt in 0..tiles.0 {
            for hh in 0..tiles.1 {
                for ww in 0..tiles.2 {
                    let mut slot = tile * TILE_ELEMS;
                    for z in tt * ts..((tt + 1) * ts).min(t) {
                        for y in hh * hs..((hh + 1) * hs).min(h) {
                            for x in ww * ws..((ww + 1) * ws).min(w) {
                                slot_src[slot] = ((z * h + y) * w + x) as u32;
                                slot += 1;
                            }
                        }
                    }
                    block_sizes[tile] = (slot - tile * TILE_ELEMS) as u32;
                    tile += 1;
                }
            }
        }
        Self {
            grid,
            tiles,
            seq: t * h * w,
            slot_src,
            block_sizes,
        }
    }

    pub fn num_tiles(&self) -> usize {
        self.block_sizes.len()
    }
}

/// What a backend needs to build its sparse-attention state for one token grid.
#[derive(Debug, Clone)]
pub struct SparseSpec {
    pub plan: TilePlan,
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub scale: f32,
}

impl SparseSpec {
    pub fn new(grid: (usize, usize, usize), heads: usize, head_dim: usize, sparsity: f64) -> Self {
        let plan = TilePlan::new(grid);
        let topk = topk_for(sparsity, plan.num_tiles());
        Self {
            plan,
            heads,
            head_dim,
            topk,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_480p_plan() {
        let p = TilePlan::new((21, 30, 52));
        assert_eq!(p.tiles, (6, 8, 13));
        assert_eq!(p.num_tiles(), 624);
        assert_eq!(topk_for(0.8, 624), 125);
        let mut seen = vec![false; p.seq];
        for &s in &p.slot_src {
            if s != PAD {
                assert!(!seen[s as usize]);
                seen[s as usize] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));
        assert_eq!(
            p.block_sizes.iter().map(|&b| b as usize).sum::<usize>(),
            p.seq
        );
        assert_eq!(p.block_sizes[p.num_tiles() - 1], 1 * 2 * 4);
    }
}
