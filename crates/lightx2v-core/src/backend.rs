//! The op vocabulary every backend implements, extended for LightX2V /
//! History-Forcing over the fastwan-rs vocabulary it mirrors.
//!
//! It is deliberately *fused* and deliberately small: each op is one kernel
//! on a GPU, and the set is exactly what a 4-step DMD chunk plus a MemRoPE
//! cache update needs — nothing a general tensor library would add. Shapes
//! are passed as integers, never stored; buffers are flat and contiguous.
//!
//! Activation precision is the backend's business (`f32` on CPU and WebGPU,
//! `bf16` on CUDA): [`Backend::upload`] / [`Backend::download`] convert at
//! the boundary and nothing above this trait sees a dtype.
//!
//! Aliasing rule: an output buffer never aliases an input, except the
//! arguments documented as *in place*.
//!
//! Extensions over the fastwan vocabulary:
//!
//! * [`Backend::causal_attention`] — softmax attention with a row-major
//!   boolean block mask (MemRoPE: new frames attend to the cache +
//!   causally within the chunk).
//! * [`Backend::kv_cache_new`] / `kv_cache_append` / `kv_cache_memrope_update`
//!   / `kv_cache_kv` / `kv_cache_len` — the per-layer KV cache, whose slots
//!   carry temporal indices for block RoPE and whose full-cache update folds
//!   the oldest recent frame into the EMA slots.
//! * [`Backend::sparse_attention`] — LightX2V sparse attention
//!   ([`crate::sparse::SparseSpec`]); the CPU reference computes it densely.
//! * [`Backend::weight_fp8`] — FP8 e4m3 weight-only matrices.
//! * [`OffloadGranularity`] — LightX2V block offload (`cpu_offload`,
//!   `offload_granularity` in `wan_moe_t2v_distill.json`).

use crate::{
    config::MemRopeConfig,
    quant::{Fp8E4M3, Q4, Q8},
    sparse::SparseSpec,
};

/// Epilogue activation of [`Backend::linear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` (PyTorch `gelu(approximate="tanh")`).
    GeluTanh,
}

/// The norm half of a fused residual + LayerNorm.
#[derive(Debug, Clone, Copy)]
pub enum NormKind<'a, T> {
    /// `LN(x) · w + b` (Wan `norm2`, the cross-attention norm).
    Affine { w: &'a T, b: &'a T },
    /// `LN(x) · (1 + mods[scale]) + mods[shift]` (AdaLN, no affine).
    Mod { scale: usize, shift: usize },
}

/// Epilogue of [`Backend::conv2d`], after the bias.
#[derive(Debug, Clone, Copy)]
pub enum ConvEpilogue<'a, T> {
    None,
    Relu,
    /// `relu(y + r)` with `r` shaped like the output (TAEHV MemBlock skip).
    AddRelu(&'a T),
    /// PReLU: `y ≥ 0 ? y : y · slope[c]`, one slope per output channel.
    Prelu(&'a T),
}

/// A 2-D convolution over `frames` independent images, NCHW, stride 1,
/// "same" padding (`k / 2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    pub frames: usize,
    /// Channels read from the first input.
    pub cin_a: usize,
    /// Channels read from the optional second input, concatenated after the
    /// first (`cat([a, b], dim=1)` without materialising it).
    pub cin_b: usize,
    pub cout: usize,
    /// Input height / width as stored.
    pub h: usize,
    pub w: usize,
    /// Read the inputs through a nearest ×2 upsample (output is `2h × 2w`),
    /// without materialising the upsampled tensor.
    pub upsample: bool,
    /// 1 or 3.
    pub k: usize,
}

impl ConvShape {
    pub fn cin(&self) -> usize {
        self.cin_a + self.cin_b
    }
    pub fn out_hw(&self) -> (usize, usize) {
        let s = if self.upsample { 2 } else { 1 };
        (self.h * s, self.w * s)
    }
    pub fn out_len(&self) -> usize {
        let (h, w) = self.out_hw();
        self.frames * self.cout * h * w
    }
    /// Reduction length of the implicit GEMM (`cin · k²`).
    pub fn k_len(&self) -> usize {
        self.cin() * self.k * self.k
    }
}

/// LightX2V offload granularity (`offload_granularity` in the distill
/// config): which weights stay resident on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadGranularity {
    /// Everything resident (no offload).
    #[default]
    None,
    /// One whole model (DiT / TAEHV / text encoder) resident at a time.
    Model,
    /// One transformer block resident at a time.
    Block,
    /// Weights stream from disk per op.
    Disk,
}

impl std::str::FromStr for OffloadGranularity {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "model" => Ok(Self::Model),
            "block" => Ok(Self::Block),
            "disk" => Ok(Self::Disk),
            _ => Err(format!(
                "unknown offload granularity `{s}` (none | model | block | disk)"
            )),
        }
    }
}

impl std::fmt::Display for OffloadGranularity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Model => "model",
            Self::Block => "block",
            Self::Disk => "disk",
        })
    }
}

pub trait Backend {
    /// A flat device buffer of activations.
    type Buf;
    /// Device state of [`Backend::sparse_attention`] for one token grid:
    /// tile tables and the coarse-stage scratch.
    type Sparse;
    /// A `[rows, cols]` row-major matrix (`nn.Linear` layout: `[out, in]`) in
    /// the backend's storage precision.
    type Weight;
    /// One layer's KV cache: K/V rows plus each row's temporal index
    /// (block RoPE) and slot kind (MemRoPE).
    type KvCache;

    fn name(&self) -> String;

    // ---- memory ----------------------------------------------------------
    /// Zero-filled.
    fn alloc(&self, len: usize) -> Self::Buf;
    fn upload(&self, data: &[f32]) -> Self::Buf;
    /// A lookup table the kernels read in full precision even on a backend
    /// whose activations are narrower (the RoPE cos/sin: bf16 has 8 bits of
    /// mantissa, too few for rotation angles). Only [`Backend::head_split`]
    /// reads these.
    fn upload_table(&self, data: &[f32]) -> Self::Buf {
        self.upload(data)
    }
    /// Overwrite the first `data.len()` elements.
    fn write(&self, buf: &Self::Buf, data: &[f32]);
    /// Blocking read of the whole buffer (native targets).
    fn download(&self, buf: &Self::Buf) -> Vec<f32>;
    fn buf_len(&self, buf: &Self::Buf) -> usize;
    /// `dst[dst_off..][..len] = src[src_off..][..len]`.
    fn copy(&self, src: &Self::Buf, src_off: usize, dst: &Self::Buf, dst_off: usize, len: usize);
    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> Self::Weight;
    /// An int8 weight-only matrix for [`Backend::linear`]. The result must
    /// equal `linear` with `weight(&q.dequantize())` up to the backend's own
    /// activation rounding.
    fn weight_q8(&self, q: &Q8) -> Self::Weight;
    /// A 4-bit weight-only matrix: the same contract as [`Backend::weight_q8`]
    /// with `weight(&q.dequantize())`.
    fn weight_q4(&self, q: &Q4) -> Self::Weight;
    /// An FP8 e4m3 weight-only matrix: the same contract with
    /// `weight(&q.dequantize())`.
    fn weight_fp8(&self, q: &Fp8E4M3) -> Self::Weight;
    /// Wait for queued work (timing only; downloads synchronise themselves).
    fn sync(&self);

    /// Per-kernel GPU time since the last [`Backend::profile_reset`].
    fn profile_report(&self) -> Option<String> {
        None
    }
    fn profile_reset(&self) {}

    // ---- transformer -----------------------------------------------------
    /// `y[y_row + i, :] = act(x[x_row + i, :] · Wᵀ + b)` for `i < rows`.
    /// Row offsets let the FFN run in token chunks.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: &Self::Buf,
        x_row: usize,
        rows: usize,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        act: Act,
        y: &Self::Buf,
        y_row: usize,
    );

    /// `out = norm(x)` over rows of width `dim`. `mods` is the `[slots, dim]`
    /// modulation table ([`NormKind::Mod`] reads it).
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        x: &Self::Buf,
        rows: usize,
        dim: usize,
        norm: NormKind<'_, Self::Buf>,
        mods: &Self::Buf,
        eps: f32,
        out: &Self::Buf,
    );

    /// In place on `h`: `h += a · mods[gate]` (or `h += a` without a gate),
    /// then `normed = norm(h)`. The norm reads the unrounded sum.
    #[allow(clippy::too_many_arguments)]
    fn residual_norm(
        &self,
        h: &Self::Buf,
        a: &Self::Buf,
        rows: usize,
        dim: usize,
        mods: &Self::Buf,
        gate: Option<usize>,
        norm: NormKind<'_, Self::Buf>,
        eps: f32,
        normed: &Self::Buf,
    );

    /// In place on `h`: `h += a · mods[gate]`.
    fn gate_residual(
        &self,
        h: &Self::Buf,
        a: &Self::Buf,
        rows: usize,
        dim: usize,
        mods: &Self::Buf,
        gate: usize,
    );

    /// Columns `[col_off, col_off + heads·head_dim)` of `src` (`rows` rows of
    /// width `src_stride`) → `out` as `[heads, rows, head_dim]`, optionally
    /// RMS-normalised across all heads (`x · rsqrt(mean(x²) + eps) · w`) and
    /// rotated by RoPE (`cos`/`sin` are `[rows, head_dim / 2]`, interleaved
    /// pairs).
    #[allow(clippy::too_many_arguments)]
    fn head_split(
        &self,
        src: &Self::Buf,
        rows: usize,
        src_stride: usize,
        col_off: usize,
        heads: usize,
        head_dim: usize,
        rms_w: Option<&Self::Buf>,
        rope: Option<(&Self::Buf, &Self::Buf)>,
        eps: f32,
        out: &Self::Buf,
    );

    /// Softmax attention, scale `1/√head_dim`, no mask. `q` is
    /// `[heads, sq, d]`, `k`/`v` `[heads, sk, d]`; `out` is written with heads
    /// merged, `[sq, heads·d]`, ready for `to_out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        out: &Self::Buf,
    );

    /// Softmax attention with a block mask: `mask` is `[sq, sk]` row-major,
    /// `false` = forbidden (score −∞). This is the MemRoPE attention: the
    /// new chunk's queries attend to the cached frames and causally within
    /// the chunk. `None` behaves exactly like [`Backend::attention`].
    #[allow(clippy::too_many_arguments)]
    fn causal_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        mask: Option<&[bool]>,
        out: &Self::Buf,
    );

    /// Whether [`Backend::sparse_attention`] can run at this head dim on this
    /// device.
    fn sparse_supported(&self, _head_dim: usize) -> bool {
        true
    }

    /// Build the sparse-attention state for one token grid
    /// ([`SparseSpec`]).
    fn sparse_prepare(&self, spec: &SparseSpec) -> Self::Sparse;

    /// Sparse attention ([`crate::sparse`]). `q`, `k`, `v`, `gate` are
    /// `[heads, seq, d]`; `out` is `[seq, heads·d]`, as from
    /// [`Backend::attention`]. The CPU reference computes this densely;
    /// a GPU backend skips the unselected tiles.
    #[allow(clippy::too_many_arguments)]
    fn sparse_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        gate: &Self::Buf,
        sparse: &Self::Sparse,
        out: &Self::Buf,
    );

    // ---- KV cache / MemRoPE ----------------------------------------------
    /// An empty per-layer KV cache holding up to `capacity_tokens` rows of
    /// `[heads, head_dim]` K/V.
    fn kv_cache_new(&self, heads: usize, head_dim: usize, capacity_tokens: usize) -> Self::KvCache;

    /// Append `rows` K/V rows (`k`/`v` are `[heads, rows, head_dim]`) with
    /// one temporal index per row (block RoPE rotates by these, not by the
    /// cache slot). Rows are appended as `New` slots; the caller runs
    /// [`Backend::kv_cache_memrope_update`] to fold/evict.
    fn kv_cache_append(
        &self,
        cache: &mut Self::KvCache,
        k: &Self::Buf,
        v: &Self::Buf,
        rows: usize,
        temporal: &[usize],
    );

    /// Write `rows` K/V rows with the fastvideo-rs causal discipline
    /// (`fastvideo-cudarc/src/wan/causal.rs`): appended when the block is
    /// new, **written over the block's own slots when the same block is run
    /// again** (the next denoising step of the same chunk, then the
    /// clean-context pass). Without this, a 4-step chunk would occupy the
    /// cache four times over and evict real history. The overwrite keeps
    /// the rows' slot kinds; only the K/V values change.
    fn kv_cache_put(
        &self,
        cache: &mut Self::KvCache,
        k: &Self::Buf,
        v: &Self::Buf,
        rows: usize,
        temporal: &[usize],
    );

    /// Number of rows currently cached.
    fn kv_cache_len(&self, cache: &Self::KvCache) -> usize;

    /// Temporal indices of the cached rows, in cache order.
    fn kv_cache_temporal(&self, cache: &Self::KvCache) -> Vec<usize>;

    /// The cached K and V as `[heads, rows, head_dim]` buffers, in cache order.
    fn kv_cache_kv(&self, cache: &Self::KvCache) -> (Self::Buf, Self::Buf);

    /// MemRoPE cache update, applied after an attention layer: while the
    /// cache exceeds `cfg.cache_frames · frame_tokens` rows, fold the oldest
    /// recent frame into the long/short EMA slots (`ema_alpha_long` /
    /// `ema_alpha_short`) and evict it; sink frames are never evicted.
    /// `frame_tokens` is the token count of one latent frame.
    fn kv_cache_memrope_update(
        &self,
        cache: &mut Self::KvCache,
        cfg: &MemRopeConfig,
        frame_tokens: usize,
    );

    // ---- patchify / sampler / decoder ------------------------------------
    /// Latent `[c, t, h, w]` → tokens `[t·(h/p)·(w/p), c·p²]`, feature index
    /// `c·p² + kh·p + kw` (the Conv3d `(1, p, p)` weight flattened).
    #[allow(clippy::too_many_arguments)]
    fn patchify(
        &self,
        latent: &Self::Buf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &Self::Buf,
    );

    /// Tokens `[t·(h/p)·(w/p), c·p²]` (feature `cc·p² + kh·p + kw`,
    /// channel-major — the checkpoint contract, matching [`Backend::patchify`])
    /// → latent `[c, t, h, w]`.
    #[allow(clippy::too_many_arguments)]
    fn unpatchify(
        &self,
        x: &Self::Buf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &Self::Buf,
    );

    /// In place on `x` (`n` elements): `x0 = x − σ_t·v`, then
    /// `x = (1 − σ_n)·x0 + σ_n·noise` when `noise` is given.
    fn dmd_step(
        &self,
        x: &Self::Buf,
        v: &Self::Buf,
        noise: Option<&Self::Buf>,
        sigma_t: f32,
        sigma_next: f32,
        n: usize,
    );

    /// `out = epilogue(conv(cat([a, b]), W) + bias)`; `W` is `[cout, cin·k²]`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        a: &Self::Buf,
        b: Option<&Self::Buf>,
        shape: &ConvShape,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        epilogue: ConvEpilogue<'_, Self::Buf>,
        out: &Self::Buf,
    );

    /// In place: `x = tanh(x / s) · s` over the first `n` elements.
    fn tanh_clamp(&self, x: &Self::Buf, n: usize, s: f32);
}
