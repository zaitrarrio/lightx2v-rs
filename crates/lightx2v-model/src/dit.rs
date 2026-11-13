//! The causal Wan DiT (Diffusers `WanTransformer3DModel`, batch 1, no CFG).
//!
//! Per block (slot order of the `[6, dim]` modulation: shift, scale, gate of
//! the self-attention, then of the FFN):
//!
//! ```text
//! n   = LN(h)·(1+scale_msa)+shift_msa                    layer_norm
//! qkv = n·W_qkvᵀ                                         linear (q|k|v fused)
//! q,k = rope(rms(q|k)), v                                head_split ×3
//! k,v → layer cache (MemRoPE append + update)            kv_cache_*
//! h  += to_out(causal_attn(q, cache_k, cache_v))·gate_msa
//! n   = norm2(h)                                         residual_norm
//! h  += to_out(attn(rms(n·W_qᵀ), K_text, V_text))        linear, head_split, attention, linear
//! n   = LN(h)·(1+scale_ffn)+shift_ffn                    ...fused into the residual above
//! h  += ffn(n)·gate_ffn                                  linear ×2 per token chunk, gate_residual
//! ```
//!
//! Differences from a plain Wan DiT:
//!
//! * Self-attention reads the **layer's MemRoPE KV cache** (all cached
//!   frames plus the new chunk) through a causal block mask — a query token
//!   attends to a key token iff the key's temporal index is ≤ the query's
//!   (`tk <= tq`). The mask is built once per chunk from the cache's
//!   temporal indices and shared by all blocks.
//! * The cache update runs **once per layer per chunk**, right after the
//!   chunk's K/V are appended and before attention: the oldest recent frame
//!   is folded into the EMA slots and evicted. Folding the *oldest recent*
//!   frame (never a frame of the current chunk) makes the before/after
//!   ordering equivalent to upstream's apply-after-layer.
//! * RoPE for the chunk uses the frames' **original** temporal indices
//!   (`rope_table_for_indices`), so a frame compressed into an EMA slot
//!   still rotates by the position it was generated at.

use crate::timestep::{matvec, silu, sinusoid};
use lightx2v_core::{
    backend::{Act, Backend, NormKind},
    config::{MemRopeConfig, WanConfig},
    quant::{Stored, WeightQuant, Q4, Q8},
    rope::rope_table_for_indices,
    weights::WeightSource,
    Error, Result,
};
use std::collections::HashMap;

const SHIFT_MSA: usize = 0;
const SCALE_MSA: usize = 1;
const GATE_MSA: usize = 2;
const SHIFT_FFN: usize = 3;
const SCALE_FFN: usize = 4;
const GATE_FFN: usize = 5;

struct Linear<B: Backend> {
    w: B::Weight,
    b: B::Buf,
}

type Shapes = HashMap<String, Vec<usize>>;

/// Several `nn.Linear`s stacked by rows into one matrix (fused QKV / KV),
/// each checked against its manifest shape.
fn load_linear<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    shapes: &Shapes,
    keys: &[&str],
    quant: WeightQuant,
) -> Result<Linear<B>> {
    let err = |m: String| Error::Weights(m);
    let (mut w, mut b, mut stored, mut rows, mut cols) = (Vec::new(), Vec::new(), Vec::new(), 0, 0);
    for k in keys {
        let shape = shapes
            .get(&format!("{k}.weight"))
            .ok_or_else(|| err(format!("`{k}` is not in the manifest")))?;
        let (o, i) = (shape[0], shape[1..].iter().product::<usize>());
        match src.stored(k)? {
            Some(s) => {
                let (r, c) = match &s {
                    Stored::Q8(q) => (q.rows, q.cols),
                    Stored::Q4(q) => (q.rows, q.cols),
                    Stored::Fp8(q) => (q.rows, q.cols),
                };
                if (r, c) != (o, i) {
                    return Err(err(format!("`{k}` is stored {r}x{c}, expected {o}x{i}")));
                }
                stored.push(s);
            }
            None => w.extend(src.get(&format!("{k}.weight"), shape)?),
        }
        b.extend(src.get(&format!("{k}.bias"), &[o])?);
        rows += o;
        cols = i;
    }
    let w = if stored.is_empty() {
        match quant.for_key(keys[0]) {
            WeightQuant::Native => be.weight(&w, rows, cols),
            WeightQuant::Int8 => be.weight_q8(&Q8::quantize(&w, rows, cols)),
            WeightQuant::Int4 { group, hqq, .. } => {
                be.weight_q4(&Q4::quantize(&w, rows, cols, group, hqq))
            }
            WeightQuant::Fp8 => {
                be.weight_fp8(&lightx2v_core::quant::Fp8E4M3::quantize(&w, rows, cols))
            }
        }
    } else if stored.len() != keys.len() {
        return Err(err(format!("{keys:?}: only some are stored quantized")));
    } else if stored.iter().all(|s| matches!(s, Stored::Q8(_))) {
        be.weight_q8(&Q8::concat(
            stored
                .into_iter()
                .map(|s| match s {
                    Stored::Q8(q) => q,
                    _ => unreachable!(),
                })
                .collect(),
        ))
    } else if stored.iter().all(|s| matches!(s, Stored::Q4(_))) {
        be.weight_q4(&Q4::concat(
            stored
                .into_iter()
                .map(|s| match s {
                    Stored::Q4(q) => q,
                    _ => unreachable!(),
                })
                .collect(),
        ))
    } else if stored.iter().all(|s| matches!(s, Stored::Fp8(_))) {
        let parts: Vec<lightx2v_core::quant::Fp8E4M3> = stored
            .into_iter()
            .map(|s| match s {
                Stored::Fp8(q) => q,
                _ => unreachable!(),
            })
            .collect();
        let (mut q, mut scales, rows) = (Vec::new(), Vec::new(), 0);
        let (cols, group) = (parts[0].cols, parts[0].group);
        let mut total_rows = 0;
        for p in parts {
            q.extend(p.q);
            scales.extend(p.scales);
            total_rows += p.rows;
        }
        let _ = rows;
        be.weight_fp8(&lightx2v_core::quant::Fp8E4M3 {
            q,
            scales,
            rows: total_rows,
            cols,
            group,
        })
    } else {
        return Err(err(format!("{keys:?}: mixed quantized storage")));
    };
    Ok(Linear {
        w,
        b: be.upload(&b),
    })
}

struct Block<B: Backend> {
    qkv: Linear<B>,
    norm_q1: B::Buf,
    norm_k1: B::Buf,
    out1: Linear<B>,
    q2: Linear<B>,
    kv2: Linear<B>,
    norm_q2: B::Buf,
    norm_k2: B::Buf,
    out2: Linear<B>,
    norm2_w: B::Buf,
    norm2_b: B::Buf,
    ffn1: Linear<B>,
    ffn2: Linear<B>,
    /// `to_gate_compress`: present on sparse-attention checkpoints.
    /// Phase 1 runs causal dense attention and does not read it; the
    /// Phase 2 sparse path will.
    #[allow(dead_code)]
    gate: Option<Linear<B>>,
    /// `[6·dim]`, host: added to `timestep_proj` once per step.
    table: Vec<f32>,
}

/// Host copies of the timestep MLP.
struct TimeEmbed {
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    pw: Vec<f32>,
    pb: Vec<f32>,
    head_table: Vec<f32>,
}

pub struct LightDit<B: Backend> {
    pub cfg: WanConfig,
    pub memrope: MemRopeConfig,
    patch: Linear<B>,
    text1: Linear<B>,
    text2: Linear<B>,
    blocks: Vec<Block<B>>,
    proj_out: Linear<B>,
    time: TimeEmbed,
}

/// Per-step conditioning, on the device: one `[6, dim]` table per block and
/// the `[2, dim]` output-head table.
pub struct StepMods<B: Backend> {
    pub blocks: Vec<B::Buf>,
    pub head: B::Buf,
}

/// Per-prompt cross-attention K/V of every block, `[heads, text_len, d]`.
pub struct Prompt<B: Backend> {
    kv: Vec<(B::Buf, B::Buf)>,
}

/// Activation buffers for one chunk, reused by every block and step.
pub struct Workspace<B: Backend> {
    /// Tokens in the chunk.
    pub tokens: usize,
    /// Token grid of the chunk `(chunk_frames, h/p, w/p)`.
    pub grid: (usize, usize, usize),
    pub ffn_chunk: usize,
    /// One MemRoPE KV cache per layer.
    pub caches: Vec<B::KvCache>,
    pub h: B::Buf,
    pub normed: B::Buf,
    pub qkv: B::Buf,
    pub q: B::Buf,
    pub k: B::Buf,
    pub v: B::Buf,
    pub attn: B::Buf,
    pub tmp: B::Buf,
    pub hid: B::Buf,
    pub patches: B::Buf,
    pub rope_cos: B::Buf,
    pub rope_sin: B::Buf,
}

/// One transformer block's weights, loaded from `src`.
fn load_block<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    shapes: &Shapes,
    cfg: &WanConfig,
    l: usize,
    quant: WeightQuant,
) -> Result<Block<B>> {
    let d = cfg.dim();
    let lin = |keys: &[&str]| load_linear(be, src, shapes, keys, quant);
    let vec = |k: &str, n: usize| -> Result<B::Buf> { Ok(be.upload(&src.get(k, &[n])?)) };
    let p = |s: &str| format!("blocks.{l}.{s}");
    Ok(Block {
        qkv: lin(&[&p("attn1.to_q"), &p("attn1.to_k"), &p("attn1.to_v")])?,
        norm_q1: vec(&p("attn1.norm_q.weight"), d)?,
        norm_k1: vec(&p("attn1.norm_k.weight"), d)?,
        out1: lin(&[&p("attn1.to_out.0")])?,
        q2: lin(&[&p("attn2.to_q")])?,
        kv2: lin(&[&p("attn2.to_k"), &p("attn2.to_v")])?,
        norm_q2: vec(&p("attn2.norm_q.weight"), d)?,
        norm_k2: vec(&p("attn2.norm_k.weight"), d)?,
        out2: lin(&[&p("attn2.to_out.0")])?,
        norm2_w: vec(&p("norm2.weight"), d)?,
        norm2_b: vec(&p("norm2.bias"), d)?,
        ffn1: lin(&[&p("ffn.net.0.proj")])?,
        ffn2: lin(&[&p("ffn.net.2")])?,
        gate: if src.has_linear(&p("to_gate_compress")) {
            Some(lin(&[&p("to_gate_compress")])?)
        } else {
            None
        },
        table: src.get(&p("scale_shift_table"), &[1, 6, d])?,
    })
}

/// The DiT's non-block weights (patch embedder, text embedder, output
/// projection, timestep MLP) plus the head modulation table.
struct Head<B: Backend> {
    patch: Linear<B>,
    text1: Linear<B>,
    text2: Linear<B>,
    proj_out: Linear<B>,
    time: TimeEmbed,
}

fn load_head<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    shapes: &Shapes,
    cfg: &WanConfig,
    quant: WeightQuant,
) -> Result<Head<B>> {
    let d = cfg.dim();
    let lin = |keys: &[&str]| load_linear(be, src, shapes, keys, quant);
    let ce = "condition_embedder";
    Ok(Head {
        patch: lin(&["patch_embedding"])?,
        text1: lin(&[&format!("{ce}.text_embedder.linear_1")])?,
        text2: lin(&[&format!("{ce}.text_embedder.linear_2")])?,
        proj_out: lin(&["proj_out"])?,
        time: TimeEmbed {
            l1w: src.get(
                &format!("{ce}.time_embedder.linear_1.weight"),
                &[d, cfg.freq_dim],
            )?,
            l1b: src.get(&format!("{ce}.time_embedder.linear_1.bias"), &[d])?,
            l2w: src.get(&format!("{ce}.time_embedder.linear_2.weight"), &[d, d])?,
            l2b: src.get(&format!("{ce}.time_embedder.linear_2.bias"), &[d])?,
            pw: src.get(&format!("{ce}.time_proj.weight"), &[6 * d, d])?,
            pb: src.get(&format!("{ce}.time_proj.bias"), &[6 * d])?,
            head_table: src.get("scale_shift_table", &[1, 2, d])?,
        },
    })
}

/// Incremental DiT loading: one transformer block at a time, then the head.
/// The browser engine (`lightx2v-web`) streams a block-ordered checkpoint
/// with one HTTP Range request per block, so wasm memory holds one block at
/// a time; the native `load_with` path is this builder fed from one source.
pub struct DitBuilder<B: Backend> {
    cfg: WanConfig,
    memrope: MemRopeConfig,
    quant: WeightQuant,
    shapes: Shapes,
    blocks: Vec<Block<B>>,
}

impl<B: Backend> DitBuilder<B> {
    pub fn new(cfg: WanConfig, memrope: MemRopeConfig, quant: WeightQuant) -> Self {
        let d = cfg.dim();
        let layers = cfg.layers;
        let mut shapes: Shapes = lightx2v_core::weights::wan_manifest(&cfg)
            .into_iter()
            .collect();
        for l in 0..layers {
            shapes.insert(format!("blocks.{l}.to_gate_compress.weight"), vec![d, d]);
        }
        Self {
            cfg,
            memrope,
            quant,
            shapes,
            blocks: Vec::with_capacity(layers),
        }
    }

    /// Blocks loaded so far (the page compares this with `cfg.layers`).
    pub fn loaded(&self) -> usize {
        self.blocks.len()
    }

    /// Load the next block (`blocks.<loaded>.*`) from `src`. The caller
    /// hands it exactly one block's tensors — a block-ordered checkpoint
    /// sliced into per-block safetensors blobs.
    pub fn add_block(&mut self, be: &B, src: &dyn WeightSource) -> Result<usize> {
        let l = self.blocks.len();
        if l >= self.cfg.layers {
            return Err(Error::Weights(format!(
                "block {l} is past the model's {} layers",
                self.cfg.layers
            )));
        }
        let b = load_block(be, src, &self.shapes, &self.cfg, l, self.quant)?;
        self.blocks.push(b);
        Ok(self.blocks.len())
    }

    /// Load the head tensors and complete the DiT. Fails unless every
    /// block was added first.
    pub fn finish(self, be: &B, src: &dyn WeightSource) -> Result<LightDit<B>> {
        if self.blocks.len() != self.cfg.layers {
            return Err(Error::Weights(format!(
                "DiT has {} of {} blocks; the head cannot load yet",
                self.blocks.len(),
                self.cfg.layers
            )));
        }
        let head = load_head(be, src, &self.shapes, &self.cfg, self.quant)?;
        Ok(LightDit {
            patch: head.patch,
            text1: head.text1,
            text2: head.text2,
            blocks: self.blocks,
            proj_out: head.proj_out,
            time: head.time,
            cfg: self.cfg,
            memrope: self.memrope,
        })
    }
}

impl<B: Backend> LightDit<B> {
    pub fn load(
        be: &B,
        cfg: WanConfig,
        memrope: MemRopeConfig,
        src: &dyn WeightSource,
    ) -> Result<Self> {
        Self::load_with(be, cfg, memrope, src, WeightQuant::Native)
    }

    pub fn load_with(
        be: &B,
        cfg: WanConfig,
        memrope: MemRopeConfig,
        src: &dyn WeightSource,
        quant: WeightQuant,
    ) -> Result<Self> {
        let mut b = DitBuilder::new(cfg, memrope, quant);
        for _ in 0..b.cfg.layers {
            b.add_block(be, src)?;
        }
        b.finish(be, src)
    }

    /// Host timestep conditioning for one DMD timestep, uploaded.
    pub fn step_mods(&self, be: &B, timestep: f32) -> StepMods<B> {
        let t = &self.time;
        let emb = sinusoid(timestep, self.cfg.freq_dim);
        let h = silu(&matvec(&t.l1w, Some(&t.l1b), &emb));
        let temb = matvec(&t.l2w, Some(&t.l2b), &h);
        let proj = matvec(&t.pw, Some(&t.pb), &silu(&temb));
        let d = self.cfg.dim();
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let m: Vec<f32> = proj.iter().zip(&b.table).map(|(a, c)| a + c).collect();
                be.upload(&m)
            })
            .collect();
        let head: Vec<f32> = (0..2 * d).map(|i| t.head_table[i] + temb[i % d]).collect();
        StepMods {
            blocks,
            head: be.upload(&head),
        }
    }

    /// Text projection + every block's cross K/V for `[text_len, text_dim]`
    /// prompt embeddings (zero-padded to `text_len`).
    pub fn prepare_prompt(&self, be: &B, embeds: &[f32]) -> Prompt<B> {
        let (l, d, c) = (self.cfg.text_len, self.cfg.dim(), &self.cfg);
        assert_eq!(
            embeds.len(),
            l * c.text_dim,
            "prompt embeds must be [text_len, text_dim]"
        );
        let x = be.upload(embeds);
        let mid = be.alloc(l * d);
        let ctx = be.alloc(l * d);
        be.linear(
            &x,
            0,
            l,
            &self.text1.w,
            Some(&self.text1.b),
            Act::GeluTanh,
            &mid,
            0,
        );
        be.linear(
            &mid,
            0,
            l,
            &self.text2.w,
            Some(&self.text2.b),
            Act::None,
            &ctx,
            0,
        );
        let kv = be.alloc(l * 2 * d);
        let kv = self
            .blocks
            .iter()
            .map(|b| {
                be.linear(&ctx, 0, l, &b.kv2.w, Some(&b.kv2.b), Act::None, &kv, 0);
                let (k, v) = (be.alloc(l * d), be.alloc(l * d));
                be.head_split(
                    &kv,
                    l,
                    2 * d,
                    0,
                    c.heads,
                    c.head_dim,
                    Some(&b.norm_k2),
                    None,
                    c.eps,
                    &k,
                );
                be.head_split(&kv, l, 2 * d, d, c.heads, c.head_dim, None, None, c.eps, &v);
                (k, v)
            })
            .collect();
        Prompt { kv }
    }

    /// Workspace for one chunk: `chunk_frames` latent frames at the given
    /// latent spatial size, starting at `start_frame` (block RoPE).
    pub fn workspace(
        &self,
        be: &B,
        chunk_frames: usize,
        latent_hw: (usize, usize),
        start_frame: usize,
        ffn_chunk: usize,
    ) -> Workspace<B> {
        let c = &self.cfg;
        let grid = (chunk_frames, latent_hw.0 / c.patch, latent_hw.1 / c.patch);
        let s = grid.0 * grid.1 * grid.2;
        let d = c.dim();
        let chunk = ffn_chunk.clamp(1, s);
        let indices: Vec<usize> = (start_frame..start_frame + chunk_frames).collect();
        let rope = rope_table_for_indices(c, (grid.1, grid.2), &indices);
        let frame_tokens = grid.1 * grid.2;
        let capacity = self.memrope.cache_frames * frame_tokens;
        Workspace {
            tokens: s,
            grid,
            ffn_chunk: chunk,
            caches: (0..c.layers)
                .map(|_| be.kv_cache_new(c.heads, c.head_dim, capacity))
                .collect(),
            h: be.alloc(s * d),
            normed: be.alloc(s * d),
            qkv: be.alloc(s * 3 * d),
            q: be.alloc(s * d),
            k: be.alloc(s * d),
            v: be.alloc(s * d),
            attn: be.alloc(s * d),
            tmp: be.alloc(s * d),
            hid: be.alloc(chunk * c.ffn_dim),
            patches: be.alloc(s * c.patch_features_in().max(c.patch_features_out())),
            rope_cos: be.upload_table(&rope.cos),
            rope_sin: be.upload_table(&rope.sin),
        }
    }

    /// Patch-embed a chunk's latent into tokens. `latent` is
    /// `[C, chunk_frames, H, W]`; `out` is `[tokens, dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn patch_embed(
        &self,
        be: &B,
        ws: &Workspace<B>,
        latent: &B::Buf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
    ) {
        be.patchify(latent, c, t, h, w, self.cfg.patch, &ws.patches);
        be.linear(
            &ws.patches,
            0,
            ws.tokens,
            &self.patch.w,
            Some(&self.patch.b),
            Act::None,
            &ws.h,
            0,
        );
    }

    /// Flow prediction `v` for one chunk of tokens (`ws.h`, `[tokens, dim]`),
    /// written to `out` as patch features (`[tokens, patch_features_out]`).
    /// `start_frame` is the chunk's first latent frame; `frame_tokens` the
    /// tokens per latent frame. The per-layer caches in `ws` are appended to
    /// and updated by this call.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_chunk(
        &self,
        be: &B,
        ws: &mut Workspace<B>,
        mods: &StepMods<B>,
        prompt: &Prompt<B>,
        start_frame: usize,
        frame_tokens: usize,
        out: &B::Buf,
    ) {
        let c = &self.cfg;
        let (s, d, hd, nh, eps) = (ws.tokens, c.dim(), c.head_dim, c.heads, c.eps);
        let rope = Some((&ws.rope_cos, &ws.rope_sin));
        // Temporal index of every token in the chunk.
        let token_temporal: Vec<usize> = (0..s).map(|i| start_frame + i / frame_tokens).collect();
        // The causal block mask is built once per chunk (after the first
        // layer's cache update) and shared by all layers: the temporal
        // lists are identical across layers.
        let mut mask: Option<Vec<bool>> = None;
        for (li, (bl, m)) in self.blocks.iter().zip(&mods.blocks).enumerate() {
            // Self-attention.
            be.layer_norm(
                &ws.h,
                s,
                d,
                NormKind::Mod {
                    scale: SCALE_MSA,
                    shift: SHIFT_MSA,
                },
                m,
                eps,
                &ws.normed,
            );
            be.linear(
                &ws.normed,
                0,
                s,
                &bl.qkv.w,
                Some(&bl.qkv.b),
                Act::None,
                &ws.qkv,
                0,
            );
            be.head_split(
                &ws.qkv,
                s,
                3 * d,
                0,
                nh,
                hd,
                Some(&bl.norm_q1),
                rope,
                eps,
                &ws.q,
            );
            be.head_split(
                &ws.qkv,
                s,
                3 * d,
                d,
                nh,
                hd,
                Some(&bl.norm_k1),
                rope,
                eps,
                &ws.k,
            );
            be.head_split(&ws.qkv, s, 3 * d, 2 * d, nh, hd, None, None, eps, &ws.v);
            // MemRoPE: write the chunk's K/V (append when new, overwrite
            // when this chunk is re-run on the next DMD step), fold/evict,
            // then attend.
            let cache = &mut ws.caches[li];
            be.kv_cache_put(cache, &ws.k, &ws.v, s, &token_temporal);
            be.kv_cache_memrope_update(cache, &self.memrope, frame_tokens);
            if mask.is_none() {
                let temporal = be.kv_cache_temporal(cache);
                let rows = be.kv_cache_len(cache);
                debug_assert_eq!(temporal.len(), rows);
                let mut msk = Vec::with_capacity(s * rows);
                for &tq in token_temporal.iter().take(s) {
                    for &tk in temporal.iter() {
                        msk.push(tk <= tq);
                    }
                }
                mask = Some(msk);
            }
            let (k_all, v_all) = be.kv_cache_kv(cache);
            let sk = be.kv_cache_len(cache);
            be.causal_attention(
                &ws.q,
                &k_all,
                &v_all,
                nh,
                s,
                sk,
                hd,
                mask.as_deref(),
                &ws.attn,
            );
            be.linear(
                &ws.attn,
                0,
                s,
                &bl.out1.w,
                Some(&bl.out1.b),
                Act::None,
                &ws.tmp,
                0,
            );
            let norm2 = NormKind::Affine {
                w: &bl.norm2_w,
                b: &bl.norm2_b,
            };
            be.residual_norm(
                &ws.h,
                &ws.tmp,
                s,
                d,
                m,
                Some(GATE_MSA),
                norm2,
                eps,
                &ws.normed,
            );
            // Cross-attention against the prompt's cached K/V.
            let (k2, v2) = &prompt.kv[li];
            be.linear(
                &ws.normed,
                0,
                s,
                &bl.q2.w,
                Some(&bl.q2.b),
                Act::None,
                &ws.tmp,
                0,
            );
            be.head_split(
                &ws.tmp,
                s,
                d,
                0,
                nh,
                hd,
                Some(&bl.norm_q2),
                None,
                eps,
                &ws.q,
            );
            be.attention(&ws.q, k2, v2, nh, s, c.text_len, hd, &ws.attn);
            be.linear(
                &ws.attn,
                0,
                s,
                &bl.out2.w,
                Some(&bl.out2.b),
                Act::None,
                &ws.tmp,
                0,
            );
            let ffn_norm = NormKind::Mod {
                scale: SCALE_FFN,
                shift: SHIFT_FFN,
            };
            be.residual_norm(&ws.h, &ws.tmp, s, d, m, None, ffn_norm, eps, &ws.normed);
            // FFN in token chunks: the `ffn_dim`-wide hidden never exceeds
            // one chunk.
            let mut r = 0;
            while r < s {
                let n = ws.ffn_chunk.min(s - r);
                be.linear(
                    &ws.normed,
                    r,
                    n,
                    &bl.ffn1.w,
                    Some(&bl.ffn1.b),
                    Act::GeluTanh,
                    &ws.hid,
                    0,
                );
                be.linear(
                    &ws.hid,
                    0,
                    n,
                    &bl.ffn2.w,
                    Some(&bl.ffn2.b),
                    Act::None,
                    &ws.tmp,
                    r,
                );
                r += n;
            }
            be.gate_residual(&ws.h, &ws.tmp, s, d, m, GATE_FFN);
        }
        // Output head: AdaLN with the [2, dim] table (shift = row 0, scale = row 1).
        be.layer_norm(
            &ws.h,
            s,
            d,
            NormKind::Mod { scale: 1, shift: 0 },
            &mods.head,
            eps,
            &ws.normed,
        );
        be.linear(
            &ws.normed,
            0,
            s,
            &self.proj_out.w,
            Some(&self.proj_out.b),
            Act::None,
            out,
            0,
        );
    }
}
