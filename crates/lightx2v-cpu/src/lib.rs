//! CPU reference backend: the oracle every other backend is checked
//! against. Straightforward f32 loops, no unsafe shortcuts.
//!
//! Buffers are `Rc<RefCell<Vec<f32>>>` so the in-place ops
//! (`residual_norm`, `gate_residual`, `dmd_step`, `tanh_clamp`) can borrow
//! mutably through a shared `&self`. Weights are `Rc`'d flat f32 matrices.
//!
//! The KV cache stores rows as `[rows, heads·head_dim]` (row-major over
//! tokens) with a temporal index and a [`SlotKind`] per row, mirroring
//! [`lightx2v_core::memrope::MemRopeCache`]; `kv_cache_kv` re-emits them in
//! the `[heads, rows, head_dim]` layout attention reads.

use lightx2v_core::{
    backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind},
    config::MemRopeConfig,
    memrope::{ema_update, SlotKind},
    quant::{Fp8E4M3, Q4, Q8},
    sparse::SparseSpec,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Buf(pub Rc<RefCell<Vec<f32>>>);

impl Buf {
    fn get(&self) -> std::cell::Ref<'_, Vec<f32>> {
        self.0.borrow()
    }
    fn get_mut(&self) -> std::cell::RefMut<'_, Vec<f32>> {
        self.0.borrow_mut()
    }
}

#[derive(Debug, Clone)]
pub struct Weight {
    pub data: Rc<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
}

/// One layer's KV cache.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub heads: usize,
    pub head_dim: usize,
    pub capacity_tokens: usize,
    /// K rows, `[rows, heads·head_dim]`.
    pub k: Vec<f32>,
    /// V rows, same layout.
    pub v: Vec<f32>,
    /// Temporal index per row (block RoPE).
    pub temporal: Vec<usize>,
    /// Slot kind per row (MemRoPE).
    pub kinds: Vec<SlotKind>,
}

impl KvCache {
    pub fn rows(&self) -> usize {
        self.temporal.len()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CpuBackend;

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn softmax_row(scores: &mut [f32]) {
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    let inv = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv;
    }
}

/// Plain softmax attention over `[heads, sq, d]` × `[heads, sk, d]`.
/// `mask` is `[sq, sk]` row-major when given; `false` forbids the pair.
#[allow(clippy::too_many_arguments)]
fn attention_impl(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    sq: usize,
    sk: usize,
    d: usize,
    mask: Option<&[bool]>,
    out: &mut [f32],
) {
    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; sk];
    for h in 0..heads {
        for qi in 0..sq {
            for ki in 0..sk {
                let allowed = mask.is_none_or(|m| m[qi * sk + ki]);
                if !allowed {
                    scores[ki] = f32::NEG_INFINITY;
                    continue;
                }
                let mut s = 0.0;
                for j in 0..d {
                    s += q[(h * sq + qi) * d + j] * k[(h * sk + ki) * d + j];
                }
                scores[ki] = s * scale;
            }
            softmax_row(&mut scores);
            for j in 0..d {
                let mut acc = 0.0;
                for ki in 0..sk {
                    acc += scores[ki] * v[(h * sk + ki) * d + j];
                }
                out[(qi * heads + h) * d + j] = acc;
            }
        }
    }
}

impl Backend for CpuBackend {
    type Buf = Buf;
    type Sparse = SparseSpec;
    type Weight = Weight;
    type KvCache = KvCache;

    fn name(&self) -> String {
        "cpu".to_string()
    }

    fn alloc(&self, len: usize) -> Buf {
        Buf(Rc::new(RefCell::new(vec![0.0; len])))
    }

    fn upload(&self, data: &[f32]) -> Buf {
        Buf(Rc::new(RefCell::new(data.to_vec())))
    }

    fn write(&self, buf: &Buf, data: &[f32]) {
        buf.get_mut()[..data.len()].copy_from_slice(data);
    }

    fn download(&self, buf: &Buf) -> Vec<f32> {
        buf.get().clone()
    }

    fn buf_len(&self, buf: &Buf) -> usize {
        buf.get().len()
    }

    fn copy(&self, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, len: usize) {
        let tmp: Vec<f32> = src.get()[src_off..src_off + len].to_vec();
        dst.get_mut()[dst_off..dst_off + len].copy_from_slice(&tmp);
    }

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> Weight {
        assert_eq!(data.len(), rows * cols);
        Weight {
            data: Rc::new(data.to_vec()),
            rows,
            cols,
        }
    }

    fn weight_q8(&self, q: &Q8) -> Weight {
        let d = q.dequantize();
        Weight {
            data: Rc::new(d),
            rows: q.rows,
            cols: q.cols,
        }
    }

    fn weight_q4(&self, q: &Q4) -> Weight {
        let d = q.dequantize();
        Weight {
            data: Rc::new(d),
            rows: q.rows,
            cols: q.cols,
        }
    }

    fn weight_fp8(&self, q: &Fp8E4M3) -> Weight {
        let d = q.dequantize();
        Weight {
            data: Rc::new(d),
            rows: q.rows,
            cols: q.cols,
        }
    }

    fn sync(&self) {}

    fn linear(
        &self,
        x: &Buf,
        x_row: usize,
        rows: usize,
        w: &Weight,
        bias: Option<&Buf>,
        act: Act,
        y: &Buf,
        y_row: usize,
    ) {
        let x = x.get();
        let mut y = y.get_mut();
        let bias = bias.map(|b| b.get());
        for r in 0..rows {
            for o in 0..w.rows {
                let mut acc = 0.0;
                for i in 0..w.cols {
                    acc += x[(x_row + r) * w.cols + i] * w.data[o * w.cols + i];
                }
                if let Some(b) = &bias {
                    acc += b[o];
                }
                y[(y_row + r) * w.rows + o] = match act {
                    Act::None => acc,
                    Act::GeluTanh => gelu_tanh(acc),
                };
            }
        }
    }

    fn layer_norm(
        &self,
        x: &Buf,
        rows: usize,
        dim: usize,
        norm: NormKind<'_, Buf>,
        mods: &Buf,
        eps: f32,
        out: &Buf,
    ) {
        let x = x.get();
        let mods = mods.get();
        let mut out = out.get_mut();
        let affine = match &norm {
            NormKind::Affine { w, b } => Some((w.get(), b.get())),
            NormKind::Mod { .. } => None,
        };
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..dim {
                let n = (row[j] - mean) * inv;
                out[r * dim + j] = match (&norm, &affine) {
                    (NormKind::Affine { .. }, Some((w, b))) => n * w[j] + b[j],
                    (NormKind::Mod { scale, shift }, _) => {
                        n * (1.0 + mods[scale * dim + j]) + mods[shift * dim + j]
                    }
                    _ => unreachable!(),
                };
            }
        }
    }

    fn residual_norm(
        &self,
        h: &Buf,
        a: &Buf,
        rows: usize,
        dim: usize,
        mods: &Buf,
        gate: Option<usize>,
        norm: NormKind<'_, Buf>,
        eps: f32,
        normed: &Buf,
    ) {
        {
            let mut h = h.get_mut();
            let a = a.get();
            let mods = mods.get();
            for r in 0..rows {
                for j in 0..dim {
                    let gv = match gate {
                        Some(gi) => mods[gi * dim + j],
                        None => 1.0,
                    };
                    h[r * dim + j] += a[r * dim + j] * gv;
                }
            }
        }
        self.layer_norm(h, rows, dim, norm, mods, eps, normed);
    }

    fn gate_residual(&self, h: &Buf, a: &Buf, rows: usize, dim: usize, mods: &Buf, gate: usize) {
        let mut h = h.get_mut();
        let a = a.get();
        let mods = mods.get();
        for r in 0..rows {
            for j in 0..dim {
                h[r * dim + j] += a[r * dim + j] * mods[gate * dim + j];
            }
        }
    }

    fn head_split(
        &self,
        src: &Buf,
        rows: usize,
        src_stride: usize,
        col_off: usize,
        heads: usize,
        head_dim: usize,
        rms_w: Option<&Buf>,
        rope: Option<(&Buf, &Buf)>,
        eps: f32,
        out: &Buf,
    ) {
        let src = src.get();
        let mut out = out.get_mut();
        let rms_w = rms_w.map(|b| b.get());
        let rope = rope.map(|(c, s)| (c.get(), s.get()));
        let d = heads * head_dim;
        for r in 0..rows {
            let row = &src[r * src_stride + col_off..r * src_stride + col_off + d];
            // RMS norm across all heads, then per-head weight — but only
            // when a weight is given: v passes through unnormalised.
            let mut tmp = vec![0.0f32; d];
            if let Some(w) = &rms_w {
                let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                for j in 0..d {
                    tmp[j] = row[j] * inv * w[j];
                }
            } else {
                tmp.copy_from_slice(row);
            }
            // RoPE: rotate interleaved pairs within each head.
            if let Some((cos, sin)) = &rope {
                for h in 0..heads {
                    for i in 0..head_dim / 2 {
                        let (a, b) = (tmp[h * head_dim + 2 * i], tmp[h * head_dim + 2 * i + 1]);
                        let (c, s) = (cos[r * head_dim / 2 + i], sin[r * head_dim / 2 + i]);
                        tmp[h * head_dim + 2 * i] = a * c - b * s;
                        tmp[h * head_dim + 2 * i + 1] = a * s + b * c;
                    }
                }
            }
            for h in 0..heads {
                for j in 0..head_dim {
                    out[(h * rows + r) * head_dim + j] = tmp[h * head_dim + j];
                }
            }
        }
    }

    fn attention(
        &self,
        q: &Buf,
        k: &Buf,
        v: &Buf,
        heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        out: &Buf,
    ) {
        let (q, k, v) = (q.get(), k.get(), v.get());
        let mut out = out.get_mut();
        attention_impl(&q, &k, &v, heads, sq, sk, head_dim, None, &mut out);
    }

    fn causal_attention(
        &self,
        q: &Buf,
        k: &Buf,
        v: &Buf,
        heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        mask: Option<&[bool]>,
        out: &Buf,
    ) {
        let (q, k, v) = (q.get(), k.get(), v.get());
        let mut out = out.get_mut();
        attention_impl(&q, &k, &v, heads, sq, sk, head_dim, mask, &mut out);
    }

    fn sparse_prepare(&self, spec: &SparseSpec) -> SparseSpec {
        spec.clone()
    }

    fn sparse_attention(
        &self,
        q: &Buf,
        k: &Buf,
        v: &Buf,
        gate: &Buf,
        sparse: &SparseSpec,
        out: &Buf,
    ) {
        // The CPU reference is the full VSA algorithm (see crate::sparse):
        // coarse mean-pool attention over tiles, top-k tile selection by the
        // coarse scores, fine attention to the selected tiles only, and the
        // combine `out = coarse \u00b7 gate + fine`. A GPU kernel must match
        // this exactly; with `gate = 0` and sparsity 0 (top-k = all tiles)
        // it reduces to dense attention.
        let (q, k, v, gate) = (q.get(), k.get(), v.get(), gate.get());
        let plan = &sparse.plan;
        let seq = plan.seq;
        let nt = plan.num_tiles();
        let d = sparse.head_dim;
        let scale = sparse.scale;
        let mut out = out.get_mut();
        let mut token_tile = vec![0usize; seq];
        let mut tile_tokens: Vec<Vec<usize>> = vec![Vec::new(); nt];
        for (slot, &s) in plan.slot_src.iter().enumerate() {
            if s != lightx2v_core::sparse::PAD {
                let tile = slot / lightx2v_core::sparse::TILE_ELEMS;
                token_tile[s as usize] = tile;
                tile_tokens[tile].push(s as usize);
            }
        }
        for h in 0..sparse.heads {
            let mut cq = vec![0.0f32; nt * d];
            let mut ck = vec![0.0f32; nt * d];
            let mut cv = vec![0.0f32; nt * d];
            for (ti, toks) in tile_tokens.iter().enumerate() {
                let n = toks.len().max(1) as f32;
                for &tok in toks {
                    for j in 0..d {
                        cq[ti * d + j] += q[(h * seq + tok) * d + j] / n;
                        ck[ti * d + j] += k[(h * seq + tok) * d + j] / n;
                        cv[ti * d + j] += v[(h * seq + tok) * d + j] / n;
                    }
                }
            }
            let mut coarse_scores = vec![vec![0.0f32; nt]; nt];
            let mut coarse_out = vec![0.0f32; nt * d];
            for qi in 0..nt {
                let mut row = vec![0.0f32; nt];
                for ki in 0..nt {
                    let mut s = 0.0;
                    for j in 0..d {
                        s += cq[qi * d + j] * ck[ki * d + j];
                    }
                    row[ki] = s * scale;
                }
                softmax_row(&mut row);
                coarse_scores[qi] = row.clone();
                for j in 0..d {
                    let mut acc = 0.0;
                    for ki in 0..nt {
                        acc += row[ki] * cv[ki * d + j];
                    }
                    coarse_out[qi * d + j] = acc;
                }
            }
            let mut selected: Vec<Vec<usize>> = Vec::with_capacity(nt);
            for scores in &coarse_scores {
                let mut order: Vec<usize> = (0..nt).collect();
                order.sort_by(|&a, &b| {
                    scores[b]
                        .partial_cmp(&scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                order.truncate(sparse.topk);
                selected.push(order);
            }
            for qi_tok in 0..seq {
                let qt = token_tile[qi_tok];
                let mut keys: Vec<usize> = Vec::new();
                for &kt in &selected[qt] {
                    keys.extend(tile_tokens[kt].iter().copied());
                }
                let mut scores = vec![0.0f32; keys.len()];
                for (i, &kt) in keys.iter().enumerate() {
                    let mut s = 0.0;
                    for j in 0..d {
                        s += q[(h * seq + qi_tok) * d + j] * k[(h * seq + kt) * d + j];
                    }
                    scores[i] = s * scale;
                }
                softmax_row(&mut scores);
                for j in 0..d {
                    let mut fine = 0.0;
                    for (i, &kt) in keys.iter().enumerate() {
                        fine += scores[i] * v[(h * seq + kt) * d + j];
                    }
                    let coarse = coarse_out[qt * d + j];
                    let g = gate[(h * seq + qi_tok) * d + j];
                    out[(qi_tok * sparse.heads + h) * d + j] = coarse * g + fine;
                }
            }
        }
    }

    fn kv_cache_new(&self, heads: usize, head_dim: usize, capacity_tokens: usize) -> KvCache {
        KvCache {
            heads,
            head_dim,
            capacity_tokens,
            k: Vec::new(),
            v: Vec::new(),
            temporal: Vec::new(),
            kinds: Vec::new(),
        }
    }

    fn kv_cache_append(
        &self,
        cache: &mut KvCache,
        k: &Buf,
        v: &Buf,
        rows: usize,
        temporal: &[usize],
    ) {
        debug_assert_eq!(temporal.len(), rows);
        let (k, v) = (k.get(), v.get());
        let d = cache.heads * cache.head_dim;
        for (r, &t) in temporal.iter().enumerate().take(rows) {
            for h in 0..cache.heads {
                let hd = cache.head_dim;
                let src_off = (h * rows + r) * hd;
                cache.k.extend_from_slice(&k[src_off..src_off + hd]);
                cache.v.extend_from_slice(&v[src_off..src_off + hd]);
            }
            cache.temporal.push(t);
            // Rows are New until memrope_update graduates them; the first
            // `sink_frames` frames are reclassified there.
            cache.kinds.push(SlotKind::New);
        }
        debug_assert_eq!(cache.k.len(), cache.rows() * d);
    }

    fn kv_cache_put(&self, cache: &mut KvCache, k: &Buf, v: &Buf, rows: usize, temporal: &[usize]) {
        debug_assert_eq!(temporal.len(), rows);
        // Overwrite in place when these temporal indices are already cached
        // (the same chunk re-run on the next DMD step).
        // Skip the EMA slots: they share the first non-sink frame's
        // temporal index, and overwriting one would corrupt the memory.
        if let Some(pos) = cache
            .temporal
            .iter()
            .zip(&cache.kinds)
            .position(|(&t, &k)| {
                t == temporal[0] && !matches!(k, SlotKind::LongEma | SlotKind::ShortEma)
            })
        {
            let matches = (0..rows).all(|i| cache.temporal.get(pos + i) == Some(&temporal[i]));
            if matches {
                let (k, v) = (k.get(), v.get());
                let d = cache.head_dim;
                for r in 0..rows {
                    for h in 0..cache.heads {
                        let src = (h * rows + r) * d;
                        let dst = (pos + r) * cache.heads * d + h * d;
                        cache.k[dst..dst + d].copy_from_slice(&k[src..src + d]);
                        cache.v[dst..dst + d].copy_from_slice(&v[src..src + d]);
                    }
                }
                return;
            }
        }
        self.kv_cache_append(cache, k, v, rows, temporal);
    }

    fn kv_cache_len(&self, cache: &KvCache) -> usize {
        cache.rows()
    }

    fn kv_cache_temporal(&self, cache: &KvCache) -> Vec<usize> {
        cache.temporal.clone()
    }

    fn kv_cache_kv(&self, cache: &KvCache) -> (Buf, Buf) {
        let rows = cache.rows();
        let d = cache.head_dim;
        let mut k = vec![0.0f32; cache.heads * rows * d];
        let mut v = vec![0.0f32; cache.heads * rows * d];
        let stride = cache.heads * d;
        for r in 0..rows {
            for h in 0..cache.heads {
                let src = r * stride + h * d;
                let dst = (h * rows + r) * d;
                k[dst..dst + d].copy_from_slice(&cache.k[src..src + d]);
                v[dst..dst + d].copy_from_slice(&cache.v[src..src + d]);
            }
        }
        (self.upload(&k), self.upload(&v))
    }

    fn kv_cache_memrope_update(
        &self,
        cache: &mut KvCache,
        cfg: &MemRopeConfig,
        frame_tokens: usize,
    ) {
        let d = cache.heads * cache.head_dim;
        let frame_rows = |cache: &KvCache| cache.rows() / frame_tokens;
        // Classify rows into frames (frame_tokens consecutive rows each).
        // Sinks: the first sink_frames frames (temporal indices 0..sink).
        let n_frames = frame_rows(cache);
        if n_frames == 0 {
            return;
        }
        // Reclassify kinds by frame position, keeping EMA slots.
        let mut frame_kind: Vec<SlotKind> = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let first = cache.kinds[f * frame_tokens];
            frame_kind.push(first);
        }
        // Seed EMA slots (zero frames) after the sinks if missing and there
        // is at least one non-sink frame.
        let has_long = frame_kind.contains(&SlotKind::LongEma);
        if !has_long && n_frames > cfg.sink_frames {
            let insert_row = cfg.sink_frames * frame_tokens;
            let zero = vec![0.0f32; frame_tokens * d];
            let t = cache.temporal[insert_row.min(cache.rows().saturating_sub(1))];
            for kind in [SlotKind::ShortEma, SlotKind::LongEma] {
                // Insert Long first so cache order is [sink, long, short].
                let at = insert_row;
                cache.k.splice(at * d..at * d, zero.iter().copied());
                cache.v.splice(at * d..at * d, zero.iter().copied());
                cache
                    .temporal
                    .splice(at..at, std::iter::repeat_n(t, frame_tokens));
                cache
                    .kinds
                    .splice(at..at, std::iter::repeat_n(kind, frame_tokens));
                let _ = at;
                // Recompute frame kinds after the splice.
                frame_kind = (0..frame_rows(cache))
                    .map(|f| cache.kinds[f * frame_tokens])
                    .collect();
            }
        }
        // Graduate New frames beyond the new window to Recent.
        let mut new_frames: Vec<usize> = frame_kind
            .iter()
            .enumerate()
            .filter(|(_, &k)| k == SlotKind::New)
            .map(|(i, _)| i)
            .collect();
        if new_frames.len() > cfg.new_frames() {
            let excess = new_frames.len() - cfg.new_frames();
            for f in new_frames.drain(..excess) {
                for r in 0..frame_tokens {
                    cache.kinds[f * frame_tokens + r] = SlotKind::Recent;
                }
            }
        }
        // Evict while over capacity: fold the oldest Recent (else oldest
        // New) frame into the EMA slots, then drop it.
        while frame_rows(cache) > cfg.cache_frames {
            let kinds_now: Vec<SlotKind> = (0..frame_rows(cache))
                .map(|f| cache.kinds[f * frame_tokens])
                .collect();
            let evict = kinds_now
                .iter()
                .position(|&k| k == SlotKind::Recent)
                .or_else(|| kinds_now.iter().position(|&k| k == SlotKind::New));
            let Some(f) = evict else { break };
            let ev_k: Vec<f32> = cache.k[f * frame_tokens * d..(f + 1) * frame_tokens * d].to_vec();
            let ev_v: Vec<f32> = cache.v[f * frame_tokens * d..(f + 1) * frame_tokens * d].to_vec();
            for (slot_kind, alpha) in [
                (SlotKind::LongEma, cfg.ema_alpha_long),
                (SlotKind::ShortEma, cfg.ema_alpha_short),
            ] {
                if let Some(sf) = kinds_now.iter().position(|&k| k == slot_kind) {
                    let base = sf * frame_tokens * d;
                    let ema_k = ema_update(&cache.k[base..base + frame_tokens * d], &ev_k, alpha);
                    let ema_v = ema_update(&cache.v[base..base + frame_tokens * d], &ev_v, alpha);
                    cache.k[base..base + frame_tokens * d].copy_from_slice(&ema_k);
                    cache.v[base..base + frame_tokens * d].copy_from_slice(&ema_v);
                }
            }
            let base = f * frame_tokens;
            cache.k.drain(base * d..(base + frame_tokens) * d);
            cache.v.drain(base * d..(base + frame_tokens) * d);
            cache.temporal.drain(base..base + frame_tokens);
            cache.kinds.drain(base..base + frame_tokens);
        }
        // Reclassify the first sink_frames frames as sinks (their rows were
        // appended as New).
        for f in 0..cfg.sink_frames.min(frame_rows(cache)) {
            for r in 0..frame_tokens {
                cache.kinds[f * frame_tokens + r] = SlotKind::Sink;
            }
        }
    }

    fn patchify(&self, latent: &Buf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &Buf) {
        let latent = latent.get();
        let mut out = out.get_mut();
        // latent [c, t, h, w]; token (tt, hh, ww) -> [tt*(h/p)*(w/p) + hh*(w/p) + ww],
        // feature c*p² + kh*p + kw.
        for tt in 0..t {
            for hh in 0..h / p {
                for ww in 0..w / p {
                    let tok = (tt * (h / p) + hh) * (w / p) + ww;
                    for cc in 0..c {
                        for kh in 0..p {
                            for kw in 0..p {
                                out[tok * c * p * p + cc * p * p + kh * p + kw] =
                                    latent[((cc * t + tt) * h + hh * p + kh) * w + ww * p + kw];
                            }
                        }
                    }
                }
            }
        }
    }

    fn unpatchify(&self, x: &Buf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &Buf) {
        let x = x.get();
        let mut out = out.get_mut();
        // Token feature cc*p² + kh*p + kw (channel-major, the
        // checkpoint contract) -> latent [c, t, h, w].
        for tt in 0..t {
            for hh in 0..h / p {
                for ww in 0..w / p {
                    let tok = (tt * (h / p) + hh) * (w / p) + ww;
                    for cc in 0..c {
                        for kh in 0..p {
                            for kw in 0..p {
                                out[((cc * t + tt) * h + hh * p + kh) * w + ww * p + kw] =
                                    x[tok * c * p * p + cc * p * p + kh * p + kw];
                            }
                        }
                    }
                }
            }
        }
    }

    fn dmd_step(
        &self,
        x: &Buf,
        v: &Buf,
        noise: Option<&Buf>,
        sigma_t: f32,
        sigma_next: f32,
        n: usize,
    ) {
        let mut x = x.get_mut();
        let v = v.get();
        let noise = noise.map(|b| b.get());
        for i in 0..n {
            let x0 = x[i] - sigma_t * v[i];
            x[i] = match &noise {
                Some(noise) => (1.0 - sigma_next) * x0 + sigma_next * noise[i],
                None => x0,
            };
        }
    }

    fn conv2d(
        &self,
        a: &Buf,
        b: Option<&Buf>,
        shape: &ConvShape,
        w: &Weight,
        bias: Option<&Buf>,
        epilogue: ConvEpilogue<'_, Buf>,
        out: &Buf,
    ) {
        let (a, b) = (a.get(), b.map(|x| x.get()));
        let mut out = out.get_mut();
        let bias = bias.map(|x| x.get());
        let (oh, ow) = shape.out_hw();
        let k = shape.k;
        let pad = k / 2;
        let cin = shape.cin();
        let read = |buf: &[f32],
                    frame: usize,
                    ch: usize,
                    y: isize,
                    x: isize,
                    h: usize,
                    wdt: usize|
         -> f32 {
            if y < 0 || x < 0 || y >= h as isize || x >= wdt as isize {
                0.0
            } else {
                buf[(frame * shape.cin_a + ch) * h * wdt + y as usize * wdt + x as usize]
            }
        };
        // The two inputs are concatenated on channels; read through a helper
        // that picks the right buffer by channel.
        let read_cat =
            |a: &[f32], b: Option<&[f32]>, frame: usize, ch: usize, y: isize, x: isize| -> f32 {
                let (h, wdt) = (shape.h, shape.w);
                let (yy, xx) = if shape.upsample {
                    (y / 2, x / 2)
                } else {
                    (y, x)
                };
                if ch < shape.cin_a {
                    read(a, frame, ch, yy, xx, h, wdt)
                } else {
                    match b {
                        Some(b) => read(b, frame, ch - shape.cin_a, yy, xx, h, wdt),
                        None => 0.0,
                    }
                }
            };
        for f in 0..shape.frames {
            for oc in 0..shape.cout {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = 0.0;
                        for ic in 0..cin {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let iy = oy as isize + ky as isize - pad as isize;
                                    let ix = ox as isize + kx as isize - pad as isize;
                                    let val = read_cat(
                                        &a,
                                        b.as_deref().map(|v| v.as_slice()),
                                        f,
                                        ic,
                                        iy,
                                        ix,
                                    );
                                    acc += val * w.data[(oc * cin + ic) * k * k + ky * k + kx];
                                }
                            }
                        }
                        if let Some(bias) = &bias {
                            acc += bias[oc];
                        }
                        let idx = (f * shape.cout + oc) * oh * ow + oy * ow + ox;
                        out[idx] = match &epilogue {
                            ConvEpilogue::None => acc,
                            ConvEpilogue::Relu => acc.max(0.0),
                            ConvEpilogue::AddRelu(r) => (acc + r.get()[idx]).max(0.0),
                            ConvEpilogue::Prelu(slopes) => {
                                let s = slopes.get()[oc];
                                if acc >= 0.0 {
                                    acc
                                } else {
                                    acc * s
                                }
                            }
                        };
                    }
                }
            }
        }
    }

    fn tanh_clamp(&self, x: &Buf, n: usize, s: f32) {
        let mut x = x.get_mut();
        for v in x.iter_mut().take(n) {
            *v = (*v / s).tanh() * s;
        }
    }
}
