//! Direct f32 reference ops + reference rollout, shared by the conformance
//! tests and the debug harness. Independent of the model crate's DiT.

use lightx2v_core::config::WanConfig;
use std::collections::HashMap;

// ---------- plain f32 reference ops ----------

pub fn lin(x: &[f32], rows: usize, inn: usize, w: &[f32], b: &[f32], gelu: bool) -> Vec<f32> {
    let out = w.len() / inn;
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut acc = b[o];
            for i in 0..inn {
                acc += x[r * inn + i] * w[o * inn + i];
            }
            y[r * out + o] = if gelu {
                0.5 * acc
                    * (1.0
                        + ((2.0 / std::f32::consts::PI).sqrt()
                            * (acc + 0.044715 * acc * acc * acc))
                            .tanh())
            } else {
                acc
            };
        }
    }
    y
}

pub fn ln_mod(
    x: &[f32],
    rows: usize,
    dim: usize,
    mods: &[f32],
    scale: usize,
    shift: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for j in 0..dim {
            y[r * dim + j] =
                (row[j] - mean) * inv * (1.0 + mods[scale * dim + j]) + mods[shift * dim + j];
        }
    }
    y
}

pub fn ln_affine(x: &[f32], rows: usize, dim: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for j in 0..dim {
            y[r * dim + j] = (row[j] - mean) * inv * w[j] + b[j];
        }
    }
    y
}

/// RMS norm over the whole `dim` vector, then a per-element weight.
pub fn rms(x: &[f32], rows: usize, dim: usize, w: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for j in 0..dim {
            y[r * dim + j] = row[j] * inv * w[j];
        }
    }
    y
}

pub fn rope_apply(x: &mut [f32], rows: usize, heads: usize, hd: usize, cos: &[f32], sin: &[f32]) {
    // x is [rows, heads*hd]; cos/sin are [rows, hd/2].
    for r in 0..rows {
        for h in 0..heads {
            for i in 0..hd / 2 {
                let (a, b) = (
                    x[r * heads * hd + h * hd + 2 * i],
                    x[r * heads * hd + h * hd + 2 * i + 1],
                );
                let (c, s) = (cos[r * hd / 2 + i], sin[r * hd / 2 + i]);
                x[r * heads * hd + h * hd + 2 * i] = a * c - b * s;
                x[r * heads * hd + h * hd + 2 * i + 1] = a * s + b * c;
            }
        }
    }
}

/// `q/k/v` are `[heads, seq, d]`; mask is `[sq, sk]` row-major. Output is
/// `[sq, heads*d]`.
pub fn attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    sq: usize,
    sk: usize,
    d: usize,
    mask: Option<&[bool]>,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; sq * heads * d];
    for h in 0..heads {
        for qi in 0..sq {
            let mut scores = vec![0.0f32; sk];
            for ki in 0..sk {
                if mask.map_or(false, |m| !m[qi * sk + ki]) {
                    scores[ki] = f32::NEG_INFINITY;
                    continue;
                }
                let mut s = 0.0;
                for j in 0..d {
                    s += q[(h * sq + qi) * d + j] * k[(h * sk + ki) * d + j];
                }
                scores[ki] = s * scale;
            }
            let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for j in 0..d {
                out[(qi * heads + h) * d + j] = (0..sk)
                    .map(|ki| scores[ki] * v[(h * sk + ki) * d + j])
                    .sum();
            }
        }
    }
    out
}

pub fn patchify(latent: &[f32], c: usize, t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let tokens = t * (h / p) * (w / p);
    let mut out = vec![0.0f32; tokens * c * p * p];
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
    out
}

pub fn unpatchify(x: &[f32], c: usize, t: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * t * h * w];
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
    out
}

// ---------- reference rollout ----------

pub struct RefWeights {
    pub map: HashMap<String, Vec<f32>>,
}

impl RefWeights {
    pub fn get(&self, name: &str) -> &[f32] {
        &self.map[name]
    }
}

pub struct RefCacheFrame {
    pub temporal: usize,
    pub kind: &'static str, // "sink" | "ema" | "recent" | "new"
    pub k: Vec<f32>,        // [tokens, heads*hd]
    pub v: Vec<f32>,
}

pub struct RefCache {
    pub frames: Vec<RefCacheFrame>,
    sink_frames: usize,
}

impl RefCache {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            sink_frames: 3,
        }
    }

    /// Append one latent frame (one frame's worth of K/V rows).
    pub fn append(&mut self, temporal: usize, k: Vec<f32>, v: Vec<f32>) {
        // Seed the two zero EMA frames after the sinks on the first
        // non-sink append (mirrors the backend cache).
        let sink_count = self.frames.iter().filter(|f| f.kind == "sink").count();
        if sink_count >= self.sink_frames && !self.frames.iter().any(|f| f.kind == "ema") {
            let n = k.len();
            let at = self
                .frames
                .iter()
                .position(|f| f.kind != "sink")
                .unwrap_or(self.frames.len());
            for _ in 0..2 {
                self.frames.insert(
                    at,
                    RefCacheFrame {
                        temporal,
                        kind: "ema",
                        k: vec![0.0; n],
                        v: vec![0.0; n],
                    },
                );
            }
        }
        let kind = if sink_count < self.sink_frames {
            "sink"
        } else {
            "new"
        };
        self.frames.push(RefCacheFrame {
            temporal,
            kind,
            k,
            v,
        });
        // Graduate new frames beyond the 3-frame window to recent.
        let new: Vec<usize> = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.kind == "new")
            .map(|(i, _)| i)
            .collect();
        if new.len() > 3 {
            for &i in new.iter().take(new.len() - 3) {
                self.frames[i].kind = "recent";
            }
        }
    }

    /// Write one frame with append-or-overwrite semantics (see
    /// `Backend::kv_cache_put`): a frame whose temporal index is already
    /// cached is overwritten in place instead of appended.
    pub fn put(&mut self, temporal: usize, k: Vec<f32>, v: Vec<f32>) {
        // Skip the EMA slots: they share the first non-sink frame's
        // temporal index.
        if let Some(pos) = self
            .frames
            .iter()
            .position(|f| f.temporal == temporal && f.kind != "ema")
        {
            self.frames[pos].k = k;
            self.frames[pos].v = v;
            return;
        }
        self.append(temporal, k, v);
    }

    /// All cached K/V in cache order, as [heads, rows, d] plus per-row temporal.
    pub fn kv(&self, heads: usize, hd: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
        let rows: usize = self.frames.iter().map(|f| f.k.len() / (heads * hd)).sum();
        let mut k = vec![0.0f32; heads * rows * hd];
        let mut v = vec![0.0f32; heads * rows * hd];
        let mut temporal = Vec::with_capacity(rows);
        let mut r = 0;
        for f in &self.frames {
            let ft = f.k.len() / (heads * hd);
            for tok in 0..ft {
                temporal.push(f.temporal);
                for h in 0..heads {
                    for j in 0..hd {
                        k[(h * rows + r) * hd + j] = f.k[tok * heads * hd + h * hd + j];
                        v[(h * rows + r) * hd + j] = f.v[tok * heads * hd + h * hd + j];
                    }
                }
                r += 1;
            }
        }
        (k, v, temporal)
    }
}

pub fn ref_chunk(
    cfg: &WanConfig,
    w: &RefWeights,
    chunk_latent: &[f32],
    mods: &[Vec<f32>],
    head_mods: &[f32],
    text_kv: &[(Vec<f32>, Vec<f32>)],
    caches: &mut [RefCache],
    start_frame: usize,
) -> Vec<f32> {
    let (d, nh, hd) = (cfg.dim(), cfg.heads, cfg.head_dim);
    let (c, t, h, ww) = (cfg.in_channels, 3, 8, 8);
    let p = cfg.patch;
    let frame_tokens = (h / p) * (ww / p);
    let tokens = patchify(chunk_latent, c, t, h, ww, p);
    let s = tokens.len() / (c * p * p);
    let mut htok = lin(
        &tokens,
        s,
        c * p * p,
        w.get("patch_embedding.weight"),
        w.get("patch_embedding.bias"),
        false,
    );
    let rope = lightx2v_core::rope::rope_table_for_indices(
        cfg,
        (h / p, ww / p),
        &[start_frame, start_frame + 1, start_frame + 2],
    );
    for li in 0..cfg.layers {
        let pre = format!("blocks.{li}");
        let mods = &mods[li];
        let normed = ln_mod(&htok, s, d, mods, 1, 0);
        let qkv_w: Vec<f32> = ["attn1.to_q", "attn1.to_k", "attn1.to_v"]
            .iter()
            .flat_map(|m| w.get(&format!("{pre}.{m}.weight")).iter().copied())
            .collect();
        let qkv_b: Vec<f32> = ["attn1.to_q", "attn1.to_k", "attn1.to_v"]
            .iter()
            .flat_map(|m| w.get(&format!("{pre}.{m}.bias")).iter().copied())
            .collect();
        let qkv = lin(&normed, s, d, &qkv_w, &qkv_b, false);
        // q/k are the first two d-wide slices of qkv; v is the third.
        let mut q2 = vec![0.0f32; s * d];
        let mut k2 = vec![0.0f32; s * d];
        let mut v2 = vec![0.0f32; s * d];
        for r in 0..s {
            q2[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d..r * 3 * d + d]);
            k2[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + d..r * 3 * d + 2 * d]);
            v2[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + 2 * d..r * 3 * d + 3 * d]);
        }
        // RMS over the q slice and k slice separately.
        let mut q = rms(&q2, s, d, w.get(&format!("{pre}.attn1.norm_q.weight")));
        let k = rms(&k2, s, d, w.get(&format!("{pre}.attn1.norm_k.weight")));
        let mut k = k;
        rope_apply(&mut q, s, nh, hd, &rope.cos, &rope.sin);
        rope_apply(&mut k, s, nh, hd, &rope.cos, &rope.sin);
        // Append to this layer's host cache, one frame at a time
        // (token-major [frame_tokens, d] rows each).
        for f in 0..3 {
            let ft = frame_tokens;
            let ks = k[f * ft * d..(f + 1) * ft * d].to_vec();
            let vs = v2[f * ft * d..(f + 1) * ft * d].to_vec();
            caches[li].put(start_frame + f, ks, vs);
        }
        let (k_all, v_all, temporal) = caches[li].kv(nh, hd);
        let sk = temporal.len();
        // q split layout [heads, s, hd].
        let mut q_split = vec![0.0f32; nh * s * hd];
        for r in 0..s {
            for hh in 0..nh {
                for j in 0..hd {
                    q_split[(hh * s + r) * hd + j] = q[r * d + hh * hd + j];
                }
            }
        }
        let mask: Vec<bool> = (0..s)
            .flat_map(|qi| {
                let tq = start_frame + qi / frame_tokens;
                temporal.iter().map(move |&tk| tk <= tq)
            })
            .collect();
        let attn_out = attn(&q_split, &k_all, &v_all, nh, s, sk, hd, Some(&mask));
        let o1 = lin(
            &attn_out,
            s,
            d,
            w.get(&format!("{pre}.attn1.to_out.0.weight")),
            w.get(&format!("{pre}.attn1.to_out.0.bias")),
            false,
        );
        for r in 0..s {
            for j in 0..d {
                htok[r * d + j] += o1[r * d + j] * mods[2 * d + j];
            }
        }
        let normed2 = ln_affine(
            &htok,
            s,
            d,
            w.get(&format!("{pre}.norm2.weight")),
            w.get(&format!("{pre}.norm2.bias")),
        );
        // Cross-attention.
        let q2v = lin(
            &normed2,
            s,
            d,
            w.get(&format!("{pre}.attn2.to_q.weight")),
            w.get(&format!("{pre}.attn2.to_q.bias")),
            false,
        );
        let q2r = rms(&q2v, s, d, w.get(&format!("{pre}.attn2.norm_q.weight")));
        let mut q2s = vec![0.0f32; nh * s * hd];
        for r in 0..s {
            for hh in 0..nh {
                for j in 0..hd {
                    q2s[(hh * s + r) * hd + j] = q2r[r * d + hh * hd + j];
                }
            }
        }
        let (k2, v2t) = &text_kv[li];
        let cross = attn(&q2s, k2, v2t, nh, s, cfg.text_len, hd, None);
        let o2 = lin(
            &cross,
            s,
            d,
            w.get(&format!("{pre}.attn2.to_out.0.weight")),
            w.get(&format!("{pre}.attn2.to_out.0.bias")),
            false,
        );
        for r in 0..s {
            for j in 0..d {
                htok[r * d + j] += o2[r * d + j];
            }
        }
        let normed3 = ln_mod(&htok, s, d, mods, 4, 3);
        let hid = lin(
            &normed3,
            s,
            d,
            w.get(&format!("{pre}.ffn.net.0.proj.weight")),
            w.get(&format!("{pre}.ffn.net.0.proj.bias")),
            true,
        );
        let ffn = lin(
            &hid,
            s,
            cfg.ffn_dim,
            w.get(&format!("{pre}.ffn.net.2.weight")),
            w.get(&format!("{pre}.ffn.net.2.bias")),
            false,
        );
        for r in 0..s {
            for j in 0..d {
                htok[r * d + j] += ffn[r * d + j] * mods[5 * d + j];
            }
        }
    }
    let normed = ln_mod(&htok, s, d, head_mods, 1, 0);
    let feats = lin(
        &normed,
        s,
        d,
        w.get("proj_out.weight"),
        w.get("proj_out.bias"),
        false,
    );
    unpatchify(&feats, cfg.out_channels, t, h, ww, p)
}

pub fn ref_text_kv(cfg: &WanConfig, w: &RefWeights, embeds: &[f32]) -> Vec<(Vec<f32>, Vec<f32>)> {
    let (d, nh, hd, l) = (cfg.dim(), cfg.heads, cfg.head_dim, cfg.text_len);
    let mid = lin(
        embeds,
        l,
        cfg.text_dim,
        w.get("condition_embedder.text_embedder.linear_1.weight"),
        w.get("condition_embedder.text_embedder.linear_1.bias"),
        true,
    );
    let ctx = lin(
        &mid,
        l,
        d,
        w.get("condition_embedder.text_embedder.linear_2.weight"),
        w.get("condition_embedder.text_embedder.linear_2.bias"),
        false,
    );
    (0..cfg.layers)
        .map(|li| {
            let pre = format!("blocks.{li}");
            let kv_w: Vec<f32> = ["attn2.to_k", "attn2.to_v"]
                .iter()
                .flat_map(|m| w.get(&format!("{pre}.{m}.weight")).iter().copied())
                .collect();
            let kv_b: Vec<f32> = ["attn2.to_k", "attn2.to_v"]
                .iter()
                .flat_map(|m| w.get(&format!("{pre}.{m}.bias")).iter().copied())
                .collect();
            let kv = lin(&ctx, l, d, &kv_w, &kv_b, false);
            let mut k_rows = vec![0.0f32; l * d];
            let mut v_rows = vec![0.0f32; l * d];
            for r in 0..l {
                k_rows[r * d..(r + 1) * d].copy_from_slice(&kv[r * 2 * d..r * 2 * d + d]);
                v_rows[r * d..(r + 1) * d].copy_from_slice(&kv[r * 2 * d + d..r * 2 * d + 2 * d]);
            }
            let k_rms = rms(&k_rows, l, d, w.get(&format!("{pre}.attn2.norm_k.weight")));
            // Split layout [heads, l, hd].
            let mut k_split = vec![0.0f32; nh * l * hd];
            let mut v_split = vec![0.0f32; nh * l * hd];
            for r in 0..l {
                for hh in 0..nh {
                    for j in 0..hd {
                        k_split[(hh * l + r) * hd + j] = k_rms[r * d + hh * hd + j];
                        v_split[(hh * l + r) * hd + j] = v_rows[r * d + hh * hd + j];
                    }
                }
            }
            (k_split, v_split)
        })
        .collect()
}

pub fn ref_mods(cfg: &WanConfig, w: &RefWeights, timestep: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
    let d = cfg.dim();
    let emb = lightx2v_model::timestep::sinusoid(timestep, cfg.freq_dim);
    let h = lightx2v_model::timestep::silu(&lightx2v_model::timestep::matvec(
        w.get("condition_embedder.time_embedder.linear_1.weight"),
        Some(w.get("condition_embedder.time_embedder.linear_1.bias")),
        &emb,
    ));
    let temb = lightx2v_model::timestep::matvec(
        w.get("condition_embedder.time_embedder.linear_2.weight"),
        Some(w.get("condition_embedder.time_embedder.linear_2.bias")),
        &h,
    );
    let proj = lightx2v_model::timestep::matvec(
        w.get("condition_embedder.time_proj.weight"),
        Some(w.get("condition_embedder.time_proj.bias")),
        &lightx2v_model::timestep::silu(&temb),
    );
    let blocks = (0..cfg.layers)
        .map(|li| {
            let table = w.get(&format!("blocks.{li}.scale_shift_table"));
            proj.iter().zip(table).map(|(a, c)| a + c).collect()
        })
        .collect();
    let head_table = w.get("scale_shift_table");
    let head: Vec<f32> = (0..2 * d).map(|i| head_table[i] + temb[i % d]).collect();
    (blocks, head)
}
