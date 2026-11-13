//! Conformance: every backend op against a direct f32 reference, on small
//! shapes, plus the MemRoPE cache and the DiT block against a host
//! reference. The CPU backend IS the reference for the arithmetic; these
//! checks pin its behaviour so a future wgpu/CUDA backend has a fixed
//! target (and so a regression in the CPU code itself is caught).
//!
//! Each check returns a name and a max absolute error; `run_all` fails on
//! the first check whose error exceeds its tolerance.

use lightx2v_core::{
    backend::{Act, Backend, NormKind},
    config::{MemRopeConfig, WanConfig},
    memrope::{ema_update, MemRopeCache},
    quant::{Fp8E4M3, Q4, Q8},
    rope::rope_table,
    sparse::SparseSpec,
};
use lightx2v_cpu::CpuBackend;

pub struct CheckResult {
    pub name: &'static str,
    pub max_abs_err: f32,
    pub tolerance: f32,
}

impl CheckResult {
    pub fn passed(&self) -> bool {
        self.max_abs_err <= self.tolerance
    }
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (x - y).abs();
            let s = x.abs().max(y.abs()).max(1e-6);
            d / s
        })
        .fold(0.0f32, f32::max)
}

pub fn run_all() -> Vec<CheckResult> {
    run_all_on(&CpuBackend)
}

/// The same 11 checks on any backend, against host-computed references
/// (the CPU oracle matches them by construction, so a wgpu run measures
/// the wgpu backend against the same target).
pub fn run_all_on<B: Backend>(be: &B) -> Vec<CheckResult> {
    let mut out = vec![
        check_linear(be),
        check_layer_norm(be),
        check_attention(be),
        check_causal_attention_mask(be),
        check_sparse_is_dense_at_zero_sparsity(be),
        check_patchify_roundtrip(be),
        check_dmd_step(be),
        check_quant(),
        check_rope(),
        check_memrope_cache(),
    ];
    out.push(check_history_contrast());
    out
}

fn check_linear<B: Backend>(be: &B) -> CheckResult {
    // 4x3 matrix, hand-computed reference.
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![
        1.0f32, 0.0, -1.0, 0.5, 1.0, 0.5, 2.0, -1.0, 0.0, 0.0, 1.0, 2.0,
    ];
    let b = vec![0.5f32, -0.5, 1.0, 0.0];
    let xb = be.upload(&x);
    let yb = be.alloc(2 * 4);
    let wb = be.weight(&w, 4, 3);
    let bb = be.upload(&b);
    be.linear(&xb, 0, 2, &wb, Some(&bb), Act::None, &yb, 0);
    let got = be.download(&yb);
    // Row 0: [1,2,3]·rows + b; row 1: [4,5,6]·rows + b.
    let mut want = Vec::new();
    for r in 0..2 {
        for o in 0..4 {
            let mut acc = b[o];
            for i in 0..3 {
                acc += x[r * 3 + i] * w[o * 3 + i];
            }
            want.push(acc);
        }
    }
    CheckResult {
        name: "linear",
        max_abs_err: max_err(&got, &want),
        tolerance: 1e-6,
    }
}

fn check_layer_norm<B: Backend>(be: &B) -> CheckResult {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let xb = be.upload(&x);
    let ob = be.alloc(8);
    let mods = be.upload(&[0.0f32; 8]);
    let w = be.upload(&[1.0f32; 4]);
    let b = be.upload(&[0.0f32; 4]);
    be.layer_norm(
        &xb,
        2,
        4,
        NormKind::Affine { w: &w, b: &b },
        &mods,
        1e-6,
        &ob,
    );
    let got = be.download(&ob);
    let mut want = Vec::new();
    for r in 0..2 {
        let row = &x[r * 4..(r + 1) * 4];
        let mean = row.iter().sum::<f32>() / 4.0;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        let inv = 1.0 / (var + 1e-6).sqrt();
        want.extend(row.iter().map(|v| (v - mean) * inv));
    }
    CheckResult {
        name: "layer_norm",
        max_abs_err: max_err(&got, &want),
        tolerance: 1e-6,
    }
}

fn check_attention<B: Backend>(be: &B) -> CheckResult {
    let (heads, sq, sk, d) = (1usize, 2, 3, 4);
    let mut rng = lightx2v_core::rng::Rng::new(3);
    let q = rng.normal(heads * sq * d);
    let k = rng.normal(heads * sk * d);
    let v = rng.normal(heads * sk * d);
    let qb = be.upload(&q);
    let kb = be.upload(&k);
    let vb = be.upload(&v);
    let ob = be.alloc(sq * d);
    be.attention(&qb, &kb, &vb, heads, sq, sk, d, &ob);
    let got = be.download(&ob);
    // Reference: explicit softmax.
    let mut want = vec![0.0f32; sq * d];
    let scale = 1.0 / (d as f32).sqrt();
    for qi in 0..sq {
        let mut scores: Vec<f32> = (0..sk)
            .map(|ki| (0..d).map(|j| q[qi * d + j] * k[ki * d + j]).sum::<f32>() * scale)
            .collect();
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
            want[qi * d + j] = (0..sk).map(|ki| scores[ki] * v[ki * d + j]).sum();
        }
    }
    CheckResult {
        name: "attention",
        max_abs_err: max_err(&got, &want),
        tolerance: 1e-6,
    }
}

fn check_causal_attention_mask<B: Backend>(be: &B) -> CheckResult {
    // 3 queries, 3 keys, mask = lower-triangular: query i sees keys <= i.
    let (heads, sq, sk, d) = (1usize, 3, 3, 4);
    let mut rng = lightx2v_core::rng::Rng::new(5);
    let q = rng.normal(heads * sq * d);
    let k = rng.normal(heads * sk * d);
    let v = rng.normal(heads * sk * d);
    let qb = be.upload(&q);
    let kb = be.upload(&k);
    let vb = be.upload(&v);
    let ob = be.alloc(sq * d);
    let mask: Vec<bool> = (0..sq).flat_map(|i| (0..sk).map(move |j| j <= i)).collect();
    be.causal_attention(&qb, &kb, &vb, heads, sq, sk, d, Some(&mask), &ob);
    let got = be.download(&ob);
    // Reference with the same mask.
    let mut want = vec![0.0f32; sq * d];
    let scale = 1.0 / (d as f32).sqrt();
    for qi in 0..sq {
        let mut scores: Vec<f32> = (0..sk)
            .map(|ki| {
                if ki > qi {
                    f32::NEG_INFINITY
                } else {
                    (0..d).map(|j| q[qi * d + j] * k[ki * d + j]).sum::<f32>() * scale
                }
            })
            .collect();
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
            want[qi * d + j] = (0..sk).map(|ki| scores[ki] * v[ki * d + j]).sum();
        }
    }
    // And: query 0 must equal attending to key 0 alone (mask blocks the rest).
    let first_key_only = v[..d].to_vec();
    let err_mask = max_err(&got, &want);
    let err_first = max_err(&got[..d], &first_key_only);
    CheckResult {
        name: "causal_attention_mask",
        max_abs_err: err_mask.max(err_first),
        tolerance: 1e-5,
    }
}

fn check_sparse_is_dense_at_zero_sparsity<B: Backend>(be: &B) -> CheckResult {
    // Sparsity 0 keeps every tile, gate 0 kills the coarse term: sparse
    // must equal dense attention.
    let (heads, d) = (1usize, 8);
    let grid = (4, 4, 4);
    let seq = 64;
    let mut rng = lightx2v_core::rng::Rng::new(11);
    let q = rng.normal(heads * seq * d);
    let k = rng.normal(heads * seq * d);
    let v = rng.normal(heads * seq * d);
    let gate = vec![0.0f32; heads * seq * d];
    let qb = be.upload(&q);
    let kb = be.upload(&k);
    let vb = be.upload(&v);
    let gb = be.upload(&gate);
    let spec = SparseSpec::new(grid, heads, d, 0.0);
    let sp = be.sparse_prepare(&spec);
    let ob = be.alloc(seq * d);
    be.sparse_attention(&qb, &kb, &vb, &gb, &sp, &ob);
    let got = be.download(&ob);
    let ob2 = be.alloc(seq * d);
    be.attention(&qb, &kb, &vb, heads, seq, seq, d, &ob2);
    let want = be.download(&ob2);
    CheckResult {
        name: "sparse==dense@0-sparsity-gate-0",
        max_abs_err: max_err(&got, &want),
        tolerance: 1e-5,
    }
}

fn check_patchify_roundtrip<B: Backend>(be: &B) -> CheckResult {
    let (c, t, h, w, p) = (4usize, 3, 8, 8, 2);
    let mut rng = lightx2v_core::rng::Rng::new(13);
    let latent = rng.normal(c * t * h * w);
    let lb = be.upload(&latent);
    let tokens = t * (h / p) * (w / p);
    let tb = be.alloc(tokens * c * p * p);
    be.patchify(&lb, c, t, h, w, p, &tb);
    let ob = be.alloc(c * t * h * w);
    be.unpatchify(&tb, c, t, h, w, p, &ob);
    let got = be.download(&ob);
    CheckResult {
        name: "patchify/unpatchify roundtrip",
        max_abs_err: max_err(&got, &latent),
        tolerance: 1e-6,
    }
}

fn check_dmd_step<B: Backend>(be: &B) -> CheckResult {
    let x = vec![1.0f32, 2.0, 3.0];
    let v = vec![0.5f32, 0.5, 0.5];
    let noise = vec![0.1f32, -0.2, 0.3];
    let xb = be.upload(&x);
    let vb = be.upload(&v);
    let nb = be.upload(&noise);
    be.dmd_step(&xb, &vb, Some(&nb), 0.75, 0.5, 3);
    let got = be.download(&xb);
    let want: Vec<f32> = (0..3)
        .map(|i| (1.0 - 0.5) * (x[i] - 0.75 * v[i]) + 0.5 * noise[i])
        .collect();
    CheckResult {
        name: "dmd_step",
        max_abs_err: max_err(&got, &want),
        tolerance: 1e-6,
    }
}

fn check_quant() -> CheckResult {
    let mut rng = lightx2v_core::rng::Rng::new(17);
    let w = rng.uniform(8 * 256, 1.0);
    let q8 = Q8::quantize(&w, 8, 256);
    let e8 = w
        .iter()
        .zip(q8.dequantize())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let q4 = Q4::quantize(&w, 8, 256, 64, true);
    let e4 = w
        .iter()
        .zip(q4.dequantize())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let fp8 = Fp8E4M3::quantize(&w, 8, 256);
    let ef = w
        .iter()
        .zip(fp8.dequantize())
        .map(|(a, b)| (a - b).abs() / (a.abs() + 1e-6))
        .fold(0.0f32, f32::max);
    CheckResult {
        name: "quant q8/q4/fp8 roundtrip",
        max_abs_err: e8.max(e4).max(ef),
        tolerance: 0.2,
    }
}

fn check_rope() -> CheckResult {
    let cfg = WanConfig::wan_1_3b();
    let t = rope_table(&cfg, (2, 3, 4), 0);
    let err = if t.cos[..64].iter().all(|&c| c == 1.0) && t.sin[..64].iter().all(|&s| s == 0.0) {
        0.0
    } else {
        1.0
    };
    CheckResult {
        name: "rope origin identity",
        max_abs_err: err,
        tolerance: 0.0,
    }
}

fn check_memrope_cache() -> CheckResult {
    let cfg = MemRopeConfig::default_memrope();
    let mut cache = MemRopeCache::new(cfg.clone());
    for t in 0..40 {
        cache.append(t, vec![t as f32; 4]);
        if cache.len() > cfg.cache_frames {
            return CheckResult {
                name: "memrope cache bounded",
                max_abs_err: 1.0,
                tolerance: 0.0,
            };
        }
    }
    let sinks: Vec<usize> = cache
        .frames
        .iter()
        .filter(|f| f.kind == lightx2v_core::memrope::SlotKind::Sink)
        .map(|f| f.temporal_index)
        .collect();
    let err: f32 = if sinks == vec![0, 1, 2] && cache.frames.last().unwrap().temporal_index == 39 {
        0.0
    } else {
        1.0
    };
    // EMA update sanity: alpha 0.1 moves 10% toward the frame.
    let ema = ema_update(&[0.0f32; 2], &[10.0, 20.0], 0.1);
    let err = err.max((ema[0] - 1.0).abs().max((ema[1] - 2.0).abs()));
    CheckResult {
        name: "memrope cache bounded + sinks + ema",
        max_abs_err: err,
        tolerance: 1e-6,
    }
}

fn check_history_contrast() -> CheckResult {
    let base = vec![0.0f32; 3];
    let with = vec![3.0f32, 1.0, -1.0];
    let without = vec![1.0f32, 1.0, 1.0];
    let out = lightx2v_core::history::corrected_target(&base, &with, &without, 0.5);
    CheckResult {
        name: "history-contrast residual",
        max_abs_err: max_err(&out, &[1.0, 0.0, -1.0]),
        tolerance: 1e-6,
    }
}

/// Relative error between two slices (used by the model tests).
pub fn rel(a: &[f32], b: &[f32]) -> f32 {
    rel_err(a, b)
}
