//! Weight-only quantization of the DiT block linears: int8 ([`Q8`]) and
//! 4-bit ([`Q4`]), which linears get which ([`WeightQuant::for_key`]), and
//! the LightX2V FP8 path ([`Fp8E4M3`]).
//!
//! `Q8`/`Q4` mirror the semantics of fastwan-rs's `quant.rs`: symmetric
//! round-to-nearest with one f32 scale per group of [`Q8_GROUP`] input
//! columns (`w ≈ q · s`, `q ∈ [-127, 127]`), and affine 4-bit codes with an
//! HQQ-refined zero point. Activations stay in the backend's precision;
//! every backend computes `Σ x · (q · s)` with `q · s` formed in f32, so
//! [`Q8::dequantize`] / [`Q4::dequantize`] are the exact references the GPU
//! kernels are checked against.
//!
//! `Fp8E4M3` is the LightX2V FP8 path (NVFP4/FP8 quant): e4m3 codes with one
//! f32 scale per block of [`FP8_GROUP`] elements, `w ≈ code · s`. Phase 1
//! implements quantize/dequantize as the reference contract; the NVFP4
//! micro-block kernel (2-level scaling) lands in Phase 2.

/// Columns sharing one Q8 scale.
pub const Q8_GROUP: usize = 128;
/// Elements sharing one FP8 scale.
pub const FP8_GROUP: usize = 128;
/// Largest finite e4m3 magnitude (0b0_1111_110 = 448).
pub const FP8_E4M3_MAX: f32 = 448.0;

/// How the DiT's transformer-block weights are stored on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightQuant {
    /// The backend's own precision (bf16 on CUDA, f16 pairs on WebGPU).
    #[default]
    Native,
    /// [`Q8`]: half the bytes of bf16 / f16.
    Int8,
    /// 4-bit [`Q4`] with `group` columns per scale/zero, rounded to nearest
    /// or HQQ-refined. With `ffn_only`, the attention and gate linears use
    /// [`Q8`] and only the FFN is 4-bit.
    Int4 {
        group: usize,
        hqq: bool,
        ffn_only: bool,
    },
    /// FP8 e4m3, one f32 scale per [`FP8_GROUP`] elements (LightX2V FP8 path).
    Fp8,
}

impl WeightQuant {
    /// The format of the non-FFN block linears.
    pub fn attn(self) -> Self {
        match self {
            Self::Int4 { ffn_only: true, .. } => Self::Int8,
            q => q,
        }
    }

    /// The format of the block linear `key` (`blocks.N.<role>`, no
    /// `.weight`): the FFN gets `self`, attention and gate get [`Self::attn`],
    /// and everything outside the blocks is `Native`.
    pub fn for_key(self, key: &str) -> Self {
        let Some(role) = key
            .strip_prefix("blocks.")
            .and_then(|r| r.split_once('.'))
            .map(|(_, r)| r)
        else {
            return Self::Native;
        };
        match role {
            "ffn.net.0.proj" | "ffn.net.2" => self,
            "attn1.to_q" | "attn1.to_k" | "attn1.to_v" | "attn1.to_out.0" | "attn2.to_q"
            | "attn2.to_k" | "attn2.to_v" | "attn2.to_out.0" | "to_gate_compress" => self.attn(),
            _ => Self::Native,
        }
    }
}

impl std::fmt::Display for WeightQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => f.write_str("native"),
            Self::Int8 => f.write_str("int8"),
            Self::Fp8 => f.write_str("fp8-e4m3"),
            Self::Int4 {
                group,
                hqq,
                ffn_only,
            } => write!(
                f,
                "int4{}-g{group}{}",
                if *hqq { "-hqq" } else { "" },
                if *ffn_only { "-ffn" } else { "" }
            ),
        }
    }
}

impl std::str::FromStr for WeightQuant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "native" | "bf16" | "f16" => Ok(Self::Native),
            "int8" | "q8" => Ok(Self::Int8),
            "fp8" | "fp8-e4m3" => Ok(Self::Fp8),
            _ => {
                // int4[-hqq][-g<N>][-ffn]
                let (s, ffn_only) = match s.strip_suffix("-ffn") {
                    Some(r) => (r, true),
                    None => (s, false),
                };
                let rest = s.strip_prefix("int4").ok_or_else(|| {
                    format!(
                        "unknown weight format `{s}` (native | int8 | fp8 | int4[-hqq][-gN][-ffn])"
                    )
                })?;
                let (hqq, rest) = match rest.strip_prefix("-hqq") {
                    Some(r) => (true, r),
                    None => (false, rest),
                };
                let group = match rest.strip_prefix("-g") {
                    Some(g) => g.parse().map_err(|_| format!("bad group in `{s}`"))?,
                    None if rest.is_empty() => 64,
                    None => return Err(format!("bad weight format `{s}`")),
                };
                if group == 0 || group % 8 != 0 {
                    return Err(format!("int4 group must be a multiple of 8, got {group}"));
                }
                Ok(Self::Int4 {
                    group,
                    hqq,
                    ffn_only,
                })
            }
        }
    }
}

/// A block linear as stored in a quantized checkpoint.
#[derive(Debug, Clone)]
pub enum Stored {
    Q8(Q8),
    Q4(Q4),
    Fp8(Fp8E4M3),
}

/// A `[rows, cols]` matrix as int8 codes and per-group scales.
#[derive(Debug, Clone)]
pub struct Q8 {
    pub q: Vec<i8>,
    /// `[rows, cols / group]`.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

impl Q8 {
    pub fn quantize(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let group = if cols.is_multiple_of(Q8_GROUP) {
            Q8_GROUP
        } else {
            cols
        };
        let mut q = vec![0i8; data.len()];
        let mut scales = Vec::with_capacity(data.len() / group);
        for (src, dst) in data.chunks(group).zip(q.chunks_mut(group)) {
            let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (d, v) in dst.iter_mut().zip(src) {
                *d = (v / s).round().clamp(-127.0, 127.0) as i8;
            }
            scales.push(s);
        }
        Self {
            q,
            scales,
            rows,
            cols,
            group,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &q)| q as f32 * self.scales[i / self.group])
            .collect()
    }

    /// Stack matrices with the same columns and group by rows (fused QKV).
    pub fn concat(parts: Vec<Q8>) -> Self {
        let mut it = parts.into_iter();
        let mut a = it.next().expect("at least one matrix");
        for b in it {
            assert_eq!((a.cols, a.group), (b.cols, b.group));
            a.q.extend(b.q);
            a.scales.extend(b.scales);
            a.rows += b.rows;
        }
        a
    }

    /// Device bytes (codes + scales).
    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scales.len()
    }
}

/// A `[rows, cols]` matrix as unsigned 4-bit codes with an affine
/// scale / zero per group of columns: `w ≈ (q − z) · s`, `q ∈ [0, 15]`.
/// Scales and zeros are rounded to f16 when made, which is how they are
/// stored, so a quantized file and an in-memory quantization are identical.
#[derive(Debug, Clone)]
pub struct Q4 {
    /// One code per byte (packing is a storage concern; see [`Q4::bytes`]).
    pub q: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

impl Q4 {
    /// Min/max round-to-nearest; with `hqq`, then refine each group's zero
    /// point with HQQ's half-quadratic proximal solver (Badri & Shaji 2023:
    /// lp-norm 0.7, β 10, κ 1.01, ≤ 20 iters).
    pub fn quantize(data: &[f32], rows: usize, cols: usize, group: usize, hqq: bool) -> Self {
        assert_eq!(data.len(), rows * cols);
        let group = if cols.is_multiple_of(group) {
            group
        } else {
            cols
        };
        let n = data.len() / group;
        let mut q = vec![0u8; data.len()];
        let mut sz = vec![(0.0f32, 0.0f32); n];
        for ((w, qg), sz) in data
            .chunks(group)
            .zip(q.chunks_mut(group))
            .zip(sz.iter_mut())
        {
            let (lo, hi) = w
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            // HQQ's convention: inverse scale `is`, codes = round(w·is + z).
            let is = if hi > lo { 15.0 / (hi - lo) } else { 1.0 };
            let mut z = -lo * is;
            let code = |v: f32, z: f32| (v * is + z).round().clamp(0.0, 15.0);
            if hqq {
                let (p, mut beta, kappa) = (0.7f32, 10.0f32, 1.01f32);
                let (mut best, mut best_z) = (f32::MAX, z);
                for _ in 0..20 {
                    let mut err = 0.0;
                    let mut zsum = 0.0;
                    for &v in w {
                        let c = code(v, z);
                        let r = (c - z) / is;
                        let e = v - r;
                        err += e.abs();
                        let a = e.abs();
                        let sh = if a > 0.0 {
                            e.signum() * (a - a.powf(p - 1.0) / beta).max(0.0)
                        } else {
                            0.0
                        };
                        zsum += c - (v - sh) * is;
                    }
                    let err = err / w.len() as f32;
                    if err >= best {
                        break;
                    }
                    (best, best_z) = (err, z);
                    z = zsum / w.len() as f32;
                    beta *= kappa;
                }
                z = best_z;
            }
            for (d, &v) in qg.iter_mut().zip(w) {
                *d = code(v, z) as u8;
            }
            let f = |v: f32| half::f16::from_f32(v).to_f32();
            *sz = (f(1.0 / is), f(z));
        }
        let (scales, zeros) = sz.into_iter().unzip();
        Self {
            q,
            scales,
            zeros,
            rows,
            cols,
            group,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let g = i / self.group;
                (c as f32 - self.zeros[g]) * self.scales[g]
            })
            .collect()
    }

    /// Two codes per byte, element `2i` in the low nibble.
    pub fn packed(&self) -> Vec<u8> {
        self.q
            .chunks(2)
            .map(|c| c[0] | (c.get(1).copied().unwrap_or(0) << 4))
            .collect()
    }

    pub fn from_packed(
        packed: &[u8],
        scales: Vec<f32>,
        zeros: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Self {
        let n = rows * cols;
        assert_eq!(packed.len(), n.div_ceil(2));
        assert!(!scales.is_empty() && scales.len() == zeros.len());
        assert_eq!(n % scales.len(), 0);
        let q = (0..n)
            .map(|i| (packed[i / 2] >> (4 * (i % 2))) & 15)
            .collect();
        let group = n / scales.len();
        Self {
            q,
            scales,
            zeros,
            rows,
            cols,
            group,
        }
    }

    /// Stack matrices with the same columns and group by rows (fused QKV).
    pub fn concat(parts: Vec<Q4>) -> Self {
        let mut it = parts.into_iter();
        let mut a = it.next().expect("at least one matrix");
        for b in it {
            assert_eq!((a.cols, a.group), (b.cols, b.group));
            a.q.extend(b.q);
            a.scales.extend(b.scales);
            a.zeros.extend(b.zeros);
            a.rows += b.rows;
        }
        a
    }

    /// Bytes once packed: two codes per byte, f16 scale and zero per group.
    pub fn bytes(&self) -> usize {
        self.q.len().div_ceil(2) + 4 * self.scales.len()
    }
}

/// FP8 e4m3 (1 sign, 4 exponent, 3 mantissa, bias 7; NaN is the 0x7F/0xFF
/// code and is never produced here). `w ≈ code · s` with one f32 scale per
/// [`FP8_GROUP`] elements — the LightX2V FP8 path's reference contract.
/// The NVFP4 micro-block form (fp4 codes + fp8 block scales + fp32 global
/// scale) is Phase 2 work; this type is the dequantize-as-reference oracle
/// it will be checked against.
#[derive(Debug, Clone)]
pub struct Fp8E4M3 {
    /// One e4m3 code byte per element.
    pub q: Vec<u8>,
    /// One scale per group of [`FP8_GROUP`] elements.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

/// Encode one finite f32 as e4m3 (round-to-nearest-even on the mantissa).
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v == 0.0 {
        return if v.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0u8 };
    let a = v.abs();
    if !a.is_finite() || a >= 464.0 {
        // Saturate to the largest finite e4m3 (448).
        return sign | 0x7E;
    }
    // Exponent: floor(log2(a)) clamped to the e4m3 range.
    let e = a.log2().floor() as i32;
    if e < -9 {
        // Subnormal or zero: value = m/8 · 2^-9, m in 0..=7.
        let m = (a / (2f32.powi(-9) / 8.0)).round() as u8;
        return sign | m.min(7);
    }
    if e > 8 {
        return sign | 0x7E;
    }
    let exp = (e + 7) as u8;
    let mant = ((a / 2f32.powi(e) - 1.0) * 8.0).round() as u8;
    if mant >= 8 {
        // Rounded up into the next exponent.
        if exp >= 15 {
            return sign | 0x7E;
        }
        return sign | ((exp + 1) << 3);
    }
    sign | (exp << 3) | mant
}

/// Decode one e4m3 code byte to f32.
pub fn e4m3_to_f32(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (code >> 3) & 0xF;
    let mant = code & 0x7;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-9)
    } else if exp == 0xF && mant == 0x7 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    }
}

impl Fp8E4M3 {
    pub fn quantize(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let group = if cols.is_multiple_of(FP8_GROUP) {
            FP8_GROUP
        } else {
            cols
        };
        let mut q = vec![0u8; data.len()];
        let mut scales = Vec::with_capacity(data.len() / group);
        for (src, dst) in data.chunks(group).zip(q.chunks_mut(group)) {
            let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / FP8_E4M3_MAX } else { 1.0 };
            for (d, v) in dst.iter_mut().zip(src) {
                *d = f32_to_e4m3(v / s);
            }
            scales.push(s);
        }
        Self {
            q,
            scales,
            rows,
            cols,
            group,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &c)| e4m3_to_f32(c) * self.scales[i / self.group])
            .collect()
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scales.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_round_trip_error_is_half_a_step() {
        let (rows, cols) = (8, 256);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1 + i / cols) as f32)
            .collect();
        let q = Q8::quantize(&w, rows, cols);
        assert_eq!(q.group, 128);
        assert_eq!(q.scales.len(), rows * 2);
        let d = q.dequantize();
        for (i, (a, b)) in w.iter().zip(&d).enumerate() {
            assert!((a - b).abs() <= q.scales[i / 128] * 0.5 + 1e-6);
        }
    }

    #[test]
    fn q4_round_trip_and_hqq_does_not_hurt() {
        let (rows, cols) = (16, 256);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let u = (i * 7919 % 1009) as f32 / 1009.0 - 0.5;
                if i % 97 == 0 {
                    u * 8.0
                } else {
                    u * 0.3
                }
            })
            .collect();
        let err = |q: &Q4| -> f32 {
            w.iter()
                .zip(q.dequantize())
                .map(|(a, b)| (a - b).powi(2))
                .sum()
        };
        let rtn = Q4::quantize(&w, rows, cols, 64, false);
        assert!(rtn.q.iter().all(|&c| c <= 15));
        for (i, (a, b)) in w.iter().zip(rtn.dequantize()).enumerate() {
            assert!((a - b).abs() <= rtn.scales[i / 64] * 0.52 + 1e-5);
        }
        let hqq = Q4::quantize(&w, rows, cols, 64, true);
        assert!(hqq.q.iter().all(|&c| c <= 15));
        assert!(
            err(&hqq) <= err(&rtn) * 1.05,
            "{} vs {}",
            err(&hqq),
            err(&rtn)
        );
        assert_eq!(
            "int4-hqq-g32".parse(),
            Ok(WeightQuant::Int4 {
                group: 32,
                hqq: true,
                ffn_only: false
            })
        );
        assert_eq!("fp8".parse::<WeightQuant>(), Ok(WeightQuant::Fp8));
        assert_eq!(
            "int4-hqq-g64-ffn".parse::<WeightQuant>().map(|q| q.attn()),
            Ok(WeightQuant::Int8)
        );
    }

    #[test]
    fn e4m3_codec_spot_values() {
        assert_eq!(f32_to_e4m3(0.0), 0x00);
        assert_eq!(f32_to_e4m3(1.0), 0x38); // exp 7, mant 0
        assert_eq!(f32_to_e4m3(-1.0), 0xB8);
        assert_eq!(f32_to_e4m3(448.0), 0x7E);
        assert_eq!(f32_to_e4m3(2.0), 0x40);
        assert!((e4m3_to_f32(0x38) - 1.0).abs() < 1e-7);
        assert!((e4m3_to_f32(0x40) - 2.0).abs() < 1e-7);
        assert!((e4m3_to_f32(0x01) - 2f32.powi(-9) / 8.0).abs() < 1e-9);
    }

    #[test]
    fn fp8_round_trip_error_is_half_an_e4m3_step() {
        let (rows, cols) = (4, 256);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.7)
            .collect();
        let q = Fp8E4M3::quantize(&w, rows, cols);
        let d = q.dequantize();
        for (i, (a, b)) in w.iter().zip(&d).enumerate() {
            // e4m3 step is at most 1/8 of the value's magnitude (3 mantissa
            // bits), plus the scale's own rounding.
            let tol = a.abs() * 0.0625 + q.scales[i / 128] * 0.5 + 1e-6;
            assert!((a - b).abs() <= tol, "{a} vs {b}");
        }
    }
}
