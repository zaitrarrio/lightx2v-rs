//! Checkpoint access: Diffusers safetensors (mmap'd, any of F32/F16/BF16,
//! any number of shards) or deterministic synthetic weights of the same
//! names and shapes (benchmarks and parity tests without a download).
//!
//! The model asks for every tensor by its exact Diffusers key and shape. A
//! missing key or a shape mismatch is an error: a silently-defaulted weight
//! streams gray haze instead of failing.

use crate::{
    config::{TaeConfig, WanConfig},
    rng::Rng,
    Error, Result,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub trait WeightSource {
    /// The tensor as f32, checked against `shape` (size-1 dims are ignored,
    /// so a Conv3d `[o, i, 1, 2, 2]` may be asked for as `[o, i, 2, 2]`).
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>>;
    fn contains(&self, name: &str) -> bool;
    /// The linear `key` (no `.weight`) if the checkpoint stores it quantized
    /// (`<key>.weight.q8` / `.q4` / `.fp8` + `.scale` [+ `.zero`]);
    /// `None` for a plain tensor.
    fn stored(&self, _key: &str) -> Result<Option<crate::quant::Stored>> {
        Ok(None)
    }
    /// Whether the linear `key` (no `.weight`) is present, plain or quantized.
    fn has_linear(&self, key: &str) -> bool {
        self.contains(&format!("{key}.weight"))
            || self.contains(&format!("{key}.weight.q8"))
            || self.contains(&format!("{key}.weight.q4"))
            || self.contains(&format!("{key}.weight.fp8"))
    }
}

fn squeeze(s: &[usize]) -> Vec<usize> {
    s.iter().copied().filter(|&d| d != 1).collect()
}

/// A safetensors file's bytes: mapped from disk, or owned (a buffer fetched
/// by the browser).
enum Blob {
    Map(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Blob::Map(m) => m,
            Blob::Owned(v) => v,
        }
    }
}

/// All `*.safetensors` under a directory (or one file), or one in memory.
pub struct SafetensorsDir {
    maps: Vec<Blob>,
    index: HashMap<String, usize>,
}

impl SafetensorsDir {
    /// A complete safetensors file held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let index = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| Error::Weights(format!("in-memory safetensors: {e}")))?
            .names()
            .into_iter()
            .map(|n| (n.to_string(), 0))
            .collect();
        Ok(Self {
            maps: vec![Blob::Owned(bytes)],
            index,
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        let files: Vec<PathBuf> = if path.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            v.sort();
            v
        } else {
            vec![path.to_path_buf()]
        };
        if files.is_empty() {
            return Err(Error::Weights(format!(
                "no .safetensors in {}",
                path.display()
            )));
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f)?;
            let map = map_readonly(&file)?;
            let st = safetensors::SafeTensors::deserialize(&map)
                .map_err(|e| Error::Weights(format!("{}: {e}", f.display())))?;
            for name in st.names() {
                index.insert(name.to_string(), i);
            }
            maps.push(Blob::Map(map));
        }
        Ok(Self { maps, index })
    }

    /// The stored bytes of a tensor, without conversion: `(dtype, shape, data)`.
    pub fn raw(&self, name: &str) -> Result<(safetensors::Dtype, Vec<usize>, &[u8])> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        let st = safetensors::SafeTensors::deserialize(&self.maps[i])
            .map_err(|e| Error::Weights(e.to_string()))?;
        let t = st.tensor(name).map_err(|e| Error::Weights(e.to_string()))?;
        Ok((t.dtype(), t.shape().to_vec(), t.data()))
    }

    pub fn contains_raw(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }
}

#[allow(unsafe_code)]
fn map_readonly(file: &std::fs::File) -> Result<memmap2::Mmap> {
    // SAFETY: weights are read-only inputs; concurrent truncation by another
    // process is outside this program's contract (as for every mmap loader).
    Ok(unsafe { memmap2::Mmap::map(file)? })
}

impl WeightSource for SafetensorsDir {
    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn stored(&self, key: &str) -> Result<Option<crate::quant::Stored>> {
        use crate::quant::{Fp8E4M3, Stored, Q4, Q8};
        use safetensors::Dtype;
        let f32s = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let (_, shape, _) = self.raw(name)?;
            let v = self.get(name, &shape)?;
            Ok((shape, v))
        };
        let bad = |m: String| Error::Weights(format!("`{key}`: {m}"));
        let q8 = format!("{key}.weight.q8");
        let q4 = format!("{key}.weight.q4");
        let fp8 = format!("{key}.weight.fp8");
        if self.contains_raw(&q8) {
            let (dt, shape, data) = self.raw(&q8)?;
            if dt != Dtype::I8 || shape.len() != 2 {
                return Err(bad(format!("q8 is {dt:?} {shape:?}")));
            }
            let (sshape, scales) = f32s(&format!("{key}.weight.scale"))?;
            if sshape.len() != 2 || sshape[0] != shape[0] || shape[1] % sshape[1] != 0 {
                return Err(bad(format!("scale {sshape:?} for codes {shape:?}")));
            }
            return Ok(Some(Stored::Q8(Q8 {
                q: data.iter().map(|&b| b as i8).collect(),
                scales,
                rows: shape[0],
                cols: shape[1],
                group: shape[1] / sshape[1],
            })));
        }
        if self.contains_raw(&q4) {
            let (dt, shape, data) = self.raw(&q4)?;
            if dt != Dtype::U8 || shape.len() != 2 {
                return Err(bad(format!("q4 is {dt:?} {shape:?}")));
            }
            let (rows, cols) = (shape[0], 2 * shape[1]);
            let (sshape, scales) = f32s(&format!("{key}.weight.scale"))?;
            let (zshape, zeros) = f32s(&format!("{key}.weight.zero"))?;
            if sshape != zshape || sshape.len() != 2 || sshape[0] != rows || cols % sshape[1] != 0 {
                return Err(bad(format!(
                    "scale {sshape:?} / zero {zshape:?} for codes {shape:?}"
                )));
            }
            return Ok(Some(Stored::Q4(Q4::from_packed(
                data, scales, zeros, rows, cols,
            ))));
        }
        if self.contains_raw(&fp8) {
            let (dt, shape, data) = self.raw(&fp8)?;
            if dt != Dtype::U8 || shape.len() != 2 {
                return Err(bad(format!("fp8 is {dt:?} {shape:?}")));
            }
            let (sshape, scales) = f32s(&format!("{key}.weight.scale"))?;
            if sshape.len() != 2 || sshape[0] != shape[0] || shape[1] % sshape[1] != 0 {
                return Err(bad(format!("scale {sshape:?} for codes {shape:?}")));
            }
            return Ok(Some(Stored::Fp8(Fp8E4M3 {
                q: data.to_vec(),
                scales,
                rows: shape[0],
                cols: shape[1],
                group: shape[1] / sshape[1],
            })));
        }
        Ok(None)
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        let st = safetensors::SafeTensors::deserialize(&self.maps[i])
            .map_err(|e| Error::Weights(e.to_string()))?;
        let t = st.tensor(name).map_err(|e| Error::Weights(e.to_string()))?;
        if squeeze(t.shape()) != squeeze(shape) {
            return Err(Error::Weights(format!(
                "`{name}` is {:?}, expected {shape:?}",
                t.shape()
            )));
        }
        let b = t.data();
        use safetensors::Dtype;
        Ok(match t.dtype() {
            Dtype::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Dtype::F16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            Dtype::BF16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                .collect(),
            other => {
                return Err(Error::Weights(format!(
                    "`{name}`: unsupported dtype {other:?}"
                )));
            }
        })
    }
}

/// Deterministic weights for every name the model asks for: `U(−a, a)` with
/// `a = 1/√fan_in` for matrices, ones for norm weights, small values for the
/// modulation tables. Same name → same values on every backend.
pub struct Synthetic {
    pub seed: u64,
}

impl WeightSource for Synthetic {
    fn contains(&self, _name: &str) -> bool {
        true
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let n: usize = shape.iter().product();
        let mut h = self.seed;
        for b in name.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3);
        }
        let mut rng = Rng::new(h);
        let norm_weight = name.ends_with("norm_q.weight")
            || name.ends_with("norm_k.weight")
            || name.ends_with("norm2.weight");
        Ok(if norm_weight {
            rng.uniform(n, 0.1).into_iter().map(|v| 1.0 + v).collect()
        } else if name.contains("scale_shift_table") || name.ends_with(".bias") {
            rng.uniform(n, 0.1)
        } else {
            let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
            rng.uniform(n, 1.0 / (fan_in as f32).sqrt())
        })
    }
}

/// Every tensor the Wan DiT reads, with its Diffusers shape.
pub fn wan_manifest(cfg: &WanConfig) -> Vec<(String, Vec<usize>)> {
    let d = cfg.dim();
    let mut v: Vec<(String, Vec<usize>)> = vec![
        (
            "patch_embedding.weight".into(),
            vec![d, cfg.in_channels, 1, cfg.patch, cfg.patch],
        ),
        ("patch_embedding.bias".into(), vec![d]),
        (
            "condition_embedder.time_embedder.linear_1.weight".into(),
            vec![d, cfg.freq_dim],
        ),
        (
            "condition_embedder.time_embedder.linear_1.bias".into(),
            vec![d],
        ),
        (
            "condition_embedder.time_embedder.linear_2.weight".into(),
            vec![d, d],
        ),
        (
            "condition_embedder.time_embedder.linear_2.bias".into(),
            vec![d],
        ),
        ("condition_embedder.time_proj.weight".into(), vec![6 * d, d]),
        ("condition_embedder.time_proj.bias".into(), vec![6 * d]),
        (
            "condition_embedder.text_embedder.linear_1.weight".into(),
            vec![d, cfg.text_dim],
        ),
        (
            "condition_embedder.text_embedder.linear_1.bias".into(),
            vec![d],
        ),
        (
            "condition_embedder.text_embedder.linear_2.weight".into(),
            vec![d, d],
        ),
        (
            "condition_embedder.text_embedder.linear_2.bias".into(),
            vec![d],
        ),
        ("scale_shift_table".into(), vec![1, 2, d]),
        ("proj_out.weight".into(), vec![cfg.patch_features_out(), d]),
        ("proj_out.bias".into(), vec![cfg.patch_features_out()]),
    ];
    for l in 0..cfg.layers {
        let p = format!("blocks.{l}");
        for a in ["attn1", "attn2"] {
            for m in ["to_q", "to_k", "to_v", "to_out.0"] {
                v.push((format!("{p}.{a}.{m}.weight"), vec![d, d]));
                v.push((format!("{p}.{a}.{m}.bias"), vec![d]));
            }
            v.push((format!("{p}.{a}.norm_q.weight"), vec![d]));
            v.push((format!("{p}.{a}.norm_k.weight"), vec![d]));
        }
        v.push((format!("{p}.norm2.weight"), vec![d]));
        v.push((format!("{p}.norm2.bias"), vec![d]));
        v.push((format!("{p}.ffn.net.0.proj.weight"), vec![cfg.ffn_dim, d]));
        v.push((format!("{p}.ffn.net.0.proj.bias"), vec![cfg.ffn_dim]));
        v.push((format!("{p}.ffn.net.2.weight"), vec![d, cfg.ffn_dim]));
        v.push((format!("{p}.ffn.net.2.bias"), vec![d]));
        v.push((format!("{p}.scale_shift_table"), vec![1, 6, d]));
    }
    v
}

/// Every tensor the TAEHV decoder reads (positional `decoder.N` keys).
pub fn tae_manifest(cfg: &TaeConfig) -> Vec<(String, Vec<usize>)> {
    let nf = cfg.widths;
    let mut v = vec![
        (
            "decoder.1.weight".to_string(),
            vec![nf[0], cfg.latent_channels, 3, 3],
        ),
        ("decoder.1.bias".to_string(), vec![nf[0]]),
    ];
    for s in 0..3 {
        let base = 3 + 6 * s;
        for m in 0..cfg.mem_blocks {
            let p = format!("decoder.{}", base + m);
            v.push((format!("{p}.conv.0.weight"), vec![nf[s], 2 * nf[s], 3, 3]));
            v.push((format!("{p}.conv.0.bias"), vec![nf[s]]));
            v.push((format!("{p}.conv.2.weight"), vec![nf[s], nf[s], 3, 3]));
            v.push((format!("{p}.conv.2.bias"), vec![nf[s]]));
            v.push((format!("{p}.conv.4.weight"), vec![nf[s], nf[s], 3, 3]));
            v.push((format!("{p}.conv.4.bias"), vec![nf[s]]));
        }
        v.push((
            format!("decoder.{}.conv.weight", base + 4),
            vec![nf[s] * cfg.tgrow[s], nf[s], 1, 1],
        ));
        v.push((
            format!("decoder.{}.weight", base + 5),
            vec![nf[s + 1], nf[s], 3, 3],
        ));
    }
    v.push((
        "decoder.22.weight".into(),
        vec![cfg.out_channels(), nf[3], 3, 3],
    ));
    v.push(("decoder.22.bias".into(), vec![cfg.out_channels()]));
    v
}

/// Names in `manifest` that `src` lacks.
pub fn missing(src: &dyn WeightSource, manifest: &[(String, Vec<usize>)]) -> Vec<String> {
    manifest
        .iter()
        .filter(|(n, _)| {
            !(src.contains(n)
                || src.contains(&format!("{n}.q8"))
                || src.contains(&format!("{n}.q4"))
                || src.contains(&format!("{n}.fp8")))
        })
        .map(|(n, _)| n.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_matches_param_count() {
        let cfg = WanConfig::wan_1_3b();
        let n: usize = wan_manifest(&cfg)
            .iter()
            .map(|(_, s)| s.iter().product::<usize>())
            .sum();
        assert_eq!(n, cfg.params());
    }

    #[test]
    fn synthetic_is_deterministic() {
        let s = Synthetic { seed: 1 };
        assert_eq!(
            s.get("a.weight", &[4, 4]).unwrap(),
            s.get("a.weight", &[4, 4]).unwrap()
        );
        assert_ne!(
            s.get("a.weight", &[4, 4]).unwrap(),
            s.get("b.weight", &[4, 4]).unwrap()
        );
    }

    #[test]
    fn safetensors_roundtrip_bf16() {
        let dir = std::env::temp_dir().join(format!("lx2v-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vals = [1.0f32, -2.5, 0.125, 3.0];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2, 2], &bytes)
                .unwrap();
        safetensors::serialize_to_file([("w", view)], None, &dir.join("m.safetensors")).unwrap();
        let src = SafetensorsDir::open(&dir).unwrap();
        assert_eq!(src.get("w", &[2, 2]).unwrap(), vals);
        assert!(src.get("w", &[4]).is_err());
        assert!(src.get("nope", &[1]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
