//! Quantized checkpoints: the DiT with its block linears stored as int8 or
//! 4-bit codes, the format the browser downloads.
//!
//! One safetensors file. A quantized linear `X` (`X.weight` in Diffusers)
//! becomes, by [`WeightQuant::for_key`]:
//!
//! | tensor | int8 | 4-bit |
//! |---|---|---|
//! | `X.weight.q8` / `X.weight.q4` | I8 `[out, in]` | U8 `[out, in/2]`, element `2i` in the low nibble |
//! | `X.weight.scale` | F32 `[out, in/group]` | F16 `[out, in/group]` |
//! | `X.weight.zero` | — | F16 `[out, in/group]` |
//!
//! with `w ≈ q·scale` (int8) or `w ≈ (q − zero)·scale` (4-bit). Every other
//! tensor keeps its name and is stored as F16 when its values fit (F32 if
//! not, BF16 / F16 sources copied). Metadata `lightx2v.format = "quant/1"`,
//! `lightx2v.weights = "<format>"`. The loader reads the codes as they are
//! ([`crate::weights::WeightSource::stored`]), so a stored file and
//! `--weights <format>` on the original give the same block weights.
//!
//! This mirrors fastwan-rs's `export.rs` (same file layout, same tensor
//! names, same block ordering) — only the metadata keys carry the
//! `lightx2v.` prefix. A file written by `lightx2v quantize` and one
//! written by `fastwan quantize` differ only in that metadata and in the
//! DMD schedule the weights were distilled for; the loader accepts either.

use crate::{
    config::WanConfig,
    quant::{WeightQuant, Q4, Q8},
    weights::{wan_manifest, SafetensorsDir, WeightSource},
    Error, Result,
};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::Path;

/// One tensor to write: name, dtype, shape, little-endian bytes.
pub type Tensor = (String, Dtype, Vec<usize>, Vec<u8>);

/// The DiT's tensors (the manifest plus any `to_gate_compress`) with the
/// block linears quantized as `quant` says.
pub fn quantize_dit(
    src: &SafetensorsDir,
    cfg: &WanConfig,
    quant: WeightQuant,
) -> Result<Vec<Tensor>> {
    let mut names = wan_manifest(cfg);
    for l in 0..cfg.layers {
        let k = format!("blocks.{l}.to_gate_compress");
        if src.has_linear(&k) {
            names.push((format!("{k}.weight"), vec![cfg.dim(), cfg.dim()]));
            names.push((format!("{k}.bias"), vec![cfg.dim()]));
        }
    }
    let mut out = Vec::new();
    for (name, shape) in names {
        let key = name.strip_suffix(".weight");
        let fmt = key.map_or(WeightQuant::Native, |k| quant.for_key(k));
        if fmt == WeightQuant::Native {
            out.push(plain(src, &name, &shape)?);
            continue;
        }
        let (rows, cols) = (shape[0], shape[1..].iter().product::<usize>());
        let w = src.get(&name, &shape)?;
        match fmt {
            WeightQuant::Int8 => {
                let q = Q8::quantize(&w, rows, cols);
                let ng = cols / q.group;
                out.push((
                    format!("{name}.q8"),
                    Dtype::I8,
                    vec![rows, cols],
                    q.q.iter().map(|&v| v as u8).collect(),
                ));
                out.push((
                    format!("{name}.scale"),
                    Dtype::F32,
                    vec![rows, ng],
                    f32_bytes(&q.scales),
                ));
            }
            WeightQuant::Int4 { group, hqq, .. } => {
                if cols % 2 != 0 {
                    return Err(Error::Weights(format!(
                        "`{name}`: 4-bit needs an even row length, got {cols}"
                    )));
                }
                let q = Q4::quantize(&w, rows, cols, group, hqq);
                let ng = cols / q.group;
                out.push((
                    format!("{name}.q4"),
                    Dtype::U8,
                    vec![rows, cols / 2],
                    q.packed(),
                ));
                out.push((
                    format!("{name}.scale"),
                    Dtype::F16,
                    vec![rows, ng],
                    f16_bytes(&q.scales),
                ));
                out.push((
                    format!("{name}.zero"),
                    Dtype::F16,
                    vec![rows, ng],
                    f16_bytes(&q.zeros),
                ));
            }
            WeightQuant::Fp8 => {
                let q = crate::quant::Fp8E4M3::quantize(&w, rows, cols);
                let ng = cols / q.group;
                out.push((
                    format!("{name}.fp8"),
                    Dtype::U8,
                    vec![rows, cols],
                    q.q.clone(),
                ));
                out.push((
                    format!("{name}.scale"),
                    Dtype::F32,
                    vec![rows, ng],
                    f32_bytes(&q.scales),
                ));
            }
            WeightQuant::Native => unreachable!(),
        }
    }
    // Head first, then block by block (stable: manifest order within each).
    out.sort_by_key(|t| block_of(&t.0).map_or(0, |b| b + 1));
    Ok(out)
}

/// A tensor kept as is: F16 when an F32 source fits, else unchanged.
fn plain(src: &SafetensorsDir, name: &str, shape: &[usize]) -> Result<Tensor> {
    let (dt, stored_shape, data) = src.raw(name)?;
    match dt {
        Dtype::F32 => {
            let v = src.get(name, shape)?;
            if v.iter().all(|x| x.is_finite() && x.abs() <= 65504.0) {
                Ok((name.into(), Dtype::F16, stored_shape, f16_bytes(&v)))
            } else {
                Ok((name.into(), Dtype::F32, stored_shape, data.to_vec()))
            }
        }
        Dtype::F16 | Dtype::BF16 => Ok((name.into(), dt, stored_shape, data.to_vec())),
        other => Err(Error::Weights(format!(
            "`{name}`: unsupported dtype {other:?}"
        ))),
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

/// Write `tensors` as one safetensors file with `metadata`, **in the given
/// order** (the `safetensors` crate's serializer sorts by dtype and name).
/// [`quantize_dit`] orders the DiT head first and then block by block, so a
/// browser fetches each block as one HTTP range.
///
/// Streams the data to disk: a real f16 checkpoint is ~3 GB, and holding
/// a second whole-file copy in memory (as [`to_bytes`] does) OOMs a
/// small VM — the f16 export died that way on an 8 GB machine.
pub fn write(path: &Path, tensors: &[Tensor], metadata: HashMap<String, String>) -> Result<()> {
    use std::io::Write;
    let header = header_bytes(tensors, &metadata)?;
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    w.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(|e| Error::Weights(format!("cannot write {path:?}: {e}")))?;
    w.write_all(&header)
        .map_err(|e| Error::Weights(format!("cannot write {path:?}: {e}")))?;
    for t in tensors {
        w.write_all(&t.3)
            .map_err(|e| Error::Weights(format!("cannot write {path:?}: {e}")))?;
    }
    w.flush()
        .map_err(|e| Error::Weights(format!("cannot write {path:?}: {e}")))?;
    Ok(())
}

/// The bytes of [`write`] (whole file in memory; tests and small files
/// only — real checkpoints use [`write`]).
pub fn to_bytes(tensors: &[Tensor], metadata: &HashMap<String, String>) -> Result<Vec<u8>> {
    let header = header_bytes(tensors, metadata)?;
    let total: usize = tensors.iter().map(|t| t.3.len()).sum();
    let mut out = Vec::with_capacity(8 + header.len() + total);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    for t in tensors {
        out.extend_from_slice(&t.3);
    }
    Ok(out)
}

/// The padded safetensors header for `tensors` (validates every
/// tensor's byte length against its dtype × shape).
fn header_bytes(tensors: &[Tensor], metadata: &HashMap<String, String>) -> Result<Vec<u8>> {
    fn esc(s: &str) -> String {
        let mut o = String::with_capacity(s.len() + 2);
        o.push('"');
        for c in s.chars() {
            match c {
                '"' => o.push_str("\\\""),
                '\\' => o.push_str("\\\\"),
                c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                c => o.push(c),
            }
        }
        o.push('"');
        o
    }
    let mut parts = Vec::new();
    if !metadata.is_empty() {
        let mut kv: Vec<_> = metadata.iter().collect();
        kv.sort();
        let m: Vec<String> = kv
            .iter()
            .map(|(k, v)| format!("{}:{}", esc(k), esc(v)))
            .collect();
        parts.push(format!("\"__metadata__\":{{{}}}", m.join(",")));
    }
    let mut off = 0usize;
    for (name, dt, shape, data) in tensors {
        let n: usize = shape.iter().product();
        if n * dt.bitsize() / 8 != data.len() {
            return Err(Error::Weights(format!(
                "`{name}`: {} bytes for {shape:?} {dt:?}",
                data.len()
            )));
        }
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        parts.push(format!(
            "{}:{{\"dtype\":\"{dt:?}\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
            esc(name),
            dims.join(","),
            off + data.len()
        ));
        off += data.len();
    }
    let mut header = format!("{{{}}}", parts.join(",")).into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    Ok(header)
}

/// `blocks.N.…` → N; everything else (the head) sorts first.
pub fn block_of(name: &str) -> Option<usize> {
    name.strip_prefix("blocks.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// The metadata a quantized checkpoint carries.
pub fn metadata(quant: WeightQuant) -> HashMap<String, String> {
    HashMap::from([
        ("lightx2v.format".to_string(), "quant/1".to_string()),
        (
            "lightx2v.weights".to_string(),
            match quant {
                WeightQuant::Native => "f16".to_string(),
                q => q.to_string(),
            },
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Stored;
    use crate::weights::Synthetic;

    /// Synthetic tiny DiT → f32 safetensors → quantized file → `stored()`
    /// gives exactly the codes `--weights` would make from the source.
    #[test]
    fn stored_codes_equal_load_time_quantization() {
        let cfg = WanConfig::tiny();
        let syn = Synthetic { seed: 5 };
        let mut names = wan_manifest(&cfg);
        for l in 0..cfg.layers {
            names.push((
                format!("blocks.{l}.to_gate_compress.weight"),
                vec![cfg.dim(), cfg.dim()],
            ));
            names.push((format!("blocks.{l}.to_gate_compress.bias"), vec![cfg.dim()]));
        }
        let src_t: Vec<Tensor> = names
            .iter()
            .map(|(n, s)| {
                (
                    n.clone(),
                    Dtype::F32,
                    s.clone(),
                    f32_bytes(&syn.get(n, s).unwrap()),
                )
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("lightx2v-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f32_path = dir.join("src.safetensors");
        write(&f32_path, &src_t, HashMap::new()).unwrap();
        let src = SafetensorsDir::open(&f32_path).unwrap();
        for quant in [
            WeightQuant::Int8,
            WeightQuant::Int4 {
                group: 16,
                hqq: true,
                ffn_only: true,
            },
            WeightQuant::Int4 {
                group: 32,
                hqq: false,
                ffn_only: false,
            },
        ] {
            let path = dir.join(format!("{quant}.safetensors"));
            write(
                &path,
                &quantize_dit(&src, &cfg, quant).unwrap(),
                metadata(quant),
            )
            .unwrap();
            let q = SafetensorsDir::open(&path).unwrap();
            assert!(crate::weights::missing(&q, &wan_manifest(&cfg)).is_empty());
            for (n, s) in &names {
                let Some(key) = n.strip_suffix(".weight") else {
                    continue;
                };
                let (rows, cols) = (s[0], s[1..].iter().product::<usize>());
                let w = src.get(n, s).unwrap();
                match (quant.for_key(key), q.stored(key).unwrap()) {
                    (WeightQuant::Native, None) => {}
                    (WeightQuant::Int8, Some(Stored::Q8(got))) => {
                        let want = Q8::quantize(&w, rows, cols);
                        assert_eq!(
                            (got.q, got.scales, got.group),
                            (want.q, want.scales, want.group),
                            "{n}"
                        );
                    }
                    (WeightQuant::Int4 { group, hqq, .. }, Some(Stored::Q4(got))) => {
                        let want = Q4::quantize(&w, rows, cols, group, hqq);
                        assert_eq!(
                            (got.q, got.scales, got.zeros, got.group),
                            (want.q, want.scales, want.zeros, want.group),
                            "{n}"
                        );
                    }
                    (f, s) => panic!("{n}: format {f} stored as {s:?}"),
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The exported file is block-ordered: the head's tensors come first,
    /// then `blocks.0`, `blocks.1`, … with no interleaving, so a browser can
    /// fetch each block as one HTTP range.
    #[test]
    fn export_is_block_ordered() {
        let cfg = WanConfig::tiny();
        let syn = Synthetic { seed: 5 };
        let names = wan_manifest(&cfg);
        let src_t: Vec<Tensor> = names
            .iter()
            .map(|(n, s)| {
                (
                    n.clone(),
                    Dtype::F32,
                    s.clone(),
                    f32_bytes(&syn.get(n, s).unwrap()),
                )
            })
            .collect();
        let tensors = quantize_dit(
            &SafetensorsDir::from_bytes(to_bytes(&src_t, &HashMap::new()).unwrap()).unwrap(),
            &cfg,
            WeightQuant::Native,
        )
        .unwrap();
        let mut seen_block = 0usize;
        let mut seen_head_after_block = false;
        for (name, _, _, _) in &tensors {
            match block_of(name) {
                Some(b) => {
                    assert!(b >= seen_block, "{name} after block {seen_block}");
                    seen_block = b;
                }
                None => {
                    if seen_block > 0 {
                        seen_head_after_block = true;
                    }
                }
            }
        }
        assert!(!seen_head_after_block, "head tensors after block tensors");
    }
}
