//! A prompt gallery: UMT5 embeddings computed once (see
//! `scripts/encode_prompt.py`) so the engine can run without the text
//! encoder (`lightx2v serve --model-dir <dir> --gallery <dir>`).
//!
//! A directory with `gallery.json` and one safetensors file per prompt:
//!
//! ```text
//! gallery.json        {"prompts": [{"prompt": "...", "tokens": 33, "file": "01-a-red-fox-….safetensors"}]}
//! 01-….safetensors    prompt_embeds [tokens, 4096], F16 (F32 if a value does not fit)
//! ```
//!
//! Only the prompt's real tokens are stored: the model pads the rest of its
//! 512 positions with zeros, as Diffusers does. The format mirrors
//! fastwan-rs's gallery (same file layout, same tensor name) so an
//! `encode_prompt.py` written for either produces a gallery the other can
//! read — only `text_dim` differs (4096 here, same as fastwan's Wan 1.3B).

use lightx2v_core::{
    config::WanConfig,
    weights::{SafetensorsDir, WeightSource},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Serialize, Deserialize)]
pub struct Entry {
    pub prompt: String,
    pub tokens: usize,
    pub file: String,
}

#[derive(Default, Serialize, Deserialize)]
struct Index {
    prompts: Vec<Entry>,
}

pub struct Gallery {
    dir: PathBuf,
    pub entries: Vec<Entry>,
}

/// Whitespace-insensitive key: the page may reflow a prompt.
fn norm(p: &str) -> String {
    p.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Slug for a gallery file name. Only used by [`add`], which no CLI command
/// calls yet (encoding needs UMT5, Phase 3).
#[allow(dead_code)]
fn slug(p: &str) -> String {
    let mut s = String::new();
    for w in p
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        if s.len() + w.len() > 40 {
            break;
        }
        if !s.is_empty() {
            s.push('-');
        }
        s.push_str(&w.to_ascii_lowercase());
    }
    if s.is_empty() {
        "prompt".into()
    } else {
        s
    }
}

impl Gallery {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let path = dir.join("gallery.json");
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let idx: Index =
            serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;
        if idx.prompts.is_empty() {
            return Err(format!("{} lists no prompts", path.display()));
        }
        for e in &idx.prompts {
            if !dir.join(&e.file).is_file() {
                return Err(format!("{}: missing {}", path.display(), e.file));
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            entries: idx.prompts,
        })
    }

    pub fn find(&self, prompt: &str) -> Option<&Entry> {
        let k = norm(prompt);
        self.entries.iter().find(|e| norm(&e.prompt) == k)
    }

    /// `[text_len, text_dim]`, zero past the prompt's tokens.
    pub fn embeds(&self, e: &Entry, cfg: &WanConfig) -> Result<Vec<f32>, String> {
        let src = SafetensorsDir::open(&self.dir.join(&e.file)).map_err(|e| e.to_string())?;
        let rows = src
            .get("prompt_embeds", &[e.tokens, cfg.text_dim])
            .map_err(|e| e.to_string())?;
        let mut out = vec![0.0f32; cfg.text_len * cfg.text_dim];
        out[..rows.len()].copy_from_slice(&rows);
        Ok(out)
    }
}

/// Store one prompt's embeddings (the first `tokens` rows of `embeds`) and
/// list it in `gallery.json`, replacing an entry with the same prompt.
/// Used by `scripts/encode_prompt.py`'s documented format; no CLI command
/// calls it yet (encoding needs UMT5, Phase 3).
#[allow(dead_code)]
pub fn add(
    dir: &Path,
    prompt: &str,
    tokens: usize,
    embeds: &[f32],
    dim: usize,
) -> Result<Entry, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = dir.join("gallery.json");
    let mut idx: Index = match std::fs::read(&path) {
        Ok(b) => {
            serde_json::from_slice(&b).map_err(|e| format!("parse {}: {e}", path.display()))?
        }
        Err(_) => Index::default(),
    };
    let rows = &embeds[..tokens * dim];
    let (dt, bytes): (&str, Vec<u8>) = if rows.iter().all(|v| v.is_finite() && v.abs() <= 65504.0) {
        (
            "F16",
            rows.iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    } else {
        (
            "F32",
            rows.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    };
    let pos = idx
        .prompts
        .iter()
        .position(|e| norm(&e.prompt) == norm(prompt));
    let n = pos.map_or(idx.prompts.len() + 1, |i| i + 1);
    let file = format!("{n:02}-{}.safetensors", slug(prompt));
    let header = serde_json::json!({
        "prompt_embeds": {"dtype": dt, "shape": [tokens, dim], "data_offsets": [0, bytes.len()]},
        "__metadata__": {"prompt": prompt, "tokens": tokens.to_string()},
    });
    let header_bytes = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&bytes);
    std::fs::write(dir.join(&file), out).map_err(|e| e.to_string())?;
    let entry = Entry {
        prompt: prompt.to_string(),
        tokens,
        file,
    };
    match pos {
        Some(i) => idx.prompts[i] = entry.clone(),
        None => idx.prompts.push(entry.clone()),
    }
    let index_bytes = serde_json::to_vec_pretty(&idx).map_err(|e| e.to_string())?;
    std::fs::write(&path, index_bytes).map_err(|e| e.to_string())?;
    Ok(entry)
}
