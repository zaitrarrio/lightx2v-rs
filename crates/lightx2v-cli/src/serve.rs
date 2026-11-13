//! `lightx2v serve`: a local engine plus the Prompt Studio page.
//!
//! One engine thread owns the backend, the resident DiT and TAEHV, and the
//! prompt embeddings; it takes jobs one at a time from a queue. The HTTP
//! thread only reads the shared job table, so polling never waits on the
//! engine. Structure mirrors fastwan-rs's `serve.rs` (same endpoints, same
//! binary frame format, same guards), reimplemented against the lightx2v
//! pipeline — the differences are the API surface and the long-video chunk
//! progress, which fastwan has no equivalent of.
//!
//! ```text
//! GET  /                         the studio page (web/studio.html)
//! GET  /api/health               ready / backend / model / synthetic / gallery / max
//! GET  /api/gallery              prompts the page offers (restricted when a gallery is loaded)
//! POST /api/generate             {prompt, width, height, frames, seed?} -> {id, seed}
//! GET  /api/jobs/<id>            stage, per-step/chunk timings, frames decoded so far,
//!                                chunk progress (chunks_done / total_chunks)
//! GET  /api/jobs/<id>/frames?from=N
//!                                binary: u32 LE width, height, first, count; then RGB8
//! ```
//!
//! It binds 127.0.0.1 by default and sends no CORS headers, so other web
//! pages cannot queue jobs on your GPU. `--allow-origin <url>` opens the API
//! to exactly one origin (the hosted studio page) and to nothing else.
//!
//! Synthetic mode runs the tiny random model (64x64, 9 frames, 1 chunk):
//! any prompt is accepted and its embeddings are seeded randoms from the
//! prompt hash. Real mode needs `--model-dir` plus `--gallery` (precomputed
//! UMT5 embeddings — there is no text encoder yet, Phase 3); without a
//! gallery the engine fails to start and names the missing encoder.

use crate::gallery::Gallery;
use lightx2v_core::{
    backend::Backend,
    config::{DistillConfig, GenerationConfig, MemRopeConfig, TaeConfig, WanConfig},
    geometry::Geometry,
    rng::Rng,
    weights::{missing, tae_manifest, wan_manifest, SafetensorsDir, Synthetic},
};
use lightx2v_model::{
    pipeline::{CausalPipeline, Event, Options, Timings},
    LightDit, TaeHv,
};
use serde_json::{json, Value};
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

const PAGE: &str = include_str!("../../../web/studio.html");
const GALLERY_PROMPTS: &str = include_str!("../../../web/gallery-prompts.txt");
const KEEP_JOBS: usize = 8;
const MAX_QUEUE: usize = 3;
const MAX_PIXELS: usize = 832 * 480;
const MAX_FRAMES: usize = 1005;

pub enum Weights {
    /// Tiny random model: exercises the engine and the page without a download.
    /// With a gallery, only its prompts are accepted.
    Synthetic { gallery: Option<PathBuf> },
    /// Real Wan2.1-T2V-1.3B DMD checkpoint. A gallery is mandatory: without
    /// UMT5 there is no other way to encode a prompt.
    Real {
        model_dir: PathBuf,
        gallery: Option<PathBuf>,
    },
}

pub struct Settings {
    pub addr: String,
    pub weights: Weights,
    pub allow_origin: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Queued,
    Encoding,
    Preparing,
    Denoising,
    Decoding,
    Done,
    Failed,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Queued => "queued",
            Stage::Encoding => "encoding",
            Stage::Preparing => "preparing",
            Stage::Denoising => "denoising",
            Stage::Decoding => "decoding",
            Stage::Done => "done",
            Stage::Failed => "failed",
        }
    }
}

struct Job {
    id: u64,
    prompt: String,
    seed: u64,
    /// Requested video frames (81 or 1005); the decoded count is `frames`.
    req_frames: usize,
    stage: Stage,
    tokens: usize,
    encode_s: f64,
    prepare_s: f64,
    steps: Vec<f64>,
    total_steps: usize,
    decode_s: f64,
    started: Option<Instant>,
    total_s: f64,
    width: usize,
    height: usize,
    total_frames: usize,
    frames: usize,
    chunks_done: usize,
    total_chunks: usize,
    rgb: Vec<u8>,
    error: Option<String>,
}

impl Job {
    fn status(&self) -> Value {
        json!({
            "id": self.id, "prompt": self.prompt, "stage": self.stage.name(), "seed": self.seed,
            "width": self.width, "height": self.height, "total_frames": self.total_frames,
            "frames": self.frames, "chunks_done": self.chunks_done, "total_chunks": self.total_chunks,
            "tokens": self.tokens, "encode_s": self.encode_s, "prepare_s": self.prepare_s,
            "steps": self.steps, "total_steps": self.total_steps, "decode_s": self.decode_s,
            "elapsed_s": self.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0),
            "total_s": self.total_s, "error": self.error,
        })
    }
}

pub struct Shared {
    jobs: Mutex<Vec<Job>>,
    info: Mutex<Value>,
    next: Mutex<u64>,
}

impl Shared {
    fn with<R>(&self, id: u64, f: impl FnOnce(&mut Job) -> R) -> Option<R> {
        self.jobs
            .lock()
            .unwrap()
            .iter_mut()
            .find(|j| j.id == id)
            .map(f)
    }
}

/// What the engine needs from the prompt side.
enum Text {
    Synthetic {
        len: usize,
        dim: usize,
        only: Option<Gallery>,
    },
    Gallery(Gallery, WanConfig),
}

impl Text {
    fn encode(&self, prompt: &str) -> Result<(Vec<f32>, usize), String> {
        match self {
            Text::Synthetic { len, dim, only } => {
                if let Some(g) = only {
                    if g.find(prompt).is_none() {
                        return Err(format!(
                            "This engine has no text encoder: pick one of its {} gallery prompts.",
                            g.entries.len()
                        ));
                    }
                }
                // FNV-1a over the prompt bytes: same prompt, same embeddings.
                let seed = prompt.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
                });
                Ok((
                    Rng::new(seed).normal(len * dim),
                    prompt.split_whitespace().count(),
                ))
            }
            Text::Gallery(g, cfg) => {
                let e = g.find(prompt).ok_or_else(|| {
                    format!(
                        "This engine has no text encoder: pick one of its {} gallery prompts.",
                        g.entries.len()
                    )
                })?;
                Ok((g.embeds(e, cfg)?, e.tokens))
            }
        }
    }
}

/// The page's built-in examples: `web/gallery-prompts.txt`, minus comments.
fn bundled_prompts() -> Vec<String> {
    GALLERY_PROMPTS
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// `[frames, 3, H, W]` f32 in `[0, 1]` → interleaved RGB8.
fn f32_to_rgb8(frames: &lightx2v_model::taehv::Frames) -> Vec<u8> {
    let (n, h, w) = (frames.frames, frames.height, frames.width);
    let mut out = vec![0u8; n * h * w * 3];
    for fi in 0..n {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = frames.data[((fi * 3 + c) * h + y) * w + x];
                    out[(fi * h + y) * w * 3 + x * 3 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
    }
    out
}

/// Decode a denoised latent progressively, 4 latent frames at a time,
/// appending each chunk's frames to the job as they land (the TAEHV
/// `DecodeState` carries the MemBlock memory across chunk boundaries).
struct Decode<'a, B: Backend> {
    be: &'a B,
    pipe: &'a CausalPipeline<B>,
    latent: &'a [f32],
    latent_frames: usize,
    height: usize,
    width: usize,
    id: u64,
    shared: &'a Shared,
}

fn decode_progressive<B: Backend>(d: Decode<'_, B>) -> Result<f64, String> {
    let t0 = Instant::now();
    let cfg = &d.pipe.tae.cfg;
    let (lh, lw) = (
        d.height / cfg.spatial_scale(),
        d.width / cfg.spatial_scale(),
    );
    let mut st = d.pipe.tae.state();
    let mut t = 0;
    while t < d.latent_frames {
        let f = 4usize.min(d.latent_frames - t);
        // Transpose this chunk [C, f, H, W] -> frame-major [f, C, H, W].
        let mut z = vec![0.0f32; f * cfg.latent_channels * lh * lw];
        for fi in 0..f {
            for c in 0..cfg.latent_channels {
                for y in 0..lh {
                    for x in 0..lw {
                        z[((fi * cfg.latent_channels + c) * lh + y) * lw + x] =
                            d.latent[((c * d.latent_frames + t + fi) * lh + y) * lw + x];
                    }
                }
            }
        }
        let chunk = d.pipe.tae.decode_chunk(d.be, &mut st, &z, f, lh, lw);
        let rgb = f32_to_rgb8(&chunk);
        let n = chunk.frames;
        d.shared.with(d.id, |j| {
            j.rgb.extend_from_slice(&rgb);
            j.frames += n;
            j.decode_s = t0.elapsed().as_secs_f64();
        });
        t += f;
    }
    Ok(t0.elapsed().as_secs_f64())
}

/// Loads everything, then runs jobs forever. Runs on the engine thread.
pub fn engine<B: Backend>(
    be: B,
    s: &Settings,
    shared: Arc<Shared>,
    rx: mpsc::Receiver<u64>,
) -> Result<(), String> {
    let t0 = Instant::now();
    let stage = |what: &str| {
        eprintln!("engine: {what} ({:.1}s)", t0.elapsed().as_secs_f64());
        let mut info = shared.info.lock().unwrap();
        info["loading"] = json!(what);
        info["loading_s"] = json!(t0.elapsed().as_secs_f64());
    };
    stage(&format!("starting on {}", be.name()));
    let (pipe, text, model, synthetic) = match &s.weights {
        Weights::Synthetic { gallery } => {
            let (cfg, tcfg) = (WanConfig::tiny(), TaeConfig::tiny());
            let src = Synthetic { seed: 1 };
            let only = match gallery {
                Some(dir) => {
                    let g = Gallery::open(dir)?;
                    shared.info.lock().unwrap()["gallery"] =
                        json!(g.entries.iter().map(|e| &e.prompt).collect::<Vec<_>>());
                    Some(g)
                }
                None => None,
            };
            let text = Text::Synthetic {
                len: cfg.text_len,
                dim: cfg.text_dim,
                only,
            };
            stage("loading the tiny synthetic model");
            let pipe = CausalPipeline::new(
                LightDit::load(&be, cfg, MemRopeConfig::default_memrope(), &src)
                    .map_err(|e| e.to_string())?,
                TaeHv::load(&be, tcfg, &src).map_err(|e| e.to_string())?,
                DistillConfig::dmd2(),
            );
            (
                pipe,
                text,
                "synthetic tiny model (random weights: output is noise) · 64x64 · 9 frames"
                    .to_string(),
                true,
            )
        }
        Weights::Real { model_dir, gallery } => {
            let cfg = WanConfig::wan_1_3b();
            let tcfg = TaeConfig::taew2_1();
            stage("checking the checkpoint");
            let src = SafetensorsDir::open(model_dir).map_err(|e| e.to_string())?;
            let miss = missing(&src, &wan_manifest(&cfg));
            if !miss.is_empty() {
                return Err(format!(
                    "checkpoint in {} is missing {} DiT tensor(s), first: {}",
                    model_dir.display(),
                    miss.len(),
                    miss[..miss.len().min(3)].join(", ")
                ));
            }
            let miss = missing(&src, &tae_manifest(&tcfg));
            if !miss.is_empty() {
                return Err(format!(
                    "checkpoint in {} is missing {} TAEHV tensor(s), first: {}",
                    model_dir.display(),
                    miss.len(),
                    miss[..miss.len().min(3)].join(", ")
                ));
            }
            let gallery = match gallery {
                Some(dir) => {
                    stage("reading the prompt gallery");
                    let g = Gallery::open(dir)?;
                    shared.info.lock().unwrap()["gallery"] =
                        json!(g.entries.iter().map(|e| &e.prompt).collect::<Vec<_>>());
                    g
                }
                None => {
                    return Err(
                        "real-checkpoint generation is Phase 3 work: the DiT manifest \
                         validated, but the text encoder (UMT5) is not ported yet, so \
                         there are no prompt embeddings to condition on — serve with \
                         --gallery <dir> of precomputed prompt embeddings, or --synthetic"
                            .to_string(),
                    );
                }
            };
            let text = Text::Gallery(gallery, cfg.clone());
            stage("loading the transformer (1.3B parameters)");
            let pipe = CausalPipeline::new(
                LightDit::load(&be, cfg, MemRopeConfig::default_memrope(), &src)
                    .map_err(|e| e.to_string())?,
                TaeHv::load(&be, tcfg, &src).map_err(|e| e.to_string())?,
                DistillConfig::dmd2(),
            );
            (
                pipe,
                text,
                "Wan2.1-T2V-1.3B DMD · 4-step · TAEHV · prompt gallery".to_string(),
                false,
            )
        }
    };
    stage("waiting for the backend");
    be.sync();
    {
        let mut info = shared.info.lock().unwrap();
        info["ready"] = json!(true);
        info["model"] = json!(model);
        info["backend"] = json!(be.name());
        info["load_s"] = json!(t0.elapsed().as_secs_f64());
    }
    eprintln!(
        "engine ready on {} in {:.1}s",
        be.name(),
        t0.elapsed().as_secs_f64()
    );

    for id in rx {
        let Some((prompt, frames, seed)) = shared.with(id, |j| {
            j.started = Some(Instant::now());
            j.stage = Stage::Encoding;
            let frames = j.req_frames;
            // The output shape is fixed by the mode, not the request: the
            // synthetic model only ever runs its toy shape.
            if synthetic {
                j.width = 64;
                j.height = 64;
                j.total_frames = 9;
                j.total_chunks = 1;
            } else {
                j.width = 832;
                j.height = 480;
                j.total_frames = frames;
                j.total_chunks = Geometry::latent_frames(frames) / 3;
            }
            (j.prompt.clone(), frames, j.seed)
        }) else {
            continue;
        };
        let t = Instant::now();
        let run = || -> Result<(), String> {
            let (embeds, tokens) = text.encode(&prompt)?;
            shared.with(id, |j| {
                j.encode_s = t.elapsed().as_secs_f64();
                j.tokens = tokens;
                j.stage = Stage::Preparing;
            });
            let (gen, geometry) = if synthetic {
                (
                    GenerationConfig {
                        latent_frames: 3,
                        ..GenerationConfig::short_clip()
                    },
                    Geometry {
                        frames: 9,
                        height: 64,
                        width: 64,
                    },
                )
            } else if frames == 1005 {
                (GenerationConfig::demo(), Geometry::LONG_60S)
            } else {
                (GenerationConfig::short_clip(), Geometry::SHORT_480P)
            };
            let opt = Options {
                geometry,
                seed,
                ffn_chunk: usize::MAX,
            };
            let mut timings = Timings::default();
            let latent = pipe
                .denoise_with(&be, &embeds, &gen, &opt, &mut timings, &mut |ev| match ev {
                    Event::Prepared { secs } => {
                        shared.with(id, |j| {
                            j.prepare_s = secs;
                            j.total_steps = 4;
                            j.stage = Stage::Denoising;
                        });
                    }
                    Event::Step {
                        chunk,
                        index,
                        total,
                        secs,
                    } => {
                        shared.with(id, |j| {
                            j.steps.push(secs);
                            j.total_steps = total;
                            if index + 1 == total {
                                j.chunks_done = chunk + 1;
                            }
                        });
                    }
                    Event::Chunk { index, total, .. } => {
                        shared.with(id, |j| {
                            j.chunks_done = index + 1;
                            j.total_chunks = total;
                        });
                    }
                    Event::Latent { .. } => {
                        shared.with(id, |j| {
                            j.stage = Stage::Decoding;
                        });
                    }
                })
                .map_err(|e| e.to_string())?;
            decode_progressive(Decode {
                be: &be,
                pipe: &pipe,
                latent: &latent,
                latent_frames: gen.latent_frames,
                height: geometry.height,
                width: geometry.width,
                id,
                shared: &shared,
            })?;
            Ok(())
        };
        let res = run();
        shared.with(id, |j| {
            j.total_s = t.elapsed().as_secs_f64();
            match res {
                Ok(()) => j.stage = Stage::Done,
                Err(e) => {
                    j.stage = Stage::Failed;
                    j.error = Some(e);
                }
            }
        });
    }
    Ok(())
}

fn hdr(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k, v).expect("valid header")
}

/// Send with the CORS headers for an allowed origin, if this request has one.
fn send<R: std::io::Read>(
    req: tiny_http::Request,
    resp: tiny_http::Response<R>,
    cors: &[tiny_http::Header],
) {
    let mut resp = resp;
    for h in cors {
        resp = resp.with_header(h.clone());
    }
    let _ = req.respond(resp);
}

fn respond_json(req: tiny_http::Request, code: u16, v: &Value, cors: &[tiny_http::Header]) {
    let r = tiny_http::Response::from_string(v.to_string())
        .with_status_code(code)
        .with_header(hdr("Content-Type", "application/json"));
    send(req, r, cors);
}

/// CORS headers when the request's `Origin` is the one allowed origin.
fn cors_for(req: &tiny_http::Request, allow: Option<&str>) -> Vec<tiny_http::Header> {
    let origin = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("Origin"))
        .map(|h| h.value.as_str().to_string());
    match (origin, allow) {
        (Some(o), Some(a)) if o.trim_end_matches('/') == a.trim_end_matches('/') => vec![
            hdr("Access-Control-Allow-Origin", &o),
            hdr("Vary", "Origin"),
            hdr("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            hdr("Access-Control-Allow-Headers", "Content-Type"),
            // Chrome's Private Network Access: a public https page calling 127.0.0.1.
            hdr("Access-Control-Allow-Private-Network", "true"),
            hdr("Access-Control-Max-Age", "600"),
        ],
        _ => Vec::new(),
    }
}

fn parse_request(body: &str, synthetic: bool) -> Result<(String, usize, u64), String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("body is not JSON: {e}"))?;
    let prompt = v["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return Err("Write a prompt first.".into());
    }
    if prompt.chars().count() > 2000 {
        return Err("Prompts are limited to 2000 characters.".into());
    }
    let frames = v["frames"].as_u64().unwrap_or(81) as usize;
    if !matches!(frames, 81 | 1005) {
        return Err(format!(
            "unsupported frame count {frames}: only 81 (short) and 1005 (60 s demo) are supported"
        ));
    }
    if !synthetic {
        let width = v["width"].as_u64().unwrap_or(832) as usize;
        let height = v["height"].as_u64().unwrap_or(480) as usize;
        if (width, height) != (832, 480) {
            return Err(format!(
                "real weights run at 832x480 only; {width}x{height} is not supported yet"
            ));
        }
    }
    let seed = v["seed"].as_u64().unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64 % 1_000_000_000)
            .unwrap_or(7)
    });
    Ok((prompt, frames, seed))
}

/// Serve until killed. `start_engine` builds the backend on the engine thread.
pub fn serve(
    s: Settings,
    start_engine: impl FnOnce(&Settings, Arc<Shared>, mpsc::Receiver<u64>) -> Result<(), String>
        + Send
        + 'static,
) -> Result<(), String> {
    let shared = Arc::new(Shared {
        jobs: Mutex::new(Vec::new()),
        info: Mutex::new(json!({
            "ready": false, "error": null, "backend": null, "model": null,
            "synthetic": matches!(s.weights, Weights::Synthetic { .. }), "gallery": null,
            "max_pixels": MAX_PIXELS, "max_frames": MAX_FRAMES, "fps": 16,
        })),
        next: Mutex::new(1),
    });
    let (tx, rx) = mpsc::channel::<u64>();
    let server = tiny_http::Server::http(&s.addr).map_err(|e| format!("bind {}: {e}", s.addr))?;
    eprintln!("lightx2v studio: http://{}/", s.addr);
    let s = Arc::new(s);
    {
        let (shared, s) = (shared.clone(), s.clone());
        std::thread::Builder::new()
            .name("engine".into())
            .spawn(move || {
                // A panic (OOM, a failed assertion) must reach the page:
                // without this the thread died silently and the page waited
                // for ever.
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    start_engine(&s, shared.clone(), rx)
                }));
                let msg = match run {
                    Ok(Ok(())) => return,
                    Ok(Err(e)) => e,
                    Err(p) => format!(
                        "engine crashed: {}",
                        p.downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| p.downcast_ref::<&str>().copied())
                            .unwrap_or("panic")
                    ),
                };
                eprintln!("engine failed: {msg}");
                {
                    let mut info = shared.info.lock().unwrap();
                    info["ready"] = json!(false);
                    info["error"] = json!(msg.clone());
                }
                for j in shared.jobs.lock().unwrap().iter_mut() {
                    if !matches!(j.stage, Stage::Done | Stage::Failed) {
                        j.stage = Stage::Failed;
                        j.error = Some(msg.clone());
                    }
                }
            })
            .map_err(|e| format!("spawn engine: {e}"))?;
    }

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let method = req.method().clone();
        let cors = cors_for(&req, s.allow_origin.as_deref());
        match (method, path) {
            (tiny_http::Method::Options, _) => {
                let code = if cors.is_empty() { 403 } else { 204 };
                send(req, tiny_http::Response::empty(code), &cors);
            }
            (tiny_http::Method::Get, "/") | (tiny_http::Method::Get, "/index.html") => {
                let h = tiny_http::Header::from_bytes("Content-Type", "text/html; charset=utf-8")
                    .unwrap();
                send(
                    req,
                    tiny_http::Response::from_string(PAGE).with_header(h),
                    &cors,
                );
            }
            (tiny_http::Method::Get, "/api/health") | (tiny_http::Method::Get, "/api/info") => {
                let mut info = shared.info.lock().unwrap().clone();
                let jobs = shared.jobs.lock().unwrap();
                info["queued"] = json!(jobs.iter().filter(|j| j.stage == Stage::Queued).count());
                drop(jobs);
                respond_json(req, 200, &info, &cors);
            }
            (tiny_http::Method::Get, "/api/gallery") => {
                let info = shared.info.lock().unwrap().clone();
                let (prompts, restricted) = match info["gallery"].as_array() {
                    Some(g) => (
                        g.iter()
                            .filter_map(|p| p.as_str().map(str::to_string))
                            .collect(),
                        true,
                    ),
                    None => (bundled_prompts(), false),
                };
                respond_json(
                    req,
                    200,
                    &json!({"prompts": prompts, "restricted": restricted}),
                    &cors,
                );
            }
            (tiny_http::Method::Post, "/api/generate") => {
                // CSRF guard: a cross-site form or `text/plain` fetch skips the
                // CORS preflight, so check the origin and content type here.
                let header = |k: &str| {
                    req.headers()
                        .iter()
                        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(k))
                        .map(|h| h.value.as_str().to_string())
                };
                let json_body =
                    header("Content-Type").is_some_and(|c| c.starts_with("application/json"));
                let origin_ok = match header("Origin") {
                    None => true,
                    Some(o) => {
                        !cors.is_empty()
                            // Same origin, direct (http) or behind a TLS proxy / tunnel.
                            || header("Host").is_some_and(|h| {
                                o == format!("http://{h}") || o == format!("https://{h}")
                            })
                    }
                };
                if !json_body || !origin_ok {
                    respond_json(
                        req,
                        403,
                        &json!({"error": "Requests must be JSON and come from this engine's page or its --allow-origin site."}),
                        &cors,
                    );
                    continue;
                }
                let mut body = String::new();
                if req
                    .as_reader()
                    .take(64 * 1024)
                    .read_to_string(&mut body)
                    .is_err()
                {
                    respond_json(req, 400, &json!({"error": "unreadable body"}), &cors);
                    continue;
                }
                let info = shared.info.lock().unwrap().clone();
                if !info["ready"].as_bool().unwrap_or(false) {
                    let msg = info["error"]
                        .as_str()
                        .map(|e| format!("The engine failed to start: {e}"))
                        .unwrap_or_else(|| {
                            "The engine is still loading the model. Try again in a moment.".into()
                        });
                    respond_json(req, 503, &json!({"error": msg}), &cors);
                    continue;
                }
                let synthetic = info["synthetic"].as_bool().unwrap_or(false);
                // Gallery mode: refuse a prompt it cannot encode now, not
                // after it has waited in the queue.
                let parsed = parse_request(&body, synthetic).and_then(|r| {
                    let info = shared.info.lock().unwrap();
                    match info["gallery"].as_array() {
                        Some(g)
                            if !g.iter().any(|p| {
                                p.as_str().is_some_and(|p| {
                                    p.split_whitespace().eq(r.0.split_whitespace())
                                })
                            }) =>
                        {
                            Err(format!(
                                "This engine has no text encoder: pick one of its {} gallery prompts.",
                                g.len()
                            ))
                        }
                        _ => Ok(r),
                    }
                });
                match parsed {
                    Err(e) => respond_json(req, 400, &json!({"error": e}), &cors),
                    Ok((prompt, frames, seed)) => {
                        let mut jobs = shared.jobs.lock().unwrap();
                        if jobs
                            .iter()
                            .filter(|j| matches!(j.stage, Stage::Queued))
                            .count()
                            >= MAX_QUEUE
                        {
                            drop(jobs);
                            respond_json(
                                req,
                                429,
                                &json!({"error": "Three clips are already waiting. Try again when one finishes."}),
                                &cors,
                            );
                            continue;
                        }
                        let id = {
                            let mut n = shared.next.lock().unwrap();
                            *n += 1;
                            *n - 1
                        };
                        jobs.push(Job {
                            id,
                            prompt,
                            seed,
                            req_frames: frames,
                            stage: Stage::Queued,
                            tokens: 0,
                            encode_s: 0.0,
                            prepare_s: 0.0,
                            steps: Vec::new(),
                            total_steps: 4,
                            decode_s: 0.0,
                            started: None,
                            total_s: 0.0,
                            width: 0,
                            height: 0,
                            total_frames: 0,
                            frames: 0,
                            chunks_done: 0,
                            total_chunks: 1,
                            rgb: Vec::new(),
                            error: None,
                        });
                        // Drop the oldest finished jobs (their frames are the memory).
                        while jobs.len() > KEEP_JOBS {
                            match jobs
                                .iter()
                                .position(|j| matches!(j.stage, Stage::Done | Stage::Failed))
                            {
                                Some(i) => {
                                    jobs.remove(i);
                                }
                                None => break,
                            }
                        }
                        drop(jobs);
                        let _ = tx.send(id);
                        respond_json(req, 200, &json!({"id": id, "seed": seed}), &cors);
                    }
                }
            }
            (tiny_http::Method::Get, p) if p.starts_with("/pkg/") => {
                // The studio page loads its wasm engine from here. Only the
                // two built files are served (no path traversal), read from
                // disk: `web/pkg/` is a build artifact, not bundled.
                let name = &p["/pkg/".len()..];
                let (ctype, ok) = match name {
                    "lightx2v_web.js" => ("application/javascript", true),
                    "lightx2v_web_bg.wasm" => ("application/wasm", true),
                    _ => ("", false),
                };
                if !ok {
                    respond_json(req, 404, &json!({"error": "no such file"}), &cors);
                    continue;
                }
                let from_cwd = std::path::PathBuf::from("web/pkg").join(name);
                let from_manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../web/pkg")
                    .join(name);
                let bytes = std::fs::read(&from_cwd).or_else(|_| std::fs::read(&from_manifest));
                match bytes {
                    Ok(b) => {
                        let h = tiny_http::Header::from_bytes("Content-Type", ctype).unwrap();
                        send(req, tiny_http::Response::from_data(b).with_header(h), &cors);
                    }
                    Err(_) => respond_json(
                        req,
                        404,
                        &json!({"error": "wasm build not found: run scripts/build-web.sh first"}),
                        &cors,
                    ),
                }
            }
            (tiny_http::Method::Get, p) if p.starts_with("/api/jobs/") => {
                let rest = &p["/api/jobs/".len()..];
                let (id_s, frames) = match rest.strip_suffix("/frames") {
                    Some(i) => (i, true),
                    None => (rest, false),
                };
                let Ok(id) = id_s.parse::<u64>() else {
                    respond_json(req, 404, &json!({"error": "no such job"}), &cors);
                    continue;
                };
                if frames {
                    let from = query
                        .split('&')
                        .find_map(|kv| kv.strip_prefix("from="))
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    let body = shared.with(id, |j| {
                        let px = j.width * j.height * 3;
                        let first = from.min(j.frames);
                        let count = j.frames - first;
                        let mut out = Vec::with_capacity(16 + count * px);
                        for v in [j.width, j.height, first, count] {
                            out.extend_from_slice(&(v as u32).to_le_bytes());
                        }
                        out.extend_from_slice(&j.rgb[first * px..(first + count) * px]);
                        out
                    });
                    match body {
                        Some(b) => {
                            let h = tiny_http::Header::from_bytes(
                                "Content-Type",
                                "application/octet-stream",
                            )
                            .unwrap();
                            send(req, tiny_http::Response::from_data(b).with_header(h), &cors);
                        }
                        None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                    }
                } else {
                    match shared.with(id, |j| j.status()) {
                        Some(v) => respond_json(req, 200, &v, &cors),
                        None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                    }
                }
            }
            _ => respond_json(req, 404, &json!({"error": "not found"}), &cors),
        }
    }
    Ok(())
}
