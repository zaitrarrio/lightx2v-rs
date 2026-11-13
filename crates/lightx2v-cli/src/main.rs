//! `lightx2v`: plan / check / bench / generate for the Rust LightX2V +
//! History-Forcing port.
//!
//! `--backend cpu|wgpu|auto` (default `auto`) selects the backend for
//! `check`, `bench` and `generate`; `plan` is backend-independent and only
//! reports the selection. `auto` uses wgpu when an adapter answers, else
//! the CPU. When the requested backend is unavailable (no GPU adapter on
//! this machine, or a build without the `wgpu` feature) the command prints
//! a clean skip reason and exits 0 instead of failing.
//!
//! `generate --synthetic` runs the tiny random model end to end and writes
//! the denoised latent (`<out>.latent`, raw f32) plus the TAEHV-decoded
//! video (`<out>.rgb`, interleaved RGB8). Without `--synthetic` it needs a
//! real checkpoint directory; the UMT5 text encoder is still Phase 3, so
//! real generation is not available yet.
//!
//! `serve` runs the Prompt Studio web UI (see `serve.rs`).

mod gallery;
mod serve;

use clap::{Parser, Subcommand, ValueEnum};
use lightx2v_core::{
    backend::{Backend, OffloadGranularity},
    config::{DistillConfig, GenerationConfig, MemRopeConfig, TaeConfig, WanConfig},
    geometry::Geometry,
    plan::{plan, Device},
    quant::WeightQuant,
    weights::{missing, tae_manifest, wan_manifest, SafetensorsDir, Synthetic, WeightSource},
};
use lightx2v_cpu::CpuBackend;
use lightx2v_model::{
    check,
    pipeline::{CausalPipeline, Options, Timings},
    LightDit, TaeHv,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(feature = "wgpu")]
use lightx2v_wgpu::WgpuBackend;

#[derive(Parser)]
#[command(
    name = "lightx2v",
    about = "LightX2V + History-Forcing long-video inference, in Rust"
)]
struct Cli {
    /// Backend: cpu, wgpu, or auto (wgpu when an adapter answers, else cpu).
    #[arg(long, global = true, default_value = "auto")]
    backend: BackendOpt,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendOpt {
    Cpu,
    Wgpu,
    Auto,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the memory plan for a clip (weights, activations, MemRoPE
    /// cache, decode peak) and whether it fits the VRAM budget.
    Plan {
        /// VRAM budget in GiB.
        #[arg(long, default_value = "16")]
        vram_gib: f64,
        /// Video frames (81 = short clip, 1005 = the 60 s demo).
        #[arg(long, default_value = "81")]
        frames: usize,
        /// Offload granularity: none | model | block | disk.
        #[arg(long, default_value = "none")]
        offload: Offload,
        /// Weight bytes (2 = bf16/f16, 1 = int8, 4 = f32).
        #[arg(long, default_value = "2")]
        weight_bytes: u64,
    },
    /// Run the conformance checks (every op vs its f32 reference) on the
    /// selected backend.
    Check,
    /// Time the tiny synthetic DiT on the selected backend: one chunk, all
    /// 4 DMD steps.
    Bench {
        /// DiT layers in the synthetic model.
        #[arg(long, default_value = "2")]
        layers: usize,
    },
    /// Denoise a latent with the tiny synthetic model or a real checkpoint
    /// directory. Writes `<out>.latent` (raw f32) and the TAEHV-decoded
    /// `<out>.rgb` (interleaved RGB8).
    Generate {
        /// Use the tiny synthetic model (no download).
        #[arg(long)]
        synthetic: bool,
        /// Checkpoint directory with `*.safetensors` (DiT + TAEHV).
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Prompt (synthetic runs use a fixed zero embedding).
        #[arg(long, default_value = "a fox running through snow")]
        prompt: String,
        /// Video frames: 81 (short) or 1005 (60 s demo).
        #[arg(long, default_value = "81")]
        frames: usize,
        /// RNG seed.
        #[arg(long, default_value = "0")]
        seed: u64,
        /// Output path prefix.
        #[arg(long, default_value = "clip")]
        out: PathBuf,
        /// safetensors holding `prompt_embeds` `[L, 4096]`
        /// (scripts/encode_prompt.py). Without it, real generation needs
        /// the UMT5 text encoder (Phase 3) and errors.
        #[arg(long)]
        embeds: Option<PathBuf>,
        /// Load-time weight format for plain (non-quantized) tensors:
        /// native | int8 | int4[-hqq][-gN][-ffn] | fp8.
        #[arg(long, default_value = "native")]
        weights: WeightQuant,
    },
    /// Write a quantized copy of the DiT: block linears as int8 / 4-bit codes,
    /// everything else f16 (the browser download format; see
    /// `lightx2v_core::export`). Load it with --model-dir like the original.
    Quantize {
        /// The checkpoint's `transformer/` directory.
        #[arg(long)]
        transformer: PathBuf,
        /// native (all f16) | int8 | int4[-hqq][-gN][-ffn], e.g. int4-hqq-g64.
        #[arg(long)]
        weights: WeightQuant,
        #[arg(long)]
        out: PathBuf,
    },
    /// Serve the Prompt Studio web UI: a local engine (resident DiT/TAEHV)
    /// plus the prompt-to-clip page at http://<addr>/.
    Serve {
        /// Address to bind, e.g. 127.0.0.1:7860.
        #[arg(long, default_value = "127.0.0.1:7860")]
        addr: String,
        /// Use the tiny synthetic model (no download, any prompt accepted).
        #[arg(long)]
        synthetic: bool,
        /// Checkpoint directory with `*.safetensors` (DiT + TAEHV).
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Gallery directory of precomputed prompt embeddings
        /// (`gallery.json` + one safetensors file per prompt). Restricts
        /// which prompts are accepted; mandatory with `--model-dir` until
        /// the UMT5 text encoder lands in Phase 3.
        #[arg(long)]
        gallery: Option<PathBuf>,
        /// One web origin allowed to call the API from the browser
        /// (e.g. https://lightx2v.pages.dev). None = same-origin only.
        #[arg(long)]
        allow_origin: Option<String>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Offload {
    None,
    Model,
    Block,
    Disk,
}

impl From<Offload> for OffloadGranularity {
    fn from(o: Offload) -> Self {
        match o {
            Offload::None => OffloadGranularity::None,
            Offload::Model => OffloadGranularity::Model,
            Offload::Block => OffloadGranularity::Block,
            Offload::Disk => OffloadGranularity::Disk,
        }
    }
}

/// The resolved backend for a command, or the reason it is unavailable.
enum Selected {
    Cpu(CpuBackend),
    #[cfg(feature = "wgpu")]
    Wgpu(Box<WgpuBackend>),
    Unavailable(String),
}

impl Selected {
    fn label(&self) -> String {
        match self {
            Selected::Cpu(_) => "cpu".to_string(),
            #[cfg(feature = "wgpu")]
            Selected::Wgpu(be) => be.name(),
            Selected::Unavailable(reason) => format!("unavailable: {reason}"),
        }
    }
}

fn select(opt: BackendOpt) -> Selected {
    match opt {
        BackendOpt::Cpu => Selected::Cpu(CpuBackend),
        BackendOpt::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                match WgpuBackend::try_new() {
                    Some(be) => Selected::Wgpu(Box::new(be)),
                    None => Selected::Unavailable(
                        "no WebGPU adapter answered (no GPU driver / lavapipe on this machine)"
                            .to_string(),
                    ),
                }
            }
            #[cfg(not(feature = "wgpu"))]
            {
                Selected::Unavailable(
                    "this build was compiled without the wgpu feature".to_string(),
                )
            }
        }
        BackendOpt::Auto => {
            #[cfg(feature = "wgpu")]
            {
                if let Some(be) = WgpuBackend::try_new() {
                    return Selected::Wgpu(Box::new(be));
                }
            }
            Selected::Cpu(CpuBackend)
        }
    }
}

/// Print the skip reason and exit 0 (the backend is unavailable, not broken).
fn skip(reason: &str) -> ! {
    println!("skip: {reason}");
    std::process::exit(0);
}

fn geometry_for(frames: usize) -> Result<Geometry, String> {
    match frames {
        81 => Ok(Geometry::SHORT_480P),
        1005 => Ok(Geometry::LONG_60S),
        _ => Err(format!(
            "unsupported frame count {frames}: only 81 (short) and 1005 (60 s demo) are supported"
        )),
    }
}

fn generation_for(frames: usize) -> Result<GenerationConfig, String> {
    match frames {
        81 => Ok(GenerationConfig::short_clip()),
        1005 => Ok(GenerationConfig::demo()),
        _ => Err(format!("unsupported frame count {frames}")),
    }
}

/// Print the backend's per-op profile when `LIGHTX2V_PROFILE=1` is set
/// (the CPU backend records one; other backends return `None` unless they
/// implement `Backend::profile_report`).
fn print_profile<B: Backend>(be: &B) {
    if let Some(report) = be.profile_report() {
        println!("{report}");
    }
}

fn run_check<B: Backend>(be: &B) {
    println!("backend: {}", be.name());
    let results = check::run_all_on(be);
    let mut failed = 0;
    for r in &results {
        let status = if r.passed() { "PASS" } else { "FAIL" };
        if !r.passed() {
            failed += 1;
        }
        println!(
            "{status} {:<38} max_abs_err={:.3e} tol={:.0e}",
            r.name, r.max_abs_err, r.tolerance
        );
    }
    println!(
        "{} checks, {} passed, {} failed",
        results.len(),
        results.len() - failed,
        failed
    );
    print_profile(be);
    if failed > 0 {
        std::process::exit(1);
    }
}

fn run_bench<B: Backend>(be: &B, layers: usize) {
    println!("backend: {}", be.name());
    let mut cfg = WanConfig::tiny();
    cfg.layers = layers;
    let src = Synthetic { seed: 1 };
    let t0 = Instant::now();
    let dit = LightDit::load(be, cfg.clone(), MemRopeConfig::default_memrope(), &src)
        .unwrap_or_else(|e| {
            eprintln!("error: failed to load synthetic DiT: {e}");
            std::process::exit(1);
        });
    let load_s = t0.elapsed().as_secs_f64();
    let tae = TaeHv::load(be, TaeConfig::tiny(), &src).unwrap();
    let pipe = CausalPipeline::new(dit, tae, DistillConfig::dmd2());
    // One chunk of the tiny geometry: 3 latent frames, 4x4 latents.
    let gen = GenerationConfig {
        latent_frames: 3,
        ..GenerationConfig::short_clip()
    };
    let geometry = Geometry {
        frames: 9,
        height: 64,
        width: 64,
    };
    let embeds = vec![0.0f32; cfg.text_len * cfg.text_dim];
    let mut timings = Timings::default();
    let opt = Options {
        geometry,
        seed: 0,
        ffn_chunk: usize::MAX,
    };
    let t0 = Instant::now();
    match pipe.denoise(be, &embeds, &gen, &opt, &mut timings) {
        Ok(latent) => {
            println!(
                "bench: tiny DiT ({} layers), 1 chunk x 4 DMD steps: {:.3} s total \
                 (load {:.3} s, prompt {:.3} s, denoise {:.3} s), latent {} elems",
                layers,
                t0.elapsed().as_secs_f64(),
                load_s,
                timings.prompt_s,
                timings.denoise_s(),
                latent.len()
            );
        }
        Err(e) => {
            eprintln!("error: bench failed: {e}");
            std::process::exit(1);
        }
    }
    print_profile(be);
}

/// One-line per-chunk summary (mean/min/max/first/last). Flatness across
/// chunks is the MemRoPE-cache-bounding check: early chunks run on a
/// filling cache, later chunks on a full 12-frame one.
fn print_chunk_stats(t: &Timings) {
    if t.chunks_s.is_empty() {
        return;
    }
    let n = t.chunks_s.len();
    let sum: f64 = t.chunks_s.iter().sum();
    let lo = t.chunks_s.iter().fold(f64::MAX, |a, &b| a.min(b));
    let hi = t.chunks_s.iter().fold(0.0f64, |a, &b| a.max(b));
    println!(
        "chunks: {n} x mean {:.3} s (min {lo:.3} s, max {hi:.3} s, first {:.3} s, last {:.3} s)",
        sum / n as f64,
        t.chunks_s[0],
        t.chunks_s[n - 1]
    );
}

fn run_generate_real<B: Backend>(
    be: &B,
    dir: &Path,
    embeds_path: &Path,
    weights: WeightQuant,
    frames: usize,
    seed: u64,
    out: &Path,
) {
    println!("backend: {}", be.name());
    let cfg = WanConfig::wan_1_3b();
    let src = match SafetensorsDir::open(dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot open model dir {}: {e}", dir.display());
            std::process::exit(1);
        }
    };
    for (manifest, what) in [
        (wan_manifest(&cfg), "DiT"),
        (tae_manifest(&TaeConfig::taew2_1()), "TAEHV"),
    ] {
        let miss = missing(&src, &manifest);
        if !miss.is_empty() {
            eprintln!(
                "error: checkpoint in {} is missing {} {what} tensor(s), first: {}",
                dir.display(),
                miss.len(),
                miss[..miss.len().min(5)].join(", ")
            );
            std::process::exit(1);
        }
    }
    // Prompt embeddings: `prompt_embeds` [L ≤ 512, 4096], zero-padded to the
    // model's 512 positions, as Diffusers does.
    let embeds = match SafetensorsDir::open(embeds_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: cannot open embeds {}: {e}", embeds_path.display());
            std::process::exit(1);
        }
    };
    let prompt_embeds = match embeds.raw("prompt_embeds") {
        Ok((_, shape, _)) => {
            if shape.len() != 2 || shape[1] != cfg.text_dim || shape[0] > cfg.text_len {
                eprintln!(
                    "error: prompt_embeds in {} is {shape:?}, expected [L ≤ {}, {}]",
                    embeds_path.display(),
                    cfg.text_len,
                    cfg.text_dim
                );
                std::process::exit(1);
            }
            match embeds.get("prompt_embeds", &shape) {
                Ok(rows) => {
                    let mut v = vec![0.0f32; cfg.text_len * cfg.text_dim];
                    v[..rows.len()].copy_from_slice(&rows);
                    v
                }
                Err(e) => {
                    eprintln!("error: cannot read prompt_embeds: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("error: no prompt_embeds in {}: {e}", embeds_path.display());
            std::process::exit(1);
        }
    };
    let t0 = Instant::now();
    let dit = match LightDit::load_with(
        be,
        cfg.clone(),
        MemRopeConfig::default_memrope(),
        &src,
        weights,
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: failed to load DiT: {e}");
            std::process::exit(1);
        }
    };
    let tae = match TaeHv::load(be, TaeConfig::taew2_1(), &src) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: failed to load TAEHV: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "loaded real DiT + TAEHV in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let pipe = CausalPipeline::new(dit, tae, DistillConfig::dmd2());
    let (gen, geometry) = if frames == 1005 {
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
    let t0 = Instant::now();
    match pipe.denoise(be, &prompt_embeds, &gen, &opt, &mut timings) {
        Ok(latent) => {
            let path = out.with_extension("latent");
            let bytes: Vec<u8> = latent.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("error: cannot write {}: {e}", path.display());
                std::process::exit(1);
            }
            println!(
                "real latent: {} latent frames ({} video frames), {} elems, \
                 {:.3} s (prompt {:.3} s, denoise {:.3} s) -> {}",
                gen.latent_frames,
                gen.video_frames(),
                latent.len(),
                t0.elapsed().as_secs_f64(),
                timings.prompt_s,
                timings.denoise_s(),
                path.display()
            );
            print_chunk_stats(&timings);
            let dt0 = Instant::now();
            let decoded = pipe.decode(
                be,
                &latent,
                gen.latent_frames,
                geometry.height,
                geometry.width,
            );
            let decode_s = dt0.elapsed().as_secs_f64();
            match decoded {
                Ok(video) => {
                    let vpath = out.with_extension("rgb");
                    if let Err(e) = std::fs::write(&vpath, &video.rgb) {
                        eprintln!("error: cannot write {}: {e}", vpath.display());
                        std::process::exit(1);
                    }
                    println!(
                        "real video: {} frames, {}x{}, {} bytes, decode {:.3} s -> {}",
                        video.frames,
                        video.width,
                        video.height,
                        video.rgb.len(),
                        decode_s,
                        vpath.display()
                    );
                }
                Err(e) => {
                    eprintln!("error: TAEHV decode failed: {e}");
                    std::process::exit(1);
                }
            }
            print_profile(be);
        }
        Err(e) => {
            eprintln!("error: generate failed: {e}");
            std::process::exit(1);
        }
    }
}

fn run_generate_synthetic<B: Backend>(be: &B, seed: u64, frames: usize, out: &Path) {
    println!("backend: {}", be.name());
    let cfg = WanConfig::tiny();
    let src = Synthetic { seed: 1 };
    let dit = LightDit::load(be, cfg.clone(), MemRopeConfig::default_memrope(), &src)
        .unwrap_or_else(|e| {
            eprintln!("error: failed to load synthetic DiT: {e}");
            std::process::exit(1);
        });
    let tae = TaeHv::load(be, TaeConfig::tiny(), &src).unwrap();
    let pipe = CausalPipeline::new(dit, tae, DistillConfig::dmd2());
    let embeds = vec![0.0f32; cfg.text_len * cfg.text_dim];
    let mut timings = Timings::default();
    // The synthetic run honors --frames (81 or 1005, validated by the
    // dispatcher): same tiny DiT, 64x64 toy geometry, full rollout.
    let toy_gen = GenerationConfig {
        latent_frames: Geometry::latent_frames(frames),
        ..GenerationConfig::short_clip()
    };
    let toy_geometry = Geometry {
        frames,
        height: 64,
        width: 64,
    };
    let opt = Options {
        geometry: toy_geometry,
        seed,
        ffn_chunk: usize::MAX,
    };
    let t0 = Instant::now();
    match pipe.denoise(be, &embeds, &toy_gen, &opt, &mut timings) {
        Ok(latent) => {
            let path = out.with_extension("latent");
            let bytes: Vec<u8> = latent.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("error: cannot write {}: {e}", path.display());
                std::process::exit(1);
            }
            println!(
                "synthetic latent: {} latent frames ({} video frames), {} elems, \
                 {:.3} s (prompt {:.3} s, denoise {:.3} s) -> {}",
                toy_gen.latent_frames,
                toy_gen.video_frames(),
                latent.len(),
                t0.elapsed().as_secs_f64(),
                timings.prompt_s,
                timings.denoise_s(),
                path.display()
            );
            print_chunk_stats(&timings);
            let dt0 = Instant::now();
            let decoded = pipe.decode(be, &latent, toy_gen.latent_frames, 64, 64);
            let decode_s = dt0.elapsed().as_secs_f64();
            match decoded {
                Ok(video) => {
                    let vpath = out.with_extension("rgb");
                    if let Err(e) = std::fs::write(&vpath, &video.rgb) {
                        eprintln!("error: cannot write {}: {e}", vpath.display());
                        std::process::exit(1);
                    }
                    println!(
                        "synthetic video: {} frames, {}x{}, {} bytes, decode {:.3} s -> {}",
                        video.frames,
                        video.width,
                        video.height,
                        video.rgb.len(),
                        decode_s,
                        vpath.display()
                    );
                }
                Err(e) => {
                    eprintln!("error: TAEHV decode failed: {e}");
                    std::process::exit(1);
                }
            }
            print_profile(be);
        }
        Err(e) => {
            eprintln!("error: generate failed: {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Plan {
            vram_gib,
            frames,
            offload,
            weight_bytes,
        } => {
            let g = match geometry_for(frames) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(2);
                }
            };
            let gen = generation_for(frames).unwrap();
            let p = plan(
                &WanConfig::wan_1_3b(),
                &TaeConfig::taew2_1(),
                &gen,
                &MemRopeConfig::default_memrope(),
                &g,
                &Device::gib(vram_gib, weight_bytes, 2),
                offload.into(),
            );
            // The plan is backend-independent; the selection is reported,
            // not required — an unavailable wgpu still prints the plan.
            let sel = select(cli.backend);
            println!(
                "LightX2V plan: {frames} frames, {vram_gib} GiB budget, offload {}, backend {}",
                OffloadGranularity::from(offload),
                sel.label()
            );
            print!("{p}");
        }
        Cmd::Check => match select(cli.backend) {
            Selected::Cpu(be) => run_check(&be),
            #[cfg(feature = "wgpu")]
            Selected::Wgpu(be) => run_check(be.as_ref()),
            Selected::Unavailable(reason) => skip(&reason),
        },
        Cmd::Bench { layers } => match select(cli.backend) {
            Selected::Cpu(be) => run_bench(&be, layers),
            #[cfg(feature = "wgpu")]
            Selected::Wgpu(be) => run_bench(be.as_ref(), layers),
            Selected::Unavailable(reason) => skip(&reason),
        },
        Cmd::Generate {
            synthetic,
            model_dir,
            prompt,
            frames,
            seed,
            out,
            embeds,
            weights,
        } => {
            let _ = prompt; // Without --embeds there is no text encoder yet.
                            // Validate the frame count early (only 81 / 1005 are supported);
                            // the synthetic run honors it too (7 chunks at 81, 84 at 1005).
            if let Err(e) = geometry_for(frames).and_then(|_| generation_for(frames)) {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
            if !synthetic {
                let dir = match model_dir {
                    Some(d) => d,
                    None => {
                        eprintln!(
                            "error: no model to generate with: pass --synthetic for the tiny \
                             random model, or --model-dir <dir> with a Wan2.1-T2V-1.3B DMD \
                             checkpoint (transformer *.safetensors + taew2_1.safetensors)"
                        );
                        std::process::exit(2);
                    }
                };
                let embeds_path = match embeds {
                    Some(e) => e,
                    None => {
                        match SafetensorsDir::open(&dir) {
                            Ok(src) => {
                                let miss = missing(&src, &wan_manifest(&WanConfig::wan_1_3b()));
                                if !miss.is_empty() {
                                    eprintln!(
                                        "error: checkpoint in {} is missing {} DiT tensor(s), first: {}",
                                        dir.display(),
                                        miss.len(),
                                        miss[..miss.len().min(5)].join(", ")
                                    );
                                    std::process::exit(1);
                                }
                            }
                            Err(e) => {
                                eprintln!("error: cannot open model dir {}: {e}", dir.display());
                                std::process::exit(1);
                            }
                        }
                        eprintln!(
                            "error: real-checkpoint generation needs prompt embeddings: pass \
                             --embeds <file.safetensors> with `prompt_embeds` (see \
                             scripts/encode_prompt.py), or wait for the UMT5 text encoder \
                             (Phase 3)"
                        );
                        std::process::exit(1);
                    }
                };
                match select(cli.backend) {
                    Selected::Cpu(be) => {
                        run_generate_real(&be, &dir, &embeds_path, weights, frames, seed, &out)
                    }
                    #[cfg(feature = "wgpu")]
                    Selected::Wgpu(be) => run_generate_real(
                        be.as_ref(),
                        &dir,
                        &embeds_path,
                        weights,
                        frames,
                        seed,
                        &out,
                    ),
                    Selected::Unavailable(reason) => skip(&reason),
                }
            }
            match select(cli.backend) {
                Selected::Cpu(be) => run_generate_synthetic(&be, seed, frames, &out),
                #[cfg(feature = "wgpu")]
                Selected::Wgpu(be) => run_generate_synthetic(be.as_ref(), seed, frames, &out),
                Selected::Unavailable(reason) => skip(&reason),
            }
        }
        Cmd::Quantize {
            transformer,
            weights,
            out,
        } => {
            // `native` writes every tensor as f16 (the browser's f16 option).
            let cfg = WanConfig::wan_1_3b();
            let t0 = Instant::now();
            let src = match SafetensorsDir::open(&transformer) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: cannot open {}: {e}", transformer.display());
                    std::process::exit(1);
                }
            };
            let miss = missing(&src, &wan_manifest(&cfg));
            if !miss.is_empty() {
                eprintln!(
                    "error: {} tensors missing, e.g. {}",
                    miss.len(),
                    miss[..miss.len().min(3)].join(", ")
                );
                std::process::exit(1);
            }
            let tensors = match lightx2v_core::export::quantize_dit(&src, &cfg, weights) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: quantize failed: {e}");
                    std::process::exit(1);
                }
            };
            let bytes: usize = tensors.iter().map(|t| t.3.len()).sum();
            if let Err(e) = lightx2v_core::export::write(
                &out,
                &tensors,
                lightx2v_core::export::metadata(weights),
            ) {
                eprintln!("error: cannot write {}: {e}", out.display());
                std::process::exit(1);
            }
            let src_bytes: u64 = std::fs::read_dir(&transformer)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                        .filter_map(|e| e.metadata().ok())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0);
            println!(
                "quantize: {} -> {} ({:.2} GiB from {:.2} GiB, {} tensors, {:.1} s)",
                transformer.display(),
                out.display(),
                bytes as f64 / (1 << 30) as f64,
                src_bytes as f64 / (1 << 30) as f64,
                tensors.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Serve {
            addr,
            synthetic,
            model_dir,
            gallery,
            allow_origin,
        } => {
            let weights = if synthetic {
                serve::Weights::Synthetic { gallery }
            } else {
                match model_dir {
                    Some(dir) => serve::Weights::Real {
                        model_dir: dir,
                        gallery,
                    },
                    None => {
                        eprintln!(
                            "error: serve needs --synthetic for the tiny random model, or \
                             --model-dir <dir> with a Wan2.1-T2V-1.3B DMD checkpoint \
                             (plus --gallery <dir> until the UMT5 text encoder lands)"
                        );
                        std::process::exit(2);
                    }
                }
            };
            let settings = serve::Settings {
                addr,
                weights,
                allow_origin,
            };
            let run = match select(cli.backend) {
                Selected::Cpu(be) => serve::serve(settings, move |s, shared, rx| {
                    serve::engine(be, s, shared, rx)
                }),
                #[cfg(feature = "wgpu")]
                Selected::Wgpu(be) => serve::serve(settings, move |s, shared, rx| {
                    serve::engine(*be, s, shared, rx)
                }),
                Selected::Unavailable(reason) => skip(&reason),
            };
            if let Err(e) = run {
                eprintln!("error: serve failed: {e}");
                std::process::exit(1);
            }
        }
    }
}
