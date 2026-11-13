//! `lightx2v`: plan / check / bench / generate for the Rust LightX2V +
//! History-Forcing port.
//!
//! `generate` with `--synthetic` runs the tiny random model end to end on
//! the CPU and writes the denoised latent; without it, it needs a real
//! checkpoint directory and says exactly which tensors are missing.
//! There is no video output in Phase 1: the TAEHV decode is a stub that
//! returns `NotImplemented`.

use clap::{Parser, Subcommand, ValueEnum};
use lightx2v_core::{
    backend::OffloadGranularity,
    config::{DistillConfig, GenerationConfig, MemRopeConfig, TaeConfig, WanConfig},
    geometry::Geometry,
    plan::{plan, Device},
    weights::{missing, wan_manifest, Synthetic},
};
use lightx2v_cpu::CpuBackend;
use lightx2v_model::{
    check,
    pipeline::{CausalPipeline, Options, Timings},
    LightDit, TaeHv,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "lightx2v",
    about = "LightX2V + History-Forcing long-video inference, in Rust"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
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
    /// Run the CPU conformance checks (every op vs its f32 reference).
    Check,
    /// Time the tiny synthetic DiT on the CPU: one chunk, all 4 DMD steps.
    Bench {
        /// DiT layers in the synthetic model.
        #[arg(long, default_value = "2")]
        layers: usize,
    },
    /// Denoise a latent with the tiny synthetic model (CPU) or a real
    /// checkpoint directory. Writes `<out>.latent` (raw f32); video decode
    /// is Phase 2.
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

fn geometry_for(frames: usize) -> Result<Geometry, String> {
    match frames {
        81 => Ok(Geometry::SHORT_480P),
        1005 => Ok(Geometry::LONG_60S),
        _ => Err(format!(
            "unsupported frame count {frames}: Phase 1 supports 81 (short) and 1005 (60 s demo)"
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
            println!(
                "LightX2V plan: {frames} frames, {vram_gib} GiB budget, offload {}",
                OffloadGranularity::from(offload)
            );
            print!("{p}");
        }
        Cmd::Check => {
            let results = check::run_all();
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
            if failed > 0 {
                std::process::exit(1);
            }
        }
        Cmd::Bench { layers } => {
            let be = CpuBackend;
            let mut cfg = WanConfig::tiny();
            cfg.layers = layers;
            let src = Synthetic { seed: 1 };
            let t0 = Instant::now();
            let dit = LightDit::load(&be, cfg.clone(), MemRopeConfig::default_memrope(), &src)
                .unwrap_or_else(|e| {
                    eprintln!("error: failed to load synthetic DiT: {e}");
                    std::process::exit(1);
                });
            let load_s = t0.elapsed().as_secs_f64();
            let tae = TaeHv::load(&be, TaeConfig::tiny(), &src).unwrap();
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
            match pipe.denoise(&be, &embeds, &gen, &opt, &mut timings) {
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
        }
        Cmd::Generate {
            synthetic,
            model_dir,
            prompt,
            frames,
            seed,
            out,
        } => {
            let _ = prompt; // Phase 1: no text encoder; embeddings are zeros.
                            // Validate the frame count early (only 81 / 1005 are supported);
                            // the synthetic run itself is always the toy shape (see below).
            if let Err(e) = geometry_for(frames).and_then(|_| generation_for(frames)) {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
            let be = CpuBackend;
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
                match lightx2v_core::weights::SafetensorsDir::open(&dir) {
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
                        eprintln!(
                            "error: real-checkpoint generation is Phase 2 work: the DiT \
                             manifest validated, but the TAEHV decode and the text encoder \
                             (UMT5) are not ported yet"
                        );
                        std::process::exit(1);
                    }
                    Err(e) => {
                        eprintln!("error: cannot open model dir {}: {e}", dir.display());
                        std::process::exit(1);
                    }
                }
            }
            // --synthetic runs a toy rollout: the tiny DiT (2 layers, dim 24)
            // on 3 latent frames of a 64x64 clip. The 480p geometries need
            // the real 1.3B config, so --frames only selects the plan numbers
            // above; the synthetic run itself is always the toy shape, and
            // says so.
            let cfg = WanConfig::tiny();
            let src = Synthetic { seed: 1 };
            let dit = LightDit::load(&be, cfg.clone(), MemRopeConfig::default_memrope(), &src)
                .unwrap_or_else(|e| {
                    eprintln!("error: failed to load synthetic DiT: {e}");
                    std::process::exit(1);
                });
            let tae = TaeHv::load(&be, TaeConfig::tiny(), &src).unwrap();
            let pipe = CausalPipeline::new(dit, tae, DistillConfig::dmd2());
            let embeds = vec![0.0f32; cfg.text_len * cfg.text_dim];
            let mut timings = Timings::default();
            let toy_gen = GenerationConfig {
                latent_frames: 3,
                ..GenerationConfig::short_clip()
            };
            let toy_geometry = Geometry {
                frames: 9,
                height: 64,
                width: 64,
            };
            let opt = Options {
                geometry: toy_geometry,
                seed,
                ffn_chunk: usize::MAX,
            };
            let t0 = Instant::now();
            match pipe.denoise(&be, &embeds, &toy_gen, &opt, &mut timings) {
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
                    println!(
                        "note: video decode (TAEHV) is Phase 2; this file is the raw f32 latent"
                    );
                }
                Err(e) => {
                    eprintln!("error: generate failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
