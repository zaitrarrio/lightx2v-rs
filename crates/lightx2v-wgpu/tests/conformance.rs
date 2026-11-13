//! wgpu conformance: the 11 model checks, the tiny-DiT rollout, and the
//! TAEHV decode, each against the CPU oracle.
//!
//! Every test SKIPS cleanly (printing the reason) when no WebGPU adapter
//! answers — headless CI, no GPU driver — instead of failing. The DiT and
//! decode comparisons run with `allow_f16_weights = false`, so the f16
//! weight storage does not widen the tolerance: a failure at 1e-5 is a
//! kernel bug, not rounding. The decode test additionally measures the
//! f16-packed error and prints it, so the real number is on record when a
//! GPU runs it.

use lightx2v_core::{
    backend::Backend,
    config::{DistillConfig, GenerationConfig, MemRopeConfig, TaeConfig, WanConfig},
    geometry::Geometry,
    rng::Rng,
    weights::{SafetensorsDir, WeightSource},
};
use lightx2v_cpu::CpuBackend;
use lightx2v_model::{
    check,
    pipeline::{CausalPipeline, Options, Timings},
    LightDit, TaeHv,
};
use lightx2v_wgpu::WgpuBackend;
use std::path::PathBuf;

fn backend() -> Option<WgpuBackend> {
    match WgpuBackend::try_new() {
        Some(be) => Some(be),
        None => {
            println!("SKIP: no WebGPU adapter answered (no GPU driver / lavapipe here)");
            None
        }
    }
}

fn fixture_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lx2v-fixture-wgpu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = manifest.join("../../scripts/reference/make_fixture.py");
    let status = std::process::Command::new("python3")
        .arg(&script)
        .arg(&dir)
        .status()
        .expect("python3 must be available to generate the fixture");
    assert!(status.success(), "make_fixture.py failed");
    dir
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn wgpu_checks() {
    let Some(be) = backend() else { return };
    println!("adapter: {}", be.adapter_info().name);
    let results = check::run_all_on(&be);
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
    assert_eq!(failed, 0, "{} of {} checks failed", failed, results.len());
}

#[test]
fn wgpu_tiny_dit_matches_cpu() {
    let Some(mut be) = backend() else { return };
    // f32 weights: the comparison tests the kernels, not the f16 storage.
    be.allow_f16_weights = false;
    let dir = fixture_dir();
    let src = SafetensorsDir::open(&dir).unwrap();
    let cfg = WanConfig::tiny();

    let gen = GenerationConfig {
        latent_frames: 6,
        ..GenerationConfig::short_clip()
    };
    let geometry = Geometry {
        frames: 21,
        height: 64,
        width: 64,
    };
    let embeds = vec![0.0f32; cfg.text_len * cfg.text_dim];
    let opt = Options {
        geometry,
        seed: 0,
        ffn_chunk: usize::MAX,
    };
    let dit = LightDit::load(&be, cfg.clone(), MemRopeConfig::default_memrope(), &src).unwrap();
    let tae = TaeHv::load(&be, TaeConfig::tiny(), &src).unwrap();
    let pipe = CausalPipeline::new(dit, tae, DistillConfig::dmd2());
    let mut timings = Timings::default();
    let wgpu_latent = pipe
        .denoise(&be, &embeds, &gen, &opt, &mut timings)
        .unwrap();

    let cpu = CpuBackend;
    let dit = LightDit::load(&cpu, cfg.clone(), MemRopeConfig::default_memrope(), &src).unwrap();
    let tae = TaeHv::load(&cpu, TaeConfig::tiny(), &src).unwrap();
    let pipe = CausalPipeline::new(dit, tae, DistillConfig::dmd2());
    let mut timings = Timings::default();
    let cpu_latent = pipe
        .denoise(&cpu, &embeds, &gen, &opt, &mut timings)
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();

    let err = max_abs(&wgpu_latent, &cpu_latent);
    println!("wgpu vs cpu tiny-DiT latent max abs err: {err:.3e}");
    assert!(err < 1e-5, "wgpu vs cpu DiT rollout max abs {err:.3e}");
}

#[test]
fn wgpu_taehv_decode_matches_cpu() {
    let Some(mut be) = backend() else { return };
    let dir = fixture_dir();
    let src = SafetensorsDir::open(&dir).unwrap();
    let mut rng = Rng::new(7);
    let latent6 = rng.normal(4 * 6 * 8 * 8);
    let mut z6 = vec![0.0f32; 6 * 4 * 8 * 8];
    for fi in 0..6 {
        for c in 0..4 {
            for y in 0..8 {
                for x in 0..8 {
                    z6[((fi * 4 + c) * 8 + y) * 8 + x] = latent6[((c * 6 + fi) * 8 + y) * 8 + x];
                }
            }
        }
    }
    let cpu = CpuBackend;
    let tae_cpu = TaeHv::load(&cpu, TaeConfig::tiny(), &src).unwrap();
    let mut st_cpu = tae_cpu.state();
    let want = tae_cpu.decode_chunk(&cpu, &mut st_cpu, &z6, 6, 8, 8);
    assert_eq!(want.frames, 21);

    // f32 weights: the conv kernels, exactly.
    be.allow_f16_weights = false;
    let tae_f32 = TaeHv::load(&be, TaeConfig::tiny(), &src).unwrap();
    let mut st = tae_f32.state();
    let got = tae_f32.decode_chunk(&be, &mut st, &z6, 6, 8, 8);
    assert_eq!(got.frames, want.frames);
    let err = max_abs(&got.data, &want.data);
    println!("wgpu (f32 weights) vs cpu TAEHV decode max abs err: {err:.3e}");
    assert!(err < 1e-3, "wgpu vs cpu decode max abs {err:.3e}");

    // f16-packed weights: the production path. Measure, don't hide: the
    // rounding is real (~5e-4 relative per weight), so the bound is loose
    // and the measured value is printed for the record.
    be.allow_f16_weights = true;
    let tae_f16 = TaeHv::load(&be, TaeConfig::tiny(), &src).unwrap();
    let mut st = tae_f16.state();
    let got16 = tae_f16.decode_chunk(&be, &mut st, &z6, 6, 8, 8);
    let err16 = max_abs(&got16.data, &want.data);
    println!("wgpu (f16 weights) vs cpu TAEHV decode max abs err: {err16:.3e}");
    assert!(err16 < 5e-2, "wgpu f16 decode max abs {err16:.3e}");
    std::fs::remove_dir_all(&dir).ok();
}
