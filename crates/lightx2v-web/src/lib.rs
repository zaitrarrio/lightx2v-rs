//! LightX2V in the browser: the WebGPU engine behind a wasm-bindgen API
//! for Prompt Studio's "In this browser" mode. The fastwan-web shape:
//!
//! ```js
//! const eng = await createEngine("lightx2v"); // or "tiny"
//! for (const blob of blocks) await eng.addBlock(blob); // one DiT block each
//! await eng.finishDit(headBlob);
//! await eng.loadTae(taeBlob);
//! await eng.generate(embeds, 832, 480, 81, 7, onEvent);
//! ```
//!
//! Events (all plain objects through the one `on_event` callback):
//!
//! * `{type: "prepared"}` — prompt embedded, workspaces allocated, the
//!   first `download_async` round-tripped (the engine is really alive).
//! * `{type: "step", chunk, index, total, ms}` — one DMD step of one
//!   causal chunk, with honest wall time (the fence buffer is downloaded
//!   after every step, so `ms` is GPU time, not queue time).
//! * `{type: "chunk", index, total, ms}` — one 3-latent-frame chunk
//!   denoised (the MemRoPE causal rollout: a 60 s clip is 84 chunks).
//! * `{type: "frames", first, count, width, height, rgb: Uint8Array}` —
//!   decoded frames as they land, TAEHV chunk by chunk.
//!
//! There is no blocking readback in a browser: everything that leaves the
//! GPU goes through [`WgpuBackend::download_async`]. The synchronous
//! `Backend::download` would hang the tab (its `device.poll` waits
//! forever on WebGPU).
//!
//! Two models: `"lightx2v"` is the real Wan2.1-T2V-1.3B DMD student —
//! blocks stream in from the page, one HTTP Range request per block — and
//! `"tiny"` is the synthetic random DiT (no download), the dev fallback
//! that proves the same path end to end. Prompts are precomputed gallery
//! embeddings in both cases: there is no text encoder in the browser
//! (UMT5-XXL is 11 GB). The studio page fetches one `prompt_embeds`
//! safetensors file per gallery prompt.
//!
//! Schedule note: lightx2v runs its own 4-step DMD schedule
//! `[1000, 750, 500, 250]` (shift 5.0) over the FastWan 3-step-distilled
//! weights — the History-Forcing 4-step checkpoint is the alternative
//! source. The schedule is a property of the student, not the weights.

// wasm-only crate (the fastwan-web pattern): on native targets this is an
// empty rlib, so `cargo test --workspace` never pulls the wasm toolchain in.
#![cfg(target_arch = "wasm32")]

use js_sys::{Function, Object, Reflect, Uint8Array};
use lightx2v_core::{
    backend::{Backend, OffloadGranularity},
    config::{DistillConfig, GenerationConfig, MemRopeConfig, TaeConfig, WanConfig},
    geometry::Geometry,
    plan::{plan, Device},
    rng::Rng,
    schedule::DmdSchedule,
    weights::{SafetensorsDir, Synthetic, WeightSource},
};
use lightx2v_model::{dit::DitBuilder, LightDit, TaeHv};
use lightx2v_wgpu::WgpuBackend;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

fn set(o: &Object, key: &str, value: impl Into<JsValue>) {
    let _ = Reflect::set(o, &JsValue::from_str(key), &value.into());
}

struct Inner {
    be: WgpuBackend,
    cfg: WanConfig,
    tcfg: TaeConfig,
    builder: RefCell<Option<DitBuilder<WgpuBackend>>>,
    // `Rc` handles, not `Ref` guards: `generate` clones them out before
    // the first await, so no `RefCell` borrow is ever held across an
    // await point (a re-entrant JS call could otherwise panic on
    // `borrow_mut` while the engine is suspended mid-step).
    dit: RefCell<Option<Rc<LightDit<WgpuBackend>>>>,
    tae: RefCell<Option<Rc<TaeHv<WgpuBackend>>>>,
}

/// The browser engine. Constructed by [`createEngine`], never directly.
#[wasm_bindgen]
pub struct Engine {
    inner: Rc<Inner>,
}

#[wasm_bindgen]
impl Engine {
    /// Adapter name, e.g. "NVIDIA GeForce RTX 4090 (device)".
    pub fn adapter(&self) -> String {
        self.inner.be.name()
    }

    /// Adapter limits in one line, shown next to the engine status.
    pub fn limits(&self) -> String {
        self.inner.be.limits_text()
    }

    /// Largest storage-buffer binding this GPU allows (bytes).
    #[wasm_bindgen(js_name = maxBinding)]
    pub fn max_binding(&self) -> f64 {
        self.inner.be.max_binding() as f64
    }

    /// Transformer blocks in the model (30 for the real DiT, 2 for tiny).
    pub fn layers(&self) -> u32 {
        self.inner.cfg.layers as u32
    }

    /// Blocks loaded so far (the page compares this with [`layers`]).
    #[wasm_bindgen(js_name = loadedBlocks)]
    pub fn loaded_blocks(&self) -> u32 {
        if self.inner.dit.borrow().is_some() {
            self.inner.cfg.layers as u32
        } else if let Some(b) = self.inner.builder.borrow().as_ref() {
            b.loaded() as u32
        } else {
            0
        }
    }

    /// Load one transformer block (`blocks.<n>.*`) from a standalone
    /// safetensors blob — the page slices one block out of the
    /// block-ordered checkpoint per Range request. Returns the block count
    /// loaded so far. The blob must hold exactly the next block's tensors.
    #[wasm_bindgen(js_name = addBlock)]
    pub fn add_block(&self, blob: Vec<u8>) -> Result<u32, JsValue> {
        let src = SafetensorsDir::from_bytes(blob).map_err(err)?;
        let mut b = self.inner.builder.borrow_mut();
        let b = b.as_mut().ok_or_else(|| {
            err("no block builder: the DiT is already finished (or this is the tiny model)")
        })?;
        b.add_block(&self.inner.be, &src).map_err(err)?;
        Ok(b.loaded() as u32)
    }

    /// Load the head tensors (patch embedder, text embedder, output
    /// projection, timestep MLP) from a standalone safetensors blob and
    /// complete the DiT. Fails unless every block was added first.
    #[wasm_bindgen(js_name = finishDit)]
    pub fn finish_dit(&self, blob: Vec<u8>) -> Result<(), JsValue> {
        let src = SafetensorsDir::from_bytes(blob).map_err(err)?;
        let mut b = self.inner.builder.borrow_mut();
        let b = b.take().ok_or_else(|| {
            err("no block builder: the DiT is already finished (or this is the tiny model)")
        })?;
        let dit = b.finish(&self.inner.be, &src).map_err(err)?;
        *self.inner.dit.borrow_mut() = Some(Rc::new(dit));
        Ok(())
    }

    /// Load the TAEHV decoder from a standalone safetensors blob.
    #[wasm_bindgen(js_name = loadTae)]
    pub fn load_tae(&self, blob: Vec<u8>) -> Result<(), JsValue> {
        let src = SafetensorsDir::from_bytes(blob).map_err(err)?;
        *self.inner.tae.borrow_mut() = Some(Rc::new(
            TaeHv::load(&self.inner.be, self.inner.tcfg.clone(), &src).map_err(err)?,
        ));
        Ok(())
    }

    /// Generate a clip from `prompt_embeds` (`fastwan encode` / gallery
    /// file), reporting progress and decoded frames to `on_event`.
    /// `frames` is the video frame count: 81 (short) or 1005 (60 s).
    pub fn generate(
        &self,
        embeds: Vec<u8>,
        width: usize,
        height: usize,
        frames: usize,
        seed: f64,
        on_event: Function,
    ) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let dit = inner
                .dit
                .borrow()
                .as_ref()
                .cloned()
                .ok_or_else(|| err("the model is not loaded"))?;
            let tae = inner
                .tae
                .borrow()
                .as_ref()
                .cloned()
                .ok_or_else(|| err("the decoder is not loaded"))?;
            run(
                &inner,
                &dit,
                &tae,
                embeds,
                Geometry {
                    frames,
                    height,
                    width,
                },
                seed as u64,
                &on_event,
            )
            .await
            .map_err(err)
            .map(|_| JsValue::UNDEFINED)
        })
    }
}

/// Create the engine: request the WebGPU adapter, then either build the
/// tiny synthetic DiT in place (`"tiny"`) or prepare the block builder
/// for the real model (`"lightx2v"` — the page streams blocks in).
#[wasm_bindgen(js_name = createEngine)]
pub fn create_engine(model: &str) -> js_sys::Promise {
    let model = model.to_string();
    wasm_bindgen_futures::future_to_promise(async move {
        console_error_panic_hook::set_once();
        let be = WgpuBackend::new_async()
            .await
            .map_err(|e| err(format!("no WebGPU adapter: {e}")))?;
        let (cfg, tcfg, builder, dit, tae) = match model.as_str() {
            "tiny" => {
                let cfg = WanConfig::tiny();
                let tcfg = TaeConfig::tiny();
                let src = Synthetic { seed: 1 };
                let dit = LightDit::load(&be, cfg.clone(), MemRopeConfig::default_memrope(), &src)
                    .map_err(err)?;
                let tae = TaeHv::load(&be, tcfg.clone(), &src).map_err(err)?;
                (cfg, tcfg, None, Some(Rc::new(dit)), Some(Rc::new(tae)))
            }
            "lightx2v" => {
                let cfg = WanConfig::wan_1_3b();
                let tcfg = TaeConfig::taew2_1();
                let b = DitBuilder::new(
                    cfg.clone(),
                    MemRopeConfig::default_memrope(),
                    lightx2v_core::quant::WeightQuant::Native,
                );
                (cfg, tcfg, Some(b), None, None)
            }
            other => {
                return Err(err(format!(
                    "unknown model {other:?}: \"lightx2v\" or \"tiny\""
                )));
            }
        };
        Ok(Engine {
            inner: Rc::new(Inner {
                be,
                cfg,
                tcfg,
                builder: RefCell::new(builder),
                dit: RefCell::new(dit),
                tae: RefCell::new(tae),
            }),
        }
        .into())
    })
}

async fn run(
    inner: &Inner,
    dit: &LightDit<WgpuBackend>,
    tae: &TaeHv<WgpuBackend>,
    embeds: Vec<u8>,
    g: Geometry,
    seed: u64,
    on: &Function,
) -> Result<(), String> {
    let emit = |o: Object| {
        let _ = on.call1(&JsValue::NULL, &o);
    };
    let (be, cfg) = (&inner.be, &inner.cfg);
    g.validate(cfg).map_err(|e| e.to_string())?;
    let gen = match g.frames {
        81 => GenerationConfig::short_clip(),
        1005 => GenerationConfig::demo(),
        other => {
            return Err(format!(
                "frames must be 81 (short) or 1005 (60 s), got {other}"
            ));
        }
    };

    // Prompt embeddings: [L ≤ text_len, text_dim], zero-padded.
    let src = SafetensorsDir::from_bytes(embeds).map_err(|e| e.to_string())?;
    let name = src
        .names()
        .find(|n| n.contains("embed"))
        .map(str::to_string)
        .ok_or("no prompt_embeds tensor")?;
    let (_, shape, _) = src.raw(&name).map_err(|e| e.to_string())?;
    let mut e = src.get(&name, &shape).map_err(|e| e.to_string())?;
    if e.len() % cfg.text_dim != 0 || e.len() > cfg.text_len * cfg.text_dim {
        return Err(format!(
            "prompt_embeds {shape:?} is not [L, {}]",
            cfg.text_dim
        ));
    }
    e.resize(cfg.text_len * cfg.text_dim, 0.0);

    let p = plan(
        cfg,
        &inner.tcfg,
        &gen,
        &MemRopeConfig::default_memrope(),
        &g,
        &Device {
            budget_bytes: u64::MAX,
            max_binding_bytes: be.max_binding(),
            weight_bytes: 2,
            act_bytes: 4,
        },
        OffloadGranularity::None,
    );
    // Refuse a size this GPU cannot bind before spending the denoise on it.
    if p.largest_binding > be.max_binding() {
        return Err(format!(
            "{}×{} needs a {} MiB GPU buffer; this browser's GPU allows {} ({}). Pick a smaller size.",
            g.width,
            g.height,
            p.largest_binding >> 20,
            be.max_binding() >> 20,
            be.limits_text()
        ));
    }
    let schedule = DmdSchedule::dmd2();
    let distill = DistillConfig::dmd2();
    let prompt = dit.prepare_prompt(be, &e);
    let mods: Vec<_> = schedule
        .timesteps
        .iter()
        .map(|&ts| dit.step_mods(be, ts as f32))
        .collect();

    // The causal rollout, mirroring `CausalPipeline::denoise_with` but
    // with async readback: one 3-latent-frame chunk at a time, the MemRoPE
    // cache persisting across chunks in the shared workspace.
    let (lt, lh, lw) = g.latent();
    let chunk_frames = distill.num_frame_per_block;
    let chunks = lt / chunk_frames;
    let frame_tokens = (lh / cfg.patch) * (lw / cfg.patch);
    let chunk_tokens = chunk_frames * frame_tokens;
    let chunk_latent_len = cfg.in_channels * chunk_frames * lh * lw;
    let mut ws = dit.workspace(be, chunk_frames, (lh, lw), 0, p.ffn_chunk.min(chunk_tokens));
    let fence = be.alloc(1);
    be.download_async(&fence)
        .await
        .map_err(|e| format!("engine self-check failed: {e}"))?;
    emit(Object::new().tap(|o| set(o, "type", "prepared")));

    let mut rng = Rng::new(seed);
    let mut st = tae.state();
    let mut done = 0usize;
    // Decode one latent frame at a time above 480×272: it halves the
    // decoder's peak memory (browsers run on laptop GPUs too). Chunking
    // does not change the frames.
    let decode_n = if g.width * g.height > 480 * 272 { 1 } else { 2 };
    let decode_err = |e: String| format!("decode failed: {e} ({})", be.limits_text());
    for ci in 0..chunks {
        let ct0 = now_ms();
        let start_frame = ci * chunk_frames;
        let x = be.upload(&rng.normal(chunk_latent_len));
        let v = be.alloc(chunk_latent_len);
        let noise = be.alloc(chunk_latent_len);
        let mut chunk_ws = dit.workspace(be, chunk_frames, (lh, lw), start_frame, ws.ffn_chunk);
        // Move the persistent caches into the chunk workspace.
        std::mem::swap(&mut chunk_ws.caches, &mut ws.caches);
        let v_tokens = be.alloc(chunk_tokens * cfg.patch_features_out());
        for (i, m) in mods.iter().enumerate() {
            let t0 = now_ms();
            dit.patch_embed(be, &chunk_ws, &x, cfg.in_channels, chunk_frames, lh, lw);
            dit.forward_chunk(
                be,
                &mut chunk_ws,
                m,
                &prompt,
                start_frame,
                frame_tokens,
                &v_tokens,
            );
            be.unpatchify(
                &v_tokens,
                cfg.out_channels,
                chunk_frames,
                lh,
                lw,
                cfg.patch,
                &v,
            );
            let (st_sigma, next) = schedule.coeffs(i);
            if next.is_some() {
                be.write(&noise, &rng.normal(chunk_latent_len));
            }
            be.dmd_step(
                &x,
                &v,
                next.map(|_| &noise),
                st_sigma,
                next.unwrap_or(0.0),
                chunk_latent_len,
            );
            be.download_async(&fence).await.map_err(|e| e.to_string())?; // wait for the step: honest timing
            let o = Object::new();
            set(&o, "type", "step");
            set(&o, "chunk", ci as u32);
            set(&o, "index", i as u32);
            set(&o, "total", mods.len() as u32);
            set(&o, "ms", now_ms() - t0);
            emit(o);
        }
        // Return the caches to the persistent workspace.
        std::mem::swap(&mut chunk_ws.caches, &mut ws.caches);
        let chunk_latent = be.download_async(&x).await.map_err(|e| e.to_string())?;
        let o = Object::new();
        set(&o, "type", "chunk");
        set(&o, "index", ci as u32);
        set(&o, "total", chunks as u32);
        set(&o, "ms", now_ms() - ct0);
        emit(o);

        // Decode this chunk's latent frames as they land, frame-major
        // ([frames, C, h, w]) from the channel-major chunk latent.
        let c = tae.cfg.latent_channels;
        let plane = lh * lw;
        let mut f0 = 0;
        while f0 < chunk_frames {
            let nfr = decode_n.min(chunk_frames - f0);
            let mut z = Vec::with_capacity(nfr * c * plane);
            for f in f0..f0 + nfr {
                for ch in 0..c {
                    z.extend_from_slice(&chunk_latent[(ch * chunk_frames + f) * plane..][..plane]);
                }
            }
            let ck = tae.launch_chunk(be, &mut st, &z, nfr, lh, lw);
            let raw = be.download_async(&ck.rgb).await.map_err(decode_err)?;
            let (count, h, w, rgb) = ck.rgb8(&raw);
            let o = Object::new();
            set(&o, "type", "frames");
            set(&o, "first", done as u32);
            set(&o, "count", count as u32);
            set(&o, "width", w as u32);
            set(&o, "height", h as u32);
            set(&o, "rgb", Uint8Array::from(&rgb[..]));
            emit(o);
            done += count;
            f0 += nfr;
        }
    }
    Ok(())
}

trait Tap: Sized {
    fn tap(self, f: impl FnOnce(&Self)) -> Self {
        f(&self);
        self
    }
}
impl Tap for Object {}

/// Install the panic hook without constructing an engine (the page calls
/// this implicitly through `createEngine`, so it is only needed for
/// diagnostics).
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}
