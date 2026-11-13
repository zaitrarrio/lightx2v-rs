//! WebGPU backend (wgpu): one WGSL kernel per [`Backend`] op, plus the
//! LightX2V extensions — causal masked attention, the MemRoPE KV cache,
//! and sparse (VSA) attention.
//!
//! The design mirrors `fastwan-wgpu` (zaitrarrio/fastwan-rs), reimplemented
//! here against the `lightx2v_core::Backend` trait:
//!
//! * Activations are f32; weights are f16 pairs packed in `u32` and
//!   unpacked with core `unpack2x16float`, which halves weight memory and
//!   bandwidth without depending on the optional `shader-f16` feature. A
//!   weight whose values do not fit f16 (or whose row length is odd) stays
//!   f32 — never silently clipped. Q8/Q4 weights stay packed and are
//!   dequantized in the shader; FP8 weights are dequantized on the host
//!   into the same f16-packed form (the dequantize-as-reference contract
//!   holds either way).
//! * Dispatches are recorded into one command encoder and submitted in
//!   batches; a host write or read flushes first, so queue ordering always
//!   matches program order.
//! * The attention kernel is one WGSL module with three modes: dense,
//!   VSA fine stage (selected tiles + coarse·gate combine), and causal
//!   masked (MemRoPE block mask). Phase 2 uses the simple one-row-per-thread
//!   flash kernel (fastwan's fallback shape); the register-tiled variant
//!   is Phase 3 perf work, not a correctness gap.
//!
//! ## KV cache layout (wgpu)
//!
//! One layer's cache keeps its K/V on the device as `[rows, heads·d]`
//! row-major over tokens — the layout the CPU oracle uses, so `kv_cache_kv`
//! is a transpose away from attention's `[heads, rows, d]` input. The
//! temporal index and [`SlotKind`] of every row live on the host (small
//! `Vec`s); the device never sees them. `kv_cache_put` therefore decides
//! append-vs-overwrite on the host and issues one `kv_pack` kernel either
//! way. `kv_cache_memrope_update` runs the same metadata simulation as the
//! CPU oracle and executes it with two kernels: `ema_fold` (in place, one
//! frame into an EMA slot) and `kv_compact` (gather rows by an index map,
//! zero-filling freshly seeded EMA frames).

use lightx2v_core::{
    backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind},
    config::MemRopeConfig,
    memrope::SlotKind,
    quant::{Fp8E4M3, Q4, Q8},
    sparse::SparseSpec,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const COMMON: &str = include_str!("shaders/common.wgsl");
const LINEAR: &str = include_str!("shaders/linear.wgsl");
const CONV: &str = include_str!("shaders/conv.wgsl");
const NORM: &str = include_str!("shaders/norm.wgsl");
const HEADS: &str = include_str!("shaders/head_split.wgsl");
const ELEM: &str = include_str!("shaders/elementwise.wgsl");
const FLASH: &str = include_str!("shaders/flash.wgsl");
const VSA_POOL: &str = include_str!("shaders/vsa_pool.wgsl");
const VSA_COARSE: &str = include_str!("shaders/vsa_coarse.wgsl");
const KV_COMPACT: &str = include_str!("shaders/kv_compact.wgsl");

/// Submit after this many recorded dispatches.
const BATCH: usize = 64;
/// Max workgroups per dispatch dimension (WebGPU default limit).
const MAX_DIM: u32 = 65535;
/// Padding slots in the VSA tile tables and the MemRoPE row maps.
const PAD: u32 = u32::MAX;

#[derive(Clone)]
pub struct GBuf {
    buf: Arc<wgpu::Buffer>,
    len: usize,
}

/// Sparse-attention state for one token grid.
pub struct GSparse {
    slot: GBuf,
    bsz: GBuf,
    pooled: GBuf,
    coarse: GBuf,
    sel: GBuf,
    nb: usize,
    topk: usize,
    heads: usize,
    d: usize,
    seq: usize,
    scale: f32,
}

pub struct GWeight {
    buf: GBuf,
    rows: usize,
    cols: usize,
    f16: bool,
    /// Weight-only quantized: `buf` holds the codes, this buffer the
    /// per-group scales (Q8) or interleaved scale / zero pairs (Q4), then the
    /// group size.
    quant: Option<(Quant, GBuf, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Quant {
    /// [`Q8`]: 4 codes per word.
    Q8,
    /// [`Q4`]: 8 codes per word.
    Q4,
}

/// One layer's KV cache: K/V on the device as `[rows, heads·d]`, temporal
/// indices and slot kinds on the host.
pub struct GKvCache {
    heads: usize,
    head_dim: usize,
    capacity: usize,
    k: GBuf,
    v: GBuf,
    temporal: Vec<usize>,
    kinds: Vec<SlotKind>,
}

impl GKvCache {
    fn rows(&self) -> usize {
        self.temporal.len()
    }
    fn width(&self) -> usize {
        self.heads * self.head_dim
    }
}

struct Pending {
    enc: Option<wgpu::CommandEncoder>,
    n: usize,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    /// K/V keys per flash chunk: 32 when the device has the shared memory
    /// for it, else 16, else 8.
    attn_bk: usize,
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    pending: Mutex<Pending>,
    dummy: GBuf,
    /// In the browser: the first uncaptured WebGPU error or the device-loss
    /// reason. Natively wgpu's default handler panics, which inside a JS
    /// callback leaves the caller's promise pending forever (a silent hang).
    fault: Arc<Mutex<Option<String>>>,
    pub allow_f16_weights: bool,
}

#[derive(Debug)]
pub struct InitError(pub String);

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InitError {}

impl WgpuBackend {
    /// The first high-performance adapter, with the adapter's own limits
    /// requested (the WebGPU defaults cap a binding at 128 MiB; the DiT's
    /// fused QKV buffer at 480p is ~600 MiB in f32).
    pub async fn new_async() -> Result<Self, InitError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("no WebGPU adapter: {e}")))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lightx2v"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("request_device: {e}")))?;
        let fault = Arc::new(Mutex::new(None::<String>));
        #[cfg(target_arch = "wasm32")]
        {
            let f = fault.clone();
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                let mut f = f.lock().unwrap();
                if f.is_none() {
                    *f = Some(format!("WebGPU error: {e}"));
                }
            }));
            let f = fault.clone();
            device.set_device_lost_callback(move |reason, msg| {
                let mut f = f.lock().unwrap();
                if f.is_none() {
                    *f = Some(format!("the GPU device was lost ({reason:?}): {msg}"));
                }
            });
        }
        // K and V tiles of BK keys × 128 dims, f32, must fit shared memory.
        let wgs = device.limits().max_compute_workgroup_storage_size as usize;
        let attn_bk = if wgs >= 2 * 32 * 128 * 4 {
            32
        } else if wgs >= 2 * 16 * 128 * 4 {
            16
        } else {
            8
        };
        let dummy = GBuf {
            buf: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dummy"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })),
            len: 4,
        };
        Ok(Self {
            info: adapter.get_info(),
            device,
            queue,
            attn_bk,
            pipelines: Mutex::new(HashMap::new()),
            pending: Mutex::new(Pending { enc: None, n: 0 }),
            fault,
            dummy,
            allow_f16_weights: true,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, InitError> {
        pollster::block_on(Self::new_async())
    }

    /// `None` when no adapter answers (headless CI, no GPU driver): the
    /// caller skips the wgpu checks instead of failing.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_new() -> Option<Self> {
        Self::new().ok()
    }

    /// A CPU rasterizer (llvmpipe / lavapipe / SwiftShader / WARP). Fine for
    /// conformance checks, meaningless for timing: wgpu falls back to one
    /// silently when the GPU's Vulkan driver fails to load.
    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// The largest storage-buffer binding this device accepts.
    pub fn max_binding(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size
    }

    /// The first WebGPU error or device loss seen so far (browser only).
    pub fn fault(&self) -> Option<String> {
        self.fault.lock().unwrap().clone()
    }

    fn pipeline(&self, key: &str, src: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let mut map = self.pipelines.lock().unwrap();
        if let Some(p) = map.get(key) {
            return p.clone();
        }
        let body = src();
        let code = format!("{COMMON}\n{body}");
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let p = Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(key),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );
        map.insert(key.to_string(), p.clone());
        p
    }

    fn weight_src(src: &str, f16: bool) -> String {
        if f16 {
            src.replace("{W_TYPE}", "u32").replace(
                "{W_LOAD}",
                "let v = unpack2x16float(w[i >> 1u]); return select(v.x, v.y, (i & 1u) == 1u);",
            )
        } else {
            src.replace("{W_TYPE}", "f32")
                .replace("{W_LOAD}", "return w[i];")
        }
    }

    /// A GEMM shader reading int8 codes packed 4 per word, sign-extended and
    /// scaled per group (`p._p` columns per scale, binding 5).
    fn q8_src(src: &str) -> String {
        src.replace("{W_TYPE}", "u32")
            .replace(
                "fn load_w(",
                "@group(0) @binding(5) var<storage, read> wscale: array<f32>;\nfn load_w(",
            )
            .replace(
                "{W_LOAD}",
                "let b = (w[i >> 2u] >> ((i & 3u) * 8u)) << 24u; \
                 return f32(bitcast<i32>(b) >> 24u) * wscale[i / p._p];",
            )
    }

    /// The 4-bit twin: 8 codes per word, `(q − zero) · scale` per group with
    /// (scale, zero) pairs at binding 5.
    fn q4_src(src: &str) -> String {
        src.replace("{W_TYPE}", "u32")
            .replace(
                "fn load_w(",
                "@group(0) @binding(5) var<storage, read> wsz: array<vec2<f32>>;\nfn load_w(",
            )
            .replace(
                "{W_LOAD}",
                "let c = f32((w[i >> 3u] >> ((i & 7u) * 4u)) & 15u); \
                 let g = wsz[i / p._p]; return (c - g.y) * g.x;",
            )
    }

    /// Record one dispatch. `params` is the uniform block as 32-bit words.
    fn dispatch(
        &self,
        pipe: &wgpu::ComputePipeline,
        params: &[u32],
        bufs: &[&GBuf],
        groups: (u32, u32, u32),
    ) {
        let mut words = params.to_vec();
        while !words.len().is_multiple_of(4) {
            words.push(0);
        }
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.buf.as_entire_binding(),
            });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pend = self.pending.lock().unwrap();
        let enc = pend
            .enc
            .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        pend.n += 1;
        if pend.n >= BATCH {
            let enc = pend.enc.take().unwrap();
            pend.n = 0;
            drop(pend);
            self.queue.submit([enc.finish()]);
        }
    }

    fn flush(&self) {
        let mut pend = self.pending.lock().unwrap();
        let enc = pend.enc.take();
        pend.n = 0;
        drop(pend);
        if let Some(enc) = enc {
            self.queue.submit([enc.finish()]);
        }
    }

    /// `n` workgroups as a (x, y) grid within the per-dimension limit.
    fn grid(n: usize) -> (u32, u32, u32) {
        let n = n.max(1) as u32;
        if n <= MAX_DIM {
            (n, 1, 1)
        } else {
            let y = n.div_ceil(MAX_DIM);
            (n.div_ceil(y), y, 1)
        }
    }

    fn upload_u32(&self, data: &[u32]) -> GBuf {
        let data = if data.is_empty() { &[0u32][..] } else { data };
        GBuf {
            buf: Arc::new(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
            ),
            len: data.len(),
        }
    }

    /// The elementwise kernel. `dims` fills the struct after `op`
    /// (`dim, gate, has_aux, c, t, hh, ww, pp`); `extra` fills `a0..a3`.
    #[allow(clippy::too_many_arguments)]
    fn elementwise(
        &self,
        op: u32,
        x: &GBuf,
        y: Option<&GBuf>,
        aux: Option<&GBuf>,
        n: usize,
        dims: [u32; 8],
        s: [f32; 2],
        extra: [u32; 4],
    ) {
        let pipe = self.pipeline("elementwise", || ELEM.to_string());
        let mut p = vec![n as u32, op];
        p.extend_from_slice(&dims);
        p.push(s[0].to_bits());
        p.push(s[1].to_bits());
        p.extend_from_slice(&extra);
        self.dispatch(
            &pipe,
            &p,
            &[x, y.unwrap_or(&self.dummy), aux.unwrap_or(&self.dummy)],
            Self::grid(n.div_ceil(256)),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        h: &GBuf,
        a: Option<&GBuf>,
        rows: usize,
        dim: usize,
        mods: &GBuf,
        gate: Option<usize>,
        norm: NormKind<'_, GBuf>,
        eps: f32,
        out: &GBuf,
    ) {
        let pipe = self.pipeline("norm", || NORM.to_string());
        let (affine, scale, shift, w, b) = match norm {
            NormKind::Affine { w, b } => (1, 0, 0, w, b),
            NormKind::Mod { scale, shift } => {
                (0, scale as u32, shift as u32, &self.dummy, &self.dummy)
            }
        };
        let p = [
            rows as u32,
            dim as u32,
            a.is_some() as u32,
            gate.unwrap_or(0) as u32,
            gate.is_some() as u32,
            affine,
            scale,
            shift,
            eps.to_bits(),
        ];
        self.dispatch(
            &pipe,
            &p,
            &[h, a.unwrap_or(&self.dummy), mods, w, b, out],
            Self::grid(rows),
        );
    }

    /// Flash kernel dispatch: mode 0 dense, mode 1 VSA fine stage, mode 2
    /// causal masked.
    #[allow(clippy::too_many_arguments)]
    fn flash(
        &self,
        mode: u32,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        out: &GBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        scale: f32,
        vsa: Option<(&GSparse, &GBuf)>,
        mask: Option<&GBuf>,
    ) {
        let bk = self.attn_bk;
        let pipe = self.pipeline(&format!("flash_bk{bk}"), || {
            FLASH
                .replace("{BK_D}", &(bk * 128).to_string())
                .replace("{BK}", &bk.to_string())
        });
        let (nb, topk) = vsa.map_or((0, 0), |(s, _)| (s.nb, s.topk));
        let p = [
            heads as u32,
            sq as u32,
            sk as u32,
            d as u32,
            scale.to_bits(),
            mode,
            nb as u32,
            topk as u32,
        ];
        let blocks = if mode == 1 { nb } else { sq.div_ceil(64) };
        let (slot, sel, coarse, gate) = match vsa {
            Some((s, g)) => (&s.slot, &s.sel, &s.coarse, g),
            None => (&self.dummy, &self.dummy, &self.dummy, &self.dummy),
        };
        self.dispatch(
            &pipe,
            &p,
            &[
                q,
                k,
                v,
                out,
                slot,
                sel,
                coarse,
                gate,
                mask.unwrap_or(&self.dummy),
            ],
            (blocks as u32, heads as u32, 1),
        );
    }

    // ---- KV cache helpers -------------------------------------------------

    /// Grow a cache's device buffers so `need` rows fit.
    fn cache_grow(&self, cache: &mut GKvCache, need: usize) {
        if need <= cache.capacity {
            return;
        }
        let new_cap = need.max(cache.capacity * 2).max(1);
        let width = cache.width();
        let nk = self.alloc(new_cap * width);
        let nv = self.alloc(new_cap * width);
        let len = cache.rows();
        if len > 0 {
            let bytes = len * width;
            self.copy(&cache.k, 0, &nk, 0, bytes);
            self.copy(&cache.v, 0, &nv, 0, bytes);
        }
        cache.k = nk;
        cache.v = nv;
        cache.capacity = new_cap;
    }

    /// Write `rows` K/V rows (`[heads, rows, head_dim]`) into the cache at
    /// `row_off` (elementwise op 5, `kv_pack`).
    fn cache_pack(&self, cache: &GKvCache, k: &GBuf, v: &GBuf, rows: usize, row_off: usize) {
        let n = rows * cache.width();
        let extra = [
            cache.heads as u32,
            cache.head_dim as u32,
            rows as u32,
            row_off as u32,
        ];
        self.elementwise(5, &cache.k, Some(k), None, n, [0u32; 8], [0.0; 2], extra);
        self.elementwise(5, &cache.v, Some(v), None, n, [0u32; 8], [0.0; 2], extra);
    }

    /// Fold one frame into an EMA slot: `ema <- (1-alpha)*ema + alpha*frame`,
    /// on both K and V. Offsets are in elements; the evicted rows go through
    /// a temp buffer so no buffer is bound read-write and read at once.
    fn cache_ema_fold(
        &self,
        cache: &GKvCache,
        ema_off: usize,
        ev_off: usize,
        n: usize,
        alpha: f32,
    ) {
        for buf in [&cache.k, &cache.v] {
            let tmp = self.alloc(n);
            self.copy(buf, ev_off, &tmp, 0, n);
            self.elementwise(
                7,
                buf,
                Some(&tmp),
                None,
                n,
                [0u32; 8],
                [alpha, 0.0],
                [ema_off as u32, 0, 0, 0],
            );
        }
    }

    /// Rebuild the cache's K/V through a row map (`kv_compact`): out row `r`
    /// copies old row `map[r]`, or zeros when the map holds `PAD`.
    fn cache_compact(&self, cache: &mut GKvCache, map: &[u32]) {
        let width = cache.width();
        let out_rows = map.len();
        let map_buf = self.upload_u32(map);
        let nk = self.alloc(out_rows * width);
        let nv = self.alloc(out_rows * width);
        let pipe = self.pipeline("kv_compact", || KV_COMPACT.to_string());
        let p = [(out_rows * width) as u32, width as u32, out_rows as u32];
        self.dispatch(
            &pipe,
            &p,
            &[&nk, &cache.k, &map_buf],
            Self::grid((out_rows * width).div_ceil(256)),
        );
        self.dispatch(
            &pipe,
            &p,
            &[&nv, &cache.v, &map_buf],
            Self::grid((out_rows * width).div_ceil(256)),
        );
        cache.k = nk;
        cache.v = nv;
    }
}

impl Backend for WgpuBackend {
    type Buf = GBuf;
    type Weight = GWeight;
    type KvCache = GKvCache;
    type Sparse = GSparse;

    fn name(&self) -> String {
        format!("wgpu: {}", self.info.name)
    }

    // ---- memory -----------------------------------------------------------

    fn alloc(&self, len: usize) -> GBuf {
        // `create_buffer` zero-initialises; nothing to clear.
        GBuf {
            buf: Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (len * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            len,
        }
    }

    fn upload(&self, data: &[f32]) -> GBuf {
        let data = if data.is_empty() { &[0.0f32][..] } else { data };
        GBuf {
            buf: Arc::new(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
            ),
            len: data.len(),
        }
    }

    fn write(&self, b: &GBuf, data: &[f32]) {
        self.flush();
        self.queue
            .write_buffer(&b.buf, 0, bytemuck::cast_slice(data));
    }

    fn download(&self, b: &GBuf) -> Vec<f32> {
        if b.len == 0 {
            return Vec::new();
        }
        self.flush();
        let bytes = b.len * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&b.buf, 0, &staging, 0, bytes as u64);
            self.queue.submit([enc.finish()]);
        }
        let slice = staging.slice(..bytes as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let view = slice.get_mapped_range().unwrap();
        let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }

    fn copy(&self, src: &GBuf, src_off: usize, dst: &GBuf, dst_off: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pend = self.pending.lock().unwrap();
        let enc = pend
            .enc
            .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        enc.copy_buffer_to_buffer(
            &src.buf,
            (src_off * 4) as u64,
            &dst.buf,
            (dst_off * 4) as u64,
            (len * 4) as u64,
        );
        pend.n += 1;
        if pend.n >= BATCH {
            let enc = pend.enc.take().unwrap();
            pend.n = 0;
            drop(pend);
            self.queue.submit([enc.finish()]);
        }
    }

    fn buf_len(&self, b: &GBuf) -> usize {
        b.len
    }

    fn sync(&self) {
        self.flush();
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    // ---- weights ----------------------------------------------------------

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> GWeight {
        assert_eq!(data.len(), rows * cols);
        // f16 pairs pack in u32 only when every value fits f16.
        let f16 = self.allow_f16_weights
            && cols.is_multiple_of(2)
            && data.iter().all(|&v| v.is_finite() && v.abs() <= 65000.0);
        let buf = if f16 {
            let packed: Vec<u32> = data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| {
                    let a = half::f16::from_f32(c[0]).to_bits() as u32;
                    let b = half::f16::from_f32(c[1]).to_bits() as u32;
                    a | (b << 16)
                })
                .collect();
            self.upload_u32(&packed)
        } else {
            self.upload(data)
        };
        GWeight {
            buf,
            rows,
            cols,
            f16,
            quant: None,
        }
    }

    fn weight_q8(&self, q: &Q8) -> GWeight {
        // 4 codes per word; mask each i8 to its low byte (a bare `as u32`
        // sign-extends and corrupts the neighbouring bytes).
        let mut words = Vec::with_capacity(q.q.len().div_ceil(4));
        for c in q.q.chunks(4) {
            let mut w = 0u32;
            for (i, &code) in c.iter().enumerate() {
                w |= (code as u8 as u32) << (8 * i);
            }
            words.push(w);
        }
        let rows = q.rows;
        let cols = q.cols;
        let group = q.group;
        let scales = self.upload(&q.scales);
        let buf = self.upload_u32(&words);
        GWeight {
            buf,
            rows,
            cols,
            f16: false,
            quant: Some((Quant::Q8, scales, group)),
        }
    }

    fn weight_q4(&self, q: &Q4) -> GWeight {
        let packed = q.packed();
        let mut words = Vec::with_capacity(packed.len().div_ceil(4));
        for c in packed.chunks(4) {
            let mut w = 0u32;
            for (i, &byte) in c.iter().enumerate() {
                w |= (byte as u32) << (8 * i);
            }
            words.push(w);
        }
        let rows = q.rows;
        let cols = q.cols;
        let group = q.group;
        let mut sz = Vec::with_capacity(q.scales.len() * 2);
        for i in 0..q.scales.len() {
            sz.push(q.scales[i]);
            sz.push(q.zeros[i]);
        }
        let scales = self.upload(&sz);
        let buf = self.upload_u32(&words);
        GWeight {
            buf,
            rows,
            cols,
            f16: false,
            quant: Some((Quant::Q4, scales, group)),
        }
    }

    fn weight_fp8(&self, q: &Fp8E4M3) -> GWeight {
        // FP8 is never stored on the device: dequantize to f32 and use the
        // normal f16-packed path (the CPU reference does the same thing).
        self.weight(&q.dequantize(), q.rows, q.cols)
    }

    // ---- GEMM / norm / elementwise -----------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: &GBuf,
        x_row: usize,
        rows: usize,
        w: &GWeight,
        bias: Option<&GBuf>,
        act: Act,
        y: &GBuf,
        y_row: usize,
    ) {
        let p = [
            rows as u32,
            w.rows as u32,
            w.cols as u32,
            x_row as u32,
            y_row as u32,
            bias.is_some() as u32,
            match act {
                Act::None => 0,
                Act::GeluTanh => 1,
            },
            w.quant.as_ref().map_or(0, |q| q.2) as u32,
        ];
        let pipe = match &w.quant {
            Some((Quant::Q8, _, _)) => self.pipeline("linear_q8", || Self::q8_src(LINEAR)),
            Some((Quant::Q4, _, _)) => self.pipeline("linear_q4", || Self::q4_src(LINEAR)),
            None => self.pipeline(&format!("linear_f16_{}", w.f16), || {
                Self::weight_src(LINEAR, w.f16)
            }),
        };
        let mut bufs: Vec<&GBuf> = vec![x, &w.buf, bias.unwrap_or(&self.dummy), y];
        if let Some((_, scales, _)) = &w.quant {
            bufs.push(scales);
        }
        self.dispatch(
            &pipe,
            &p,
            &bufs,
            (w.rows.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1),
        );
    }

    fn layer_norm(
        &self,
        x: &GBuf,
        rows: usize,
        dim: usize,
        norm: NormKind<'_, GBuf>,
        mods: &GBuf,
        eps: f32,
        out: &GBuf,
    ) {
        self.norm(x, None, rows, dim, mods, None, norm, eps, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn residual_norm(
        &self,
        h: &GBuf,
        a: &GBuf,
        rows: usize,
        dim: usize,
        mods: &GBuf,
        gate: Option<usize>,
        norm: NormKind<'_, GBuf>,
        eps: f32,
        normed: &GBuf,
    ) {
        // The norm shader's mode 1 does the residual add in place on `h`
        // and then normalises the updated `h` into `normed` — exactly the
        // CPU's "the norm reads the unrounded sum".
        self.norm(h, Some(a), rows, dim, mods, gate, norm, eps, normed);
    }

    fn gate_residual(&self, h: &GBuf, a: &GBuf, rows: usize, dim: usize, mods: &GBuf, gate: usize) {
        let n = rows * dim;
        self.elementwise(
            0,
            h,
            Some(a),
            Some(mods),
            n,
            [dim as u32, gate as u32, 0, 0, 0, 0, 0, 0],
            [0.0; 2],
            [0u32; 4],
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn head_split(
        &self,
        src: &GBuf,
        rows: usize,
        src_stride: usize,
        col_off: usize,
        heads: usize,
        head_dim: usize,
        rms_w: Option<&GBuf>,
        rope: Option<(&GBuf, &GBuf)>,
        eps: f32,
        out: &GBuf,
    ) {
        let pipe = self.pipeline("head_split", || HEADS.to_string());
        let (cos, sin) = rope.map_or((&self.dummy, &self.dummy), |(c, s)| (c, s));
        let p = [
            rows as u32,
            src_stride as u32,
            col_off as u32,
            heads as u32,
            head_dim as u32,
            rms_w.is_some() as u32,
            rope.is_some() as u32,
            0,
            eps.to_bits(),
            0,
            0,
            0,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[src, rms_w.unwrap_or(&self.dummy), cos, sin, out],
            Self::grid(rows),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        out: &GBuf,
    ) {
        assert!(head_dim <= 128, "attention head_dim {head_dim} exceeds 128");
        let scale = 1.0 / (head_dim as f32).sqrt();
        self.flash(0, q, k, v, out, heads, sq, sk, head_dim, scale, None, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn causal_attention(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        mask: Option<&[bool]>,
        out: &GBuf,
    ) {
        assert!(
            head_dim <= 128,
            "causal_attention head_dim {head_dim} exceeds 128"
        );
        let scale = 1.0 / (head_dim as f32).sqrt();
        match mask {
            None => {
                self.flash(0, q, k, v, out, heads, sq, sk, head_dim, scale, None, None);
            }
            Some(m) => {
                assert_eq!(m.len(), sq * sk);
                let words: Vec<u32> = m.iter().map(|&b| b as u32).collect();
                let mask_buf = self.upload_u32(&words);
                self.flash(
                    2,
                    q,
                    k,
                    v,
                    out,
                    heads,
                    sq,
                    sk,
                    head_dim,
                    scale,
                    None,
                    Some(&mask_buf),
                );
            }
        }
    }

    fn sparse_supported(&self, head_dim: usize) -> bool {
        head_dim <= 128
    }

    fn sparse_prepare(&self, spec: &SparseSpec) -> GSparse {
        let nb = spec.plan.num_tiles();
        assert!(
            nb <= 2048,
            "sparse grid has {nb} tiles; VSA tables cap at 2048"
        );
        let heads = spec.heads;
        let head_dim = spec.head_dim;
        GSparse {
            slot: self.upload_u32(&spec.plan.slot_src),
            bsz: self.upload_u32(&spec.plan.block_sizes),
            pooled: self.alloc(3 * heads * nb * head_dim),
            coarse: self.alloc(heads * nb * head_dim),
            sel: self.upload_u32(&vec![0u32; heads * nb * spec.topk]),
            nb,
            topk: spec.topk,
            heads,
            d: head_dim,
            seq: spec.plan.seq,
            scale: spec.scale,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_attention(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        gate: &GBuf,
        sparse: &GSparse,
        out: &GBuf,
    ) {
        let heads = sparse.heads;
        let nb = sparse.nb;
        let d = sparse.d;
        // Coarse mean-pool of Q/K/V over each tile.
        let pool = self.pipeline("vsa_pool", || VSA_POOL.to_string());
        let p = [heads as u32, sparse.seq as u32, d as u32, nb as u32];
        self.dispatch(
            &pool,
            &p,
            &[q, k, v, &sparse.slot, &sparse.bsz, &sparse.pooled],
            (nb as u32, heads as u32, 1),
        );
        // Coarse attention + top-k tile selection.
        let coarse_pipe = self.pipeline("vsa_coarse", || VSA_COARSE.to_string());
        let p = [
            heads as u32,
            d as u32,
            nb as u32,
            sparse.topk as u32,
            sparse.scale.to_bits(),
        ];
        self.dispatch(
            &coarse_pipe,
            &p,
            &[&sparse.pooled, &sparse.coarse, &sparse.sel],
            (nb as u32, heads as u32, 1),
        );
        // Fine attention over the selected tiles, plus coarse·gate.
        self.flash(
            1,
            q,
            k,
            v,
            out,
            heads,
            sparse.seq,
            sparse.seq,
            d,
            sparse.scale,
            Some((sparse, gate)),
            None,
        );
    }

    // ---- KV cache -----------------------------------------------------------

    fn kv_cache_new(&self, heads: usize, head_dim: usize, capacity_tokens: usize) -> GKvCache {
        let width = heads * head_dim;
        GKvCache {
            heads,
            head_dim,
            capacity: capacity_tokens,
            k: self.alloc(capacity_tokens * width),
            v: self.alloc(capacity_tokens * width),
            temporal: Vec::with_capacity(capacity_tokens),
            kinds: Vec::with_capacity(capacity_tokens),
        }
    }

    fn kv_cache_append(
        &self,
        cache: &mut GKvCache,
        k: &GBuf,
        v: &GBuf,
        rows: usize,
        temporal: &[usize],
    ) {
        assert_eq!(temporal.len(), rows);
        self.cache_grow(cache, cache.rows() + rows);
        self.cache_pack(cache, k, v, rows, cache.rows());
        cache.temporal.extend_from_slice(temporal);
        cache.kinds.resize(cache.rows(), SlotKind::New);
    }

    fn kv_cache_put(
        &self,
        cache: &mut GKvCache,
        k: &GBuf,
        v: &GBuf,
        rows: usize,
        temporal: &[usize],
    ) {
        assert_eq!(temporal.len(), rows);
        // First row whose temporal matches and whose kind is not an EMA slot.
        let mut pos = None;
        if let Some(&t0) = temporal.first() {
            for i in 0..cache.rows() {
                if cache.temporal[i] == t0
                    && cache.kinds[i] != SlotKind::LongEma
                    && cache.kinds[i] != SlotKind::ShortEma
                {
                    pos = Some(i);
                    break;
                }
            }
        }
        if let Some(i) = pos {
            let ok =
                rows <= cache.rows() - i && (0..rows).all(|r| cache.temporal[i + r] == temporal[r]);
            if ok {
                self.cache_pack(cache, k, v, rows, i);
                return;
            }
        }
        self.kv_cache_append(cache, k, v, rows, temporal);
    }

    fn kv_cache_len(&self, cache: &GKvCache) -> usize {
        cache.rows()
    }

    fn kv_cache_temporal(&self, cache: &GKvCache) -> Vec<usize> {
        cache.temporal.clone()
    }

    fn kv_cache_kv(&self, cache: &GKvCache) -> (GBuf, GBuf) {
        let rows = cache.rows();
        let n = rows * cache.width();
        let k = self.alloc(n);
        let v = self.alloc(n);
        if n > 0 {
            let extra = [cache.heads as u32, cache.head_dim as u32, rows as u32, 0u32];
            self.elementwise(6, &k, Some(&cache.k), None, n, [0u32; 8], [0.0; 2], extra);
            self.elementwise(6, &v, Some(&cache.v), None, n, [0u32; 8], [0.0; 2], extra);
        }
        (k, v)
    }

    /// The same metadata simulation as the CPU oracle, executed on the
    /// device: EMA slots are seeded with zero rows (`kv_compact` with `PAD`
    /// entries), evictions fold into the EMA slots (`ema_fold`) and then
    /// drop the frame through the row map, and the sinks are reclassified.
    fn kv_cache_memrope_update(
        &self,
        cache: &mut GKvCache,
        cfg: &MemRopeConfig,
        frame_tokens: usize,
    ) {
        let rows = cache.rows();
        if rows == 0 {
            return;
        }
        let frame_rows = |cache: &GKvCache| cache.rows() / frame_tokens;
        let frame_kinds = |cache: &GKvCache| -> Vec<SlotKind> {
            (0..frame_rows(cache))
                .map(|f| cache.kinds[f * frame_tokens])
                .collect()
        };
        // Seed the EMA slots, if missing, right after the sinks — with the
        // same order as the CPU: splicing ShortEma then LongEma at the same
        // position leaves [LongEma, ShortEma].
        let mut kinds = frame_kinds(cache);
        if !kinds.contains(&SlotKind::LongEma) && kinds.len() > cfg.sink_frames {
            let at = cfg.sink_frames * frame_tokens;
            let t = cache.temporal[at.min(cache.rows() - 1)];
            let mut map: Vec<u32> = (0..cache.rows() as u32).collect();
            map.splice(at..at, [PAD, PAD]);
            self.cache_compact(cache, &map);
            cache.temporal.splice(at..at, [t, t]);
            cache
                .kinds
                .splice(at..at, [SlotKind::LongEma, SlotKind::ShortEma]);
            kinds = frame_kinds(cache);
        }
        // Graduate New frames beyond the new window to Recent, oldest first.
        let new_frames: Vec<usize> = kinds
            .iter()
            .enumerate()
            .filter(|(_, &k)| k == SlotKind::New)
            .map(|(i, _)| i)
            .collect();
        if new_frames.len() > cfg.new_frames() {
            let excess = new_frames.len() - cfg.new_frames();
            for f in new_frames.into_iter().take(excess) {
                for r in 0..frame_tokens {
                    cache.kinds[f * frame_tokens + r] = SlotKind::Recent;
                }
            }
            kinds = frame_kinds(cache);
        }
        // Evict until the frame budget holds.
        while frame_rows(cache) > cfg.cache_frames {
            let mut evict: Option<usize> = None;
            for (f, &kind) in kinds.iter().enumerate() {
                if kind == SlotKind::Recent {
                    evict = Some(f);
                    break;
                }
            }
            if evict.is_none() {
                for (f, &kind) in kinds.iter().enumerate() {
                    if kind == SlotKind::New {
                        evict = Some(f);
                        break;
                    }
                }
            }
            let Some(f) = evict else { break };
            let width = cache.width();
            let ev_off = f * frame_tokens * width;
            let n = frame_tokens * width;
            for kind in [SlotKind::LongEma, SlotKind::ShortEma] {
                let alpha = if kind == SlotKind::LongEma {
                    cfg.ema_alpha_long
                } else {
                    cfg.ema_alpha_short
                };
                let Some(slot) = kinds.iter().position(|k| *k == kind) else {
                    continue;
                };
                let ema_off = slot * frame_tokens * width;
                self.cache_ema_fold(cache, ema_off, ev_off, n, alpha);
            }
            let start = f * frame_tokens;
            let map: Vec<u32> = (0..cache.rows() as u32)
                .filter(|r| (*r as usize) < start || (*r as usize) >= start + frame_tokens)
                .collect();
            self.cache_compact(cache, &map);
            cache.temporal.drain(start..start + frame_tokens);
            cache.kinds.drain(start..start + frame_tokens);
            kinds = frame_kinds(cache);
        }
        // Reclassify the first `sink_frames` frames as sinks.
        for f in 0..cfg.sink_frames.min(kinds.len()) {
            for r in 0..frame_tokens {
                cache.kinds[f * frame_tokens + r] = SlotKind::Sink;
            }
        }
    }

    // ---- patchify / DMD / conv / tanh --------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn patchify(
        &self,
        latent: &GBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &GBuf,
    ) {
        let n = c * t * h * w;
        let dims = [0u32, 0, 0, c as u32, t as u32, h as u32, w as u32, p as u32];
        self.elementwise(3, out, Some(latent), None, n, dims, [0.0; 2], [0u32; 4]);
    }

    #[allow(clippy::too_many_arguments)]
    fn unpatchify(&self, x: &GBuf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &GBuf) {
        let n = c * t * h * w;
        let dims = [0u32, 0, 0, c as u32, t as u32, h as u32, w as u32, p as u32];
        self.elementwise(4, out, Some(x), None, n, dims, [0.0; 2], [0u32; 4]);
    }

    fn dmd_step(
        &self,
        x: &GBuf,
        v: &GBuf,
        noise: Option<&GBuf>,
        sigma_t: f32,
        sigma_next: f32,
        n: usize,
    ) {
        self.elementwise(
            1,
            x,
            Some(v),
            noise,
            n,
            [0u32, 0, noise.is_some() as u32, 0, 0, 0, 0, 0],
            [sigma_t, sigma_next],
            [0u32; 4],
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        a: &GBuf,
        b: Option<&GBuf>,
        shape: &ConvShape,
        w: &GWeight,
        bias: Option<&GBuf>,
        epilogue: ConvEpilogue<'_, GBuf>,
        out: &GBuf,
    ) {
        assert!(w.quant.is_none(), "conv2d does not take quantized weights");
        assert_eq!(w.rows, shape.cout);
        assert_eq!(w.cols, shape.k_len());
        let (oh, ow) = shape.out_hw();
        let mt = shape.frames * oh * ow;
        let (res, ep) = match epilogue {
            ConvEpilogue::None => (&self.dummy, 0u32),
            ConvEpilogue::Relu => (&self.dummy, 1u32),
            ConvEpilogue::AddRelu(r) => (r, 2u32),
            ConvEpilogue::Prelu(s) => (s, 3u32),
        };
        let pipe = self.pipeline(&format!("conv_f16_{}", w.f16), || {
            Self::weight_src(CONV, w.f16)
        });
        let p = [
            shape.frames as u32,
            shape.cin_a as u32,
            shape.cin_b as u32,
            shape.cout as u32,
            shape.h as u32,
            shape.w as u32,
            if shape.upsample { 2 } else { 1 },
            shape.k as u32,
            b.is_some() as u32,
            bias.is_some() as u32,
            ep,
            0,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[
                a,
                b.unwrap_or(&self.dummy),
                &w.buf,
                bias.unwrap_or(&self.dummy),
                res,
                out,
            ],
            (shape.cout.div_ceil(64) as u32, mt.div_ceil(64) as u32, 1),
        );
    }

    fn tanh_clamp(&self, x: &GBuf, n: usize, s: f32) {
        self.elementwise(2, x, None, None, n, [0u32; 8], [s, 0.0], [0u32; 4]);
    }
}
