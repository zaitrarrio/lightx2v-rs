//! TAEHV decoder (madebyollin/taehv) — Phase 2 implements the decode for
//! real, on any [`Backend`], using only the existing backend ops.
//!
//! ```text
//! clamp tanh(x/3)·3 → conv C→n0, relu
//! 3 stages: 3×MemBlock(n_s) → upsample ×2 → TGrow(stride) → conv n_s→n_{s+1}
//! relu → conv n3→3·p² → pixel-shuffle p
//! MemBlock(x, past) = relu(conv(relu(conv(relu(conv(cat[x, past]))))) + x)
//! ```
//!
//! Three rewrites, each exact (ported from fastwan-model's TAEHV):
//! * **TGrow before the upsample.** TGrow is a 1×1 conv, and a pointwise op
//!   commutes with nearest upsampling, so it runs at ¼ of the pixels.
//! * **Upsample fused into the next conv.** The ×2 tensor is never stored;
//!   the conv reads its input through `(y/2, x/2)` addressing.
//! * **Concat fused into the MemBlock's first conv** (`cat[x, past]` is read
//!   from two buffers).
//!
//! Frames are decoded in latent chunks of 4; each MemBlock carries its last
//! input frame across a chunk boundary (the reference's sequential
//! `memory[i]`), so chunking does not change the output. The first
//! `frames_to_trim()` video frames are priming frames and are dropped.
//!
//! Conv weights are stored as plain backend weights. A checkpoint that
//! stores a conv weight quantized (`<key>.weight.q8/.q4/.fp8`) is
//! dequantized on the host at load: `conv2d` has no quantized path (its
//! shader dequantizes nothing), so the dequantized values are what the
//! backend stores — the same contract as the CPU reference.

use crate::pipeline::Video;
use lightx2v_core::{
    backend::{Backend, ConvEpilogue, ConvShape},
    config::TaeConfig,
    quant::Stored,
    weights::{missing, tae_manifest, WeightSource},
    Error, Result,
};
use std::collections::HashMap;

/// Latent frames per decode chunk.
const CHUNK_FRAMES: usize = 4;

struct ConvW<B: Backend> {
    w: B::Weight,
    b: Option<B::Buf>,
}

struct Stage<B: Backend> {
    mems: Vec<[ConvW<B>; 3]>,
    grow: ConvW<B>,
    conv: ConvW<B>,
}

pub struct TaeHv<B: Backend> {
    pub cfg: TaeConfig,
    first: ConvW<B>,
    stages: Vec<Stage<B>>,
    last: ConvW<B>,
}

/// Carried between chunks: one saved input frame per MemBlock, and how many
/// priming frames are still to be dropped.
pub struct DecodeState<B: Backend> {
    memory: Vec<Option<B::Buf>>,
    to_trim: usize,
    /// Intermediate buffers by length, reused by the next chunk (every chunk
    /// but the last has the same shapes). Device work is ordered, so a buffer
    /// handed back once its last reader is queued is safe to overwrite.
    pool: HashMap<usize, Vec<B::Buf>>,
}

/// One chunk's decoder output on the device: `[frames, 3·p², h, w]`, of
/// which the first `skip` frames are priming frames to drop.
pub struct Chunk<B: Backend> {
    pub rgb: B::Buf,
    pub frames: usize,
    pub skip: usize,
    pub h: usize,
    pub w: usize,
    pub p: usize,
}

impl<B: Backend> Chunk<B> {
    /// Kept frames as interleaved 8-bit RGB (`[frames, H, W, 3]`), from the
    /// downloaded `raw` buffer; pixel-shuffle, clamp and rounding in one pass,
    /// rows split across the available cores.
    pub fn rgb8(&self, raw: &[f32]) -> (usize, usize, usize, Vec<u8>) {
        let (p, h, w) = (self.p, self.h, self.w);
        let (oh, ow, kept, skip) = (h * p, w * p, self.frames - self.skip, self.skip);
        let mut out = vec![0u8; kept * oh * ow * 3];
        let row = ow * 3;
        let fill = |r0: usize, dst: &mut [u8]| {
            for (ri, line) in dst.chunks_mut(row).enumerate() {
                let (fo, y) = ((r0 + ri) / oh, (r0 + ri) % oh);
                let fi = fo + skip;
                for x in 0..ow {
                    for c in 0..3 {
                        let k = c * p * p + (y % p) * p + x % p;
                        let v = raw[((fi * 3 * p * p + k) * h + y / p) * w + x / p];
                        line[x * 3 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
            }
        };
        let rows = kept * oh;
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(16);
        if threads <= 1 || rows < 64 {
            fill(0, &mut out);
        } else {
            let per = rows.div_ceil(threads);
            std::thread::scope(|sc| {
                for (i, part) in out.chunks_mut(per * row).enumerate() {
                    let fill = &fill;
                    sc.spawn(move || fill(i * per, part));
                }
            });
        }
        (kept, oh, ow, out)
    }
}

/// Decoded frames of one chunk: `[frames, 3, H, W]`, values ≈ `[0, 1]`.
pub struct Frames {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

fn conv_w<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    key: &str,
    cout: usize,
    cin: usize,
    k: usize,
    bias: bool,
) -> Result<ConvW<B>> {
    let err = |m: String| Error::Weights(m);
    let w = match src.stored(key)? {
        // Quantized conv weights are dequantized on the host: conv2d has no
        // quantized path, so the backend stores plain values either way.
        Some(Stored::Q8(q)) => {
            if (q.rows, q.cols) != (cout, cin * k * k) {
                return Err(err(format!(
                    "`{key}` is stored {}x{}, expected {cout}x{}",
                    q.rows,
                    q.cols,
                    cin * k * k
                )));
            }
            be.weight(&q.dequantize(), cout, cin * k * k)
        }
        Some(Stored::Q4(q)) => {
            if (q.rows, q.cols) != (cout, cin * k * k) {
                return Err(err(format!(
                    "`{key}` is stored {}x{}, expected {cout}x{}",
                    q.rows,
                    q.cols,
                    cin * k * k
                )));
            }
            be.weight(&q.dequantize(), cout, cin * k * k)
        }
        Some(Stored::Fp8(q)) => {
            if (q.rows, q.cols) != (cout, cin * k * k) {
                return Err(err(format!(
                    "`{key}` is stored {}x{}, expected {cout}x{}",
                    q.rows,
                    q.cols,
                    cin * k * k
                )));
            }
            be.weight(&q.dequantize(), cout, cin * k * k)
        }
        None => be.weight(
            &src.get(&format!("{key}.weight"), &[cout, cin, k, k])?,
            cout,
            cin * k * k,
        ),
    };
    Ok(ConvW {
        w,
        b: if bias {
            Some(be.upload(&src.get(&format!("{key}.bias"), &[cout])?))
        } else {
            None
        },
    })
}

impl<B: Backend> TaeHv<B> {
    pub fn load(be: &B, cfg: TaeConfig, src: &dyn WeightSource) -> Result<Self> {
        let manifest = tae_manifest(&cfg);
        let miss = missing(src, &manifest);
        if !miss.is_empty() {
            return Err(Error::Weights(format!(
                "TAEHV checkpoint is missing {} tensor(s), first: {}",
                miss.len(),
                miss[..miss.len().min(3)].join(", ")
            )));
        }
        let nf = cfg.widths;
        let mut stages = Vec::new();
        for s in 0..3 {
            let base = 3 + 6 * s;
            let mems = (0..cfg.mem_blocks)
                .map(|m| {
                    let p = format!("decoder.{}", base + m);
                    Ok([
                        conv_w(be, src, &format!("{p}.conv.0"), nf[s], 2 * nf[s], 3, true)?,
                        conv_w(be, src, &format!("{p}.conv.2"), nf[s], nf[s], 3, true)?,
                        conv_w(be, src, &format!("{p}.conv.4"), nf[s], nf[s], 3, true)?,
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push(Stage {
                mems,
                grow: conv_w(
                    be,
                    src,
                    &format!("decoder.{}.conv", base + 4),
                    nf[s] * cfg.tgrow[s],
                    nf[s],
                    1,
                    false,
                )?,
                conv: conv_w(
                    be,
                    src,
                    &format!("decoder.{}", base + 5),
                    nf[s + 1],
                    nf[s],
                    3,
                    false,
                )?,
            });
        }
        Ok(Self {
            first: conv_w(be, src, "decoder.1", nf[0], cfg.latent_channels, 3, true)?,
            last: conv_w(be, src, "decoder.22", cfg.out_channels(), nf[3], 3, true)?,
            stages,
            cfg,
        })
    }

    pub fn state(&self) -> DecodeState<B> {
        DecodeState {
            memory: (0..3 * self.cfg.mem_blocks).map(|_| None).collect(),
            to_trim: self.cfg.frames_to_trim(),
            pool: Default::default(),
        }
    }

    /// Decode `frames` latent frames given frame-major (`[frames, C, h, w]`).
    pub fn decode_chunk(
        &self,
        be: &B,
        st: &mut DecodeState<B>,
        z: &[f32],
        frames: usize,
        h: usize,
        w: usize,
    ) -> Frames {
        let c = self.launch_chunk(be, st, z, frames, h, w);
        let raw = be.download(&c.rgb);
        let (oh, ow) = (c.h * c.p, c.w * c.p);
        let mut data = vec![0.0f32; (c.frames - c.skip) * 3 * oh * ow];
        for fi in c.skip..c.frames {
            for ch in 0..3 {
                for y in 0..oh {
                    for xx in 0..ow {
                        let k = ch * c.p * c.p + (y % c.p) * c.p + xx % c.p;
                        data[(((fi - c.skip) * 3 + ch) * oh + y) * ow + xx] =
                            raw[((fi * 3 * c.p * c.p + k) * c.h + y / c.p) * c.w + xx / c.p];
                    }
                }
            }
        }
        Frames {
            frames: c.frames - c.skip,
            height: oh,
            width: ow,
            data,
        }
    }

    /// Queue one chunk's decode on the device and return without waiting:
    /// [`Chunk::rgb8`] after [`Backend::download`] turns it into frames. The
    /// pipeline launches chunk i+1 before converting chunk i, so the host
    /// work overlaps the GPU's.
    pub fn launch_chunk(
        &self,
        be: &B,
        st: &mut DecodeState<B>,
        z: &[f32],
        frames: usize,
        h: usize,
        w: usize,
    ) -> Chunk<B> {
        let cfg = &self.cfg;
        let nf = cfg.widths;
        assert_eq!(z.len(), frames * cfg.latent_channels * h * w);
        let (mut f, mut h, mut w) = (frames, h, w);
        let x0 = be.upload(z);
        be.tanh_clamp(&x0, z.len(), 3.0);
        let shape = |f, cin_a, cin_b, cout, h, w, upsample, k| ConvShape {
            frames: f,
            cin_a,
            cin_b,
            cout,
            h,
            w,
            upsample,
            k,
        };
        // Pooled buffers: contents are stale, so only for outputs that are
        // written in full (every conv output, `keep`, and `past` once a
        // previous chunk exists).
        let pool = std::cell::RefCell::new(std::mem::take(&mut st.pool));
        let take = |n: usize| {
            let reused = pool.borrow_mut().get_mut(&n).and_then(Vec::pop);
            reused.unwrap_or_else(|| be.alloc(n))
        };
        let give = |b: B::Buf| pool.borrow_mut().entry(be.buf_len(&b)).or_default().push(b);
        let conv = |x: &B::Buf,
                    b: Option<&B::Buf>,
                    s: ConvShape,
                    cw: &ConvW<B>,
                    ep: ConvEpilogue<'_, B::Buf>| {
            let out = take(s.out_len());
            be.conv2d(x, b, &s, &cw.w, cw.b.as_ref(), ep, &out);
            out
        };
        let mut x = conv(
            &x0,
            None,
            shape(f, cfg.latent_channels, 0, nf[0], h, w, false, 3),
            &self.first,
            ConvEpilogue::Relu,
        );
        for (si, stage) in self.stages.iter().enumerate() {
            let n = nf[si];
            for (mi, [c0, c2, c4]) in stage.mems.iter().enumerate() {
                let fsz = n * h * w;
                let slot = &mut st.memory[si * cfg.mem_blocks + mi];
                // The first chunk's past frame is zero: a fresh buffer.
                let past = if slot.is_some() {
                    take(f * fsz)
                } else {
                    be.alloc(f * fsz)
                };
                if let Some(prev) = slot.as_ref() {
                    be.copy(prev, 0, &past, 0, fsz);
                }
                if f > 1 {
                    be.copy(&x, 0, &past, fsz, (f - 1) * fsz);
                }
                let keep = take(fsz);
                be.copy(&x, (f - 1) * fsz, &keep, 0, fsz);
                if let Some(old) = slot.replace(keep) {
                    give(old);
                }
                let t1 = conv(
                    &x,
                    Some(&past),
                    shape(f, n, n, n, h, w, false, 3),
                    c0,
                    ConvEpilogue::Relu,
                );
                give(past);
                let t2 = conv(
                    &t1,
                    None,
                    shape(f, n, 0, n, h, w, false, 3),
                    c2,
                    ConvEpilogue::Relu,
                );
                give(t1);
                let nx = conv(
                    &t2,
                    None,
                    shape(f, n, 0, n, h, w, false, 3),
                    c4,
                    ConvEpilogue::AddRelu(&x),
                );
                give(t2);
                give(std::mem::replace(&mut x, nx));
            }
            let stride = cfg.tgrow[si];
            // [f, n·stride, h, w] is bit-for-bit [f·stride, n, h, w].
            let g = conv(
                &x,
                None,
                shape(f, n, 0, n * stride, h, w, false, 1),
                &stage.grow,
                ConvEpilogue::None,
            );
            f *= stride;
            let ep = if si == 2 {
                ConvEpilogue::Relu
            } else {
                ConvEpilogue::None
            };
            give(std::mem::replace(
                &mut x,
                conv(
                    &g,
                    None,
                    shape(f, n, 0, nf[si + 1], h, w, true, 3),
                    &stage.conv,
                    ep,
                ),
            ));
            give(g);
            h *= 2;
            w *= 2;
        }
        let rgb = conv(
            &x,
            None,
            shape(f, nf[3], 0, cfg.out_channels(), h, w, false, 3),
            &self.last,
            ConvEpilogue::None,
        );
        give(x);
        st.pool = pool.into_inner();
        let skip = st.to_trim.min(f);
        st.to_trim -= skip;
        Chunk {
            rgb,
            frames: f,
            skip,
            h,
            w,
            p: cfg.patch,
        }
    }

    /// Decode a denoised latent `[C, T, H, W]` to RGB frames.
    /// `latent_frames` is `T`; `height`/`width` are the VIDEO dimensions
    /// (the latent spatial size is video / `spatial_scale()`). They are
    /// explicit — not read from a [`GenerationConfig`] — because the toy
    /// runs denoise under the demo profile while their video is 64x64.
    pub fn decode(
        &self,
        be: &B,
        latent: &[f32],
        latent_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Video> {
        let cfg = &self.cfg;
        let (lh, lw) = (height / cfg.spatial_scale(), width / cfg.spatial_scale());
        let want = cfg.latent_channels * latent_frames * lh * lw;
        if latent.len() != want {
            return Err(Error::Shape(format!(
                "TAEHV latent is {} elements, expected {} ({}x{}x{}x{})",
                latent.len(),
                want,
                cfg.latent_channels,
                latent_frames,
                lh,
                lw
            )));
        }
        let mut st = self.state();
        let mut rgb = Vec::new();
        let mut frames = 0usize;
        let (mut oh, mut ow) = (0usize, 0usize);
        let mut t0 = 0;
        while t0 < latent_frames {
            let f = CHUNK_FRAMES.min(latent_frames - t0);
            // Transpose this chunk [C, f, H, W] -> frame-major [f, C, H, W].
            let mut z = vec![0.0f32; f * cfg.latent_channels * lh * lw];
            for fi in 0..f {
                for c in 0..cfg.latent_channels {
                    for y in 0..lh {
                        for x in 0..lw {
                            z[((fi * cfg.latent_channels + c) * lh + y) * lw + x] =
                                latent[((c * latent_frames + t0 + fi) * lh + y) * lw + x];
                        }
                    }
                }
            }
            let chunk = self.launch_chunk(be, &mut st, &z, f, lh, lw);
            let raw = be.download(&chunk.rgb);
            let (kept, h, w, bytes) = chunk.rgb8(&raw);
            (oh, ow) = (h, w);
            frames += kept;
            rgb.extend_from_slice(&bytes);
            t0 += f;
        }
        Ok(Video {
            frames,
            width: ow,
            height: oh,
            rgb,
        })
    }

    /// Output frames for a latent count (`4T − 3`).
    pub fn out_frames(&self, latent_frames: usize) -> usize {
        latent_frames * self.cfg.t_upscale() - self.cfg.frames_to_trim()
    }
}
