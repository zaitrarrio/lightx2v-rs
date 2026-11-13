//! The LightX2V / History-Forcing model: causal Wan DiT, the 4-step DMD
//! sampler, the autoregressive chunk pipeline, the TAEHV decode stub, and
//! the conformance checker.
//!
//! The DiT mirrors fastwan-rs's `WanDit` block math (same modulation slots,
//! same fused QKV/KV linears, same host-side timestep conditioning) with one
//! structural change: self-attention is **causal over chunks** through a
//! per-layer MemRoPE KV cache, so a 252-latent-frame rollout is generated
//! 3 frames at a time instead of all at once.

pub mod check;
pub mod dit;
pub mod pipeline;
pub mod taehv;
pub mod timestep;

pub use dit::LightDit;
pub use pipeline::{CausalPipeline, Options, Timings};
pub use taehv::TaeHv;
