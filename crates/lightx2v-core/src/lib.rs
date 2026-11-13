//! Core types for lightx2v-rs: config, geometry, schedule, RoPE, MemRoPE,
//! history-contrast, quantization, weights, the fused `Backend` trait, and
//! the memory plan.
//!
//! Lineage: this crate mirrors the layout and error discipline of
//! `fastwan-core` (zaitrarrio/fastwan-rs) — a fused op vocabulary, an f32
//! boundary, missing-key-is-an-error weights — reimplemented here with
//! LightX2V / History-Forcing names and semantics. No fastwan code is copied.

pub mod backend;
pub mod config;
pub mod export;
pub mod geometry;
pub mod history;
pub mod memrope;
pub mod plan;
pub mod quant;
pub mod rng;
pub mod rope;
pub mod schedule;
pub mod sparse;
pub mod weights;

use thiserror::Error;

/// Every error in the workspace. A missing weight key or a shape mismatch is
/// an error, never a default: a silently-defaulted weight streams gray haze
/// instead of failing.
#[derive(Debug, Error)]
pub enum Error {
    #[error("weights: {0}")]
    Weights(String),
    #[error("config: {0}")]
    Config(String),
    #[error("shape: {0}")]
    Shape(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Convenience: a config error.
pub fn cfg_err(msg: impl Into<String>) -> Error {
    Error::Config(msg.into())
}
