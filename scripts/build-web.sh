#!/usr/bin/env bash
# Build the in-browser engine (crates/lightx2v-web) into web/pkg for Prompt
# Studio's "In this browser" mode. Needs the wasm32 target and a
# wasm-bindgen CLI matching the crate (0.2.130):
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.130 --locked
# The build is CPU-in-wasm (the tiny synthetic model): wgpu-on-wasm32 is a
# follow-up, see docs/design.md §10. No network is needed beyond the local
# registry cache; the crate builds offline.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown -p lightx2v-web --offline
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/lightx2v_web.wasm
ls -la web/pkg
