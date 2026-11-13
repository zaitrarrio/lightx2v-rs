// Shared by every kernel. `{W_DECL}` / `{W_LOAD}` are filled per weight
// storage: f16 pairs packed in u32 (core WGSL `unpack2x16float`, so no
// `shader-f16` extension is needed) or plain f32.
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608;
    let u = c * (x + 0.044715 * x * x * x);
    // tanh via exp, clamped: WGSL tanh overflows to NaN on some drivers.
    let e = exp(-2.0 * abs(u));
    let t = sign(u) * (1.0 - e) / (1.0 + e);
    return 0.5 * x * (1.0 + t);
}
fn tanh_safe(u: f32) -> f32 {
    let e = exp(-2.0 * abs(u));
    return sign(u) * (1.0 - e) / (1.0 + e);
}
// Linear workgroup index of a (possibly 2-D folded) dispatch.
fn wg_index(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.x + wid.y * nwg.x;
}
