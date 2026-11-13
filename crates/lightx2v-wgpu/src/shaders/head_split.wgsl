// One 256-thread workgroup per token: optional RMS across all heads, optional
// interleaved-pair RoPE, scatter to [heads, rows, head_dim].
// (fastwan-rs `head_split.wgsl`.)
struct P { rows: u32, stride: u32, col_off: u32, heads: u32, hd: u32, has_norm: u32, has_rope: u32, _a: u32, eps: f32, _b: u32, _c: u32, _d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> nw: array<f32>;
@group(0) @binding(3) var<storage, read> cs: array<f32>;
@group(0) @binding(4) var<storage, read> sn: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wg_index(wid, nwg);
    if (r >= p.rows) { return; }
    let t = lid.x;
    let width = p.heads * p.hd;
    let base = r * p.stride + p.col_off;
    var inv = 1.0;
    if (p.has_norm != 0u) {
        var s = 0.0;
        for (var j = t; j < width; j = j + 256u) { let v = src[base + j]; s = s + v * v; }
        red[t] = s;
        workgroupBarrier();
        for (var k = 128u; k > 0u; k = k >> 1u) {
            if (t < k) { red[t] = red[t] + red[t + k]; }
            workgroupBarrier();
        }
        inv = inverseSqrt(red[0] / f32(width) + p.eps);
    }
    let half = p.hd / 2u;
    for (var pi = t; pi < width / 2u; pi = pi + 256u) {
        let j = 2u * pi;
        let hh = j / p.hd;
        let c = j % p.hd;
        var x1 = src[base + j];
        var x2 = src[base + j + 1u];
        if (p.has_norm != 0u) {
            x1 = x1 * inv * nw[j];
            x2 = x2 * inv * nw[j + 1u];
        }
        if (p.has_rope != 0u) {
            let ti = r * half + c / 2u;
            let co = cs[ti];
            let si = sn[ti];
            let y1 = x1 * co - x2 * si;
            let y2 = x1 * si + x2 * co;
            x1 = y1;
            x2 = y2;
        }
        let o = (hh * p.rows + r) * p.hd + c;
        out[o] = x1;
        out[o + 1u] = x2;
    }
}
