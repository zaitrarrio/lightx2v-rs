// Row kernels: one 256-thread workgroup per row.
//   mode 0: out = LN(x)                    (layer_norm)
//   mode 1: h += a·gate | h += a; out = LN(h)   (residual_norm)
// then the norm: affine (w, b) or modulated (1 + mods[scale], mods[shift]).
// (fastwan-rs `norm.wgsl`.)
struct P { rows: u32, dim: u32, mode: u32, gate: u32, has_gate: u32, affine: u32, scale: u32, shift: u32, eps: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<f32>;
@group(0) @binding(3) var<storage, read> mods: array<f32>;
@group(0) @binding(4) var<storage, read> nw: array<f32>;
@group(0) @binding(5) var<storage, read> nb: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

fn block_sum(v: f32, t: u32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg_index(wid, nwg);
    if (row >= p.rows) { return; }
    let t = lid.x;
    let base = row * p.dim;
    var s = 0.0;
    for (var j = t; j < p.dim; j = j + 256u) {
        var v = h[base + j];
        if (p.mode == 1u) {
            var g = 1.0;
            if (p.has_gate != 0u) { g = mods[p.gate * p.dim + j]; }
            v = v + a[base + j] * g;
            h[base + j] = v;
        }
        s = s + v;
    }
    let mean = block_sum(s, t) / f32(p.dim);
    var q = 0.0;
    for (var j = t; j < p.dim; j = j + 256u) {
        let dv = h[base + j] - mean;
        q = q + dv * dv;
    }
    let inv = inverseSqrt(block_sum(q, t) / f32(p.dim) + p.eps);
    for (var j = t; j < p.dim; j = j + 256u) {
        let n = (h[base + j] - mean) * inv;
        var o: f32;
        if (p.affine != 0u) {
            o = n * nw[j] + nb[j];
        } else {
            o = n * (1.0 + mods[p.scale * p.dim + j]) + mods[p.shift * p.dim + j];
        }
        out[base + j] = o;
    }
}
