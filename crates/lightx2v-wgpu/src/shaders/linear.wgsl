// y[y_row + m, n] = act(Σ_k x[x_row + m, k] · W[n, k] + b[n])
// 64×64 output tile per workgroup, 16×16 threads, 4×4 outputs per thread at
// stride 16 (coalesced stores, conflict-free shared reads), K step 16.
// (fastwan-rs `linear.wgsl`, same tiling; `{W_TYPE}` / `{W_LOAD}` are
// filled per weight storage — f16 pairs in u32, plain f32, or Q8/Q4 codes.)
struct P { m: u32, n: u32, k: u32, x_row: u32, y_row: u32, has_bias: u32, act: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn load_w(i: u32) -> f32 { {W_LOAD} }

var<workgroup> xs: array<array<f32, 64>, 16>;
var<workgroup> ws: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    var acc: array<array<f32, 4>, 4>;
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let r = e / 16u;
            let kk = e % 16u;
            let gk = k0 + kk;
            var xv = 0.0;
            if (m0 + r < p.m && gk < p.k) { xv = x[(p.x_row + m0 + r) * p.k + gk]; }
            xs[kk][r] = xv;
            var wv = 0.0;
            if (n0 + r < p.n && gk < p.k) { wv = load_w((n0 + r) * p.k + gk); }
            ws[kk][r] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i] = xs[kk][lid.y + 16u * i];
                b[i] = ws[kk][lid.x + 16u * i];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + lid.y + 16u * i;
        if (m >= p.m) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + lid.x + 16u * j;
            if (n >= p.n) { continue; }
            var v = acc[i][j];
            if (p.has_bias != 0u) { v = v + bias[n]; }
            if (p.act == 1u) { v = gelu_tanh(v); }
            y[(p.y_row + m) * p.n + n] = v;
        }
    }
}
