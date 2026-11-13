// VSA coarse stage: one workgroup per (query tile, head).
//   scores over all key tiles (mean q · mean k · scale), softmax,
//   coarse[h][qt] = Σ p · mean v, and the top-k key tiles: every tile whose
//   score is above the k-th largest, then ties at it by lower index — the same
//   set as sorting by (score desc, index asc).
// (fastwan-rs `vsa_coarse.wgsl`.)
struct P { heads: u32, d: u32, nb: u32, topk: u32, scale: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pooled: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;
@group(0) @binding(3) var<storage, read_write> sel: array<u32>;

var<workgroup> sc: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> cnt: atomic<u32>;
var<workgroup> thr: u32;

// Float → u32 with the same order.
fn okey(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let qt = wid.x;
    let h = wid.y;
    let plane = p.heads * p.nb * p.d;
    let qb = (h * p.nb + qt) * p.d;
    var mx = -3.0e38;
    for (var kt = tid; kt < p.nb; kt = kt + 256u) {
        let kb = plane + (h * p.nb + kt) * p.d;
        var dot = 0.0;
        for (var c = 0u; c < p.d; c = c + 1u) { dot = fma(pooled[qb + c], pooled[kb + c], dot); }
        let s = dot * p.scale;
        sc[kt] = s;
        mx = max(mx, s);
    }
    red[tid] = mx;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { red[tid] = max(red[tid], red[tid + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var z = 0.0;
    for (var kt = tid; kt < p.nb; kt = kt + 256u) { z = z + exp(sc[kt] - m); }
    red[tid] = z;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
        workgroupBarrier();
    }
    let zt = red[0];
    if (tid < p.d) {
        var o = 0.0;
        for (var kt = 0u; kt < p.nb; kt = kt + 1u) {
            o = fma(exp(sc[kt] - m), pooled[2u * plane + (h * p.nb + kt) * p.d + tid], o);
        }
        coarse[qb + tid] = o / zt;
    }
    // Largest key T with count(key >= T) >= topk, bit by bit from the top.
    var t = 0u;
    for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
        let cand = t | (1u << u32(bit));
        if (tid == 0u) { atomicStore(&cnt, 0u); }
        workgroupBarrier();
        var local = 0u;
        for (var kt = tid; kt < p.nb; kt = kt + 256u) { if (okey(sc[kt]) >= cand) { local = local + 1u; } }
        atomicAdd(&cnt, local);
        workgroupBarrier();
        if (atomicLoad(&cnt) >= p.topk) { t = cand; }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let out0 = (h * p.nb + qt) * p.topk;
        var n = 0u;
        for (var kt = 0u; kt < p.nb; kt = kt + 1u) { if (okey(sc[kt]) > t) { sel[out0 + n] = kt; n = n + 1u; } }
        for (var kt = 0u; kt < p.nb && n < p.topk; kt = kt + 1u) { if (okey(sc[kt]) == t) { sel[out0 + n] = kt; n = n + 1u; } }
    }
}
