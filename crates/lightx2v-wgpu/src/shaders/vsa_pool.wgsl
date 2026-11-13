// VSA tile means: one workgroup per (tile, head); thread c < d averages
// column c of q, k, v over the tile's real tokens.
// (fastwan-rs `vsa_pool.wgsl`.)
struct P { heads: u32, seq: u32, d: u32, nb: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> slot: array<u32>;
@group(0) @binding(5) var<storage, read> bsz: array<u32>;
@group(0) @binding(6) var<storage, read_write> pooled: array<f32>;   // [3][heads][nb][d]

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) c: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = wid.x;
    let h = wid.y;
    if (c >= p.d) { return; }
    let n = bsz[t];
    var sq = 0.0;
    var sk = 0.0;
    var sv = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let idx = (h * p.seq + slot[t * 64u + i]) * p.d + c;
        sq = sq + q[idx];
        sk = sk + k[idx];
        sv = sv + v[idx];
    }
    let inv = 1.0 / f32(max(n, 1u));
    let o = (h * p.nb + t) * p.d + c;
    let plane = p.heads * p.nb * p.d;
    pooled[o] = sq * inv;
    pooled[plane + o] = sk * inv;
    pooled[2u * plane + o] = sv * inv;
}
