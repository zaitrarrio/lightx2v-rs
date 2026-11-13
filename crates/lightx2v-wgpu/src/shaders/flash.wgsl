// Flash attention: one workgroup = 64 query rows of one head, one thread
// per row; K/V stream through shared memory {BK} keys at a time with an
// online softmax, so the [sq, sk] score matrix never exists. Output is
// written with heads merged ([sq, heads·d]).
//
// Three modes (the LightX2V extension over fastwan-rs `flash.wgsl`):
//   mode 0 dense: query rows wid.x·64 + r, key chunks of consecutive rows.
//   mode 1 VSA fine stage: query tile wid.x (tokens via `slot`), key chunk
//     c = selected tile sel[h][qt][c] (tokens via `slot`); the output adds
//     coarse[h][qt] · gate[h][token] (the VSA combine).
//   mode 2 causal masked: like dense, but a key is valid only where
//     mask[qi·sk + ki] != 0 (MemRoPE block mask). A row with no valid key
//     outputs zeros (the CPU reference would produce NaN there; the DiT
//     never has an all-masked row — every query sees the sink frames).
// Invalid keys (masked, or PAD slots in VSA tiles) score -1e30 and are
// skipped in the accumulation, so they contribute exactly 0.
struct P { heads: u32, sq: u32, sk: u32, d: u32, scale: f32, mode: u32, nb: u32, topk: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> slot: array<u32>;
@group(0) @binding(6) var<storage, read> sel: array<u32>;
@group(0) @binding(7) var<storage, read> coarse: array<f32>;
@group(0) @binding(8) var<storage, read> gate: array<f32>;
@group(0) @binding(9) var<storage, read> mask: array<u32>;

const BK: u32 = {BK}u;
const DMAX: u32 = 128u;
const PAD: u32 = 0xffffffffu;
var<workgroup> ks: array<f32, {BK_D}>;
var<workgroup> vs: array<f32, {BK_D}>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.y;
    let qt = wid.x;
    let d = p.d;
    // Query token of this thread (PAD = no query row).
    var qi = PAD;
    if (p.mode == 1u) {
        qi = slot[qt * 64u + lid.x];
    } else {
        let r = qt * 64u + lid.x;
        if (r < p.sq) { qi = r; }
    }
    let live = qi != PAD;
    var qr: array<f32, 128>;
    var acc: array<f32, 128>;
    if (live) {
        for (var c = 0u; c < d; c = c + 1u) { qr[c] = q[(h * p.sq + qi) * d + c] * p.scale; }
    }
    var m = -3.0e38;
    var l = 0.0;
    var s: array<f32, {BK}>;
    var nchunks = (p.sk + BK - 1u) / BK;
    if (p.mode == 1u) { nchunks = p.topk; }
    for (var c = 0u; c < nchunks; c = c + 1u) {
        // Key tokens of this chunk (PAD = padding slot / out of range).
        var krows: array<u32, {BK}>;
        for (var jj = 0u; jj < BK; jj = jj + 1u) {
            var tok = PAD;
            if (p.mode == 1u) {
                let tile = sel[(h * p.nb + qt) * p.topk + c];
                tok = slot[tile * 64u + jj];
            } else {
                let r = c * BK + jj;
                if (r < p.sk) { tok = r; }
            }
            krows[jj] = tok;
        }
        let nk = BK;
        for (var e = lid.x; e < BK * d; e = e + 64u) {
            let jj = e / d;
            var kv = 0.0;
            var vv = 0.0;
            if (krows[jj] != PAD) {
                let g = (h * p.sk + krows[jj]) * d + (e % d);
                kv = k[g];
                vv = v[g];
            }
            ks[e] = kv;
            vs[e] = vv;
        }
        workgroupBarrier();
        if (live) {
            var mt = m;
            for (var jj = 0u; jj < nk; jj = jj + 1u) {
                var valid = krows[jj] != PAD;
                if (p.mode == 2u && valid) {
                    valid = mask[qi * p.sk + krows[jj]] != 0u;
                }
                if (!valid) {
                    s[jj] = -1.0e30;
                    continue;
                }
                var dot = 0.0;
                for (var cc = 0u; cc < d; cc = cc + 1u) { dot = fma(qr[cc], ks[jj * d + cc], dot); }
                s[jj] = dot;
                mt = max(mt, dot);
            }
            let corr = exp(m - mt);
            l = l * corr;
            for (var cc = 0u; cc < d; cc = cc + 1u) { acc[cc] = acc[cc] * corr; }
            for (var jj = 0u; jj < nk; jj = jj + 1u) {
                if (s[jj] <= -1.0e29) { continue; }
                let pj = exp(s[jj] - mt);
                l = l + pj;
                for (var cc = 0u; cc < d; cc = cc + 1u) { acc[cc] = fma(pj, vs[jj * d + cc], acc[cc]); }
            }
            m = mt;
        }
        workgroupBarrier();
    }
    if (live) {
        var inv = 0.0;
        if (l > 0.0) { inv = 1.0 / l; }
        for (var cc = 0u; cc < d; cc = cc + 1u) {
            var o = acc[cc] * inv;
            if (p.mode == 1u) {
                o = o + coarse[(h * p.nb + qt) * d + cc] * gate[(h * p.sq + qi) * d + cc];
            }
            out[(qi * p.heads + h) * d + cc] = o;
        }
    }
}
