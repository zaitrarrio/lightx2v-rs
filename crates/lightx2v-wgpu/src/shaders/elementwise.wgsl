// Small elementwise kernels, one thread per output element, 256 per group.
//   op 0 gate_residual: x[i] += y[i] · aux[gate·dim + i % dim]
//   op 1 dmd_step:      x0 = x − σt·y; x = has_aux ? (1−σn)·x0 + σn·aux : x0
//   op 2 tanh_clamp:    x = tanh(x / s) · s
//   op 3 patchify:      x[tok, f] = y[latent]       (gather)
//   op 4 unpatchify:    x[latent] = y[tok, f]       (gather, channel-major
//                       features — the checkpoint contract, matching the
//                       CPU reference; fastwan's op 4 was channel-last)
//   op 5 kv_pack:       x[(off+r)·hdim + h·d + j] = y[(h·rows + r)·d + j]
//                       ([heads, rows, d] → cache rows [rows, heads·d])
//   op 6 kv_unpack:     x[(h·rows + r)·d + j] = y[r·hdim + h·d + j]
//                       (cache rows → [heads, rows, d])
//   op 7 ema_fold:      x[ema + i] = (1−α)·x[ema + i] + α·y[ev + i]
//                       (x and y are the same cache buffer, disjoint rows)
// Extra params a0..a3 carry the op's geometry; s0/s1 the scalars.
struct P { n: u32, op: u32, dim: u32, gate: u32, has_aux: u32, c: u32, t: u32, hh: u32, ww: u32, pp: u32, s0: f32, s1: f32, a0: u32, a1: u32, a2: u32, a3: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> aux: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    switch p.op {
        case 0u: { x[i] = x[i] + y[i] * aux[p.gate * p.dim + i % p.dim]; }
        case 1u: {
            let x0 = x[i] - p.s0 * y[i];
            if (p.has_aux != 0u) { x[i] = (1.0 - p.s1) * x0 + p.s1 * aux[i]; } else { x[i] = x0; }
        }
        case 2u: { x[i] = tanh_safe(x[i] / p.s0) * p.s0; }
        case 3u: {
            // i = tok·F + f, f = c'·p² + kh·p + kw
            let f = p.c * p.pp * p.pp;
            let tok = i / f;
            let fi = i % f;
            let cc = fi / (p.pp * p.pp);
            let kh = (fi / p.pp) % p.pp;
            let kw = fi % p.pp;
            let gw = p.ww / p.pp;
            let gh = p.hh / p.pp;
            let tt = tok / (gh * gw);
            let yy = (tok / gw) % gh;
            let xg = tok % gw;
            x[i] = y[((cc * p.t + tt) * p.hh + yy * p.pp + kh) * p.ww + xg * p.pp + kw];
        }
        case 4u: {
            // i = ((c'·t + tt)·hh + y)·ww + xx  (latent index)
            let xx = i % p.ww;
            let yy = (i / p.ww) % p.hh;
            let tt = (i / (p.ww * p.hh)) % p.t;
            let cc = i / (p.ww * p.hh * p.t);
            let gw = p.ww / p.pp;
            let gh = p.hh / p.pp;
            let tok = (tt * gh + yy / p.pp) * gw + xx / p.pp;
            let f = p.c * p.pp * p.pp;
            // channel-major feature: cc·p² + (yy%p)·p + (xx%p)
            x[i] = y[tok * f + cc * p.pp * p.pp + (yy % p.pp) * p.pp + xx % p.pp];
        }
        case 5u: {
            // kv_pack: a0 = heads, a1 = head_dim, a2 = rows, a3 = row offset
            let hdim = p.a0 * p.a1;
            let r = i / hdim;
            let rem = i % hdim;
            let h = rem / p.a1;
            let j = rem % p.a1;
            x[(p.a3 + r) * hdim + h * p.a1 + j] = y[(h * p.a2 + r) * p.a1 + j];
        }
        case 6u: {
            // kv_unpack: a0 = heads, a1 = head_dim, a2 = rows
            let hdim = p.a0 * p.a1;
            let h = i / (p.a2 * p.a1);
            let rem = i % (p.a2 * p.a1);
            let r = rem / p.a1;
            let j = rem % p.a1;
            x[(h * p.a2 + r) * p.a1 + j] = y[r * hdim + h * p.a1 + j];
        }
        case 7u: {
            // ema_fold: a0 = ema row offset (elements), a1 = evicted row
            // offset (elements); s0 = alpha. x and y are the same buffer.
            let v = y[p.a1 + i];
            x[p.a0 + i] = (1.0 - p.s0) * x[p.a0 + i] + p.s0 * v;
        }
        default: { }
    }
    if (p.op == 9999u) { x[0] = aux[0]; }
}
