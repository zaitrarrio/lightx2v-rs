// MemRoPE cache compaction: gather rows by an index map, zero-filling
// rows the map marks PAD. One thread per output element.
//   map[r] = source row, or 0xffffffff for a zero row (a freshly seeded
//   EMA frame). `width` = heads·head_dim elements per row.
struct P { n: u32, width: u32, rows: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> map: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    let r = i / p.width;
    let src = map[r];
    if (src == 0xffffffffu) {
        x[i] = 0.0;
    } else {
        x[i] = y[src * p.width + i % p.width];
    }
}
