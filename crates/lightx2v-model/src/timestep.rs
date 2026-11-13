//! Host-side timestep conditioning math (a few mat-vecs per DMD step, run
//! on the CPU once per step; only 4 timestep values exist).
//!
//! Diffusers `get_timestep_embedding`: `freqs = exp(arange(half) ·
//! −log(10000)/(half−1))`, `emb = [cos(t·freqs), sin(t·freqs)]`.

/// `[cos(t·freqs), sin(t·freqs)]` of length `dim` (must be even).
pub fn sinusoid(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let f = (-(10000f64).ln() * i as f64 / (half - 1) as f64).exp() as f32;
        let a = t * f;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// `y = W·x + b` for a `[rows, cols]` row-major matrix.
pub fn matvec(w: &[f32], bias: Option<&[f32]>, x: &[f32]) -> Vec<f32> {
    let cols = x.len();
    debug_assert_eq!(w.len() % cols, 0);
    let rows = w.len() / cols;
    (0..rows)
        .map(|r| {
            let mut acc = bias.map_or(0.0, |b| b[r]);
            for (i, &v) in x.iter().enumerate() {
                acc += w[r * cols + i] * v;
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoid_at_zero_is_cos_one_sin_zero() {
        let e = sinusoid(0.0, 8);
        assert_eq!(&e[..4], &[1.0; 4]);
        assert_eq!(&e[4..], &[0.0; 4]);
    }

    #[test]
    fn matvec_and_silu() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(matvec(&w, None, &[1.0, 1.0]), vec![3.0, 7.0]);
        assert_eq!(matvec(&w, Some(&[1.0, -1.0]), &[1.0, 1.0]), vec![4.0, 6.0]);
        let s = silu(&[0.0]);
        assert_eq!(s, vec![0.0]);
        assert!((silu(&[2.0])[0] - 2.0 / (1.0 + (-2.0f32).exp())).abs() < 1e-6);
    }
}
