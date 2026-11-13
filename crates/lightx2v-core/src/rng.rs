//! Deterministic RNG for synthetic weights and initial noise.
//!
//! SplitMix64 + Box–Muller. Same seed → same stream on every backend, so a
//! synthetic model is identical wherever it runs.

/// A small deterministic generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 needs a non-zero start; the seed itself is the state.
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `(-a, a)`.
    pub fn uniform(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|_| (self.next_f32() * 2.0 - 1.0) * a).collect()
    }

    /// Standard-normal samples (Box–Muller, cached spare).
    pub fn normal(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = self.next_f32().max(1e-7);
            let u2 = self.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let t = 2.0 * std::f32::consts::PI * u2;
            out.push(r * t.cos());
            if out.len() < n {
                out.push(r * t.sin());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_gaussian_shaped() {
        let a = Rng::new(7).normal(1000);
        let b = Rng::new(7).normal(1000);
        assert_eq!(a, b);
        assert_ne!(Rng::new(8).normal(16), Rng::new(7).normal(16));
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.1, "{mean}");
        assert!((var - 1.0).abs() < 0.15, "{var}");
    }
}
