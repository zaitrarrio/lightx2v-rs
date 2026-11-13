//! DMD sampling schedule (LightX2V `dmd2` / History-Forcing DMD student).
//!
//! The student denoises at `denoising_step_list = [1000, 750, 500, 250]`.
//! Each step maps onto the nearest entry of the *training* noise table — a
//! `FlowMatchEulerDiscreteScheduler(shift = 5.0)` — and:
//!
//! ```text
//! x0 = x − σ_t · v
//! x  = (1 − σ_next) · x0 + σ_next · ε      (fresh ε; skipped on the last step)
//! ```
//!
//! The shift differs from fastwan-rs's FastWan 1.3B schedule (shift 8,
//! steps `[1000, 757, 522]`): the LightX2V distill config sets
//! `sample_shift` / `timestep_shift` to 5.0.

/// The LightX2V / History-Forcing DMD step list.
pub const DMD_STEPS: [i32; 4] = [1000, 750, 500, 250];
/// `sample_shift` / `model_kwargs.timestep_shift` for the DMD student.
pub const DMD_TIMESTEP_SHIFT: f64 = 5.0;

#[derive(Debug, Clone)]
pub struct DmdSchedule {
    /// Timesteps fed to the DiT, in order.
    pub timesteps: Vec<i32>,
    /// Table sigma for each step, then a trailing `0.0`.
    pub sigmas: Vec<f64>,
    pub shift: f64,
}

/// Float32 sigma / timestep training tables, index 0 = timestep N.
/// `sigma(u) = shift·u / (1 + (shift−1)·u)` with `u = (N − i)/N`.
fn training_table(n: usize, shift: f64) -> (Vec<f32>, Vec<f32>) {
    let (nf, shift_f, shift_m1) = (n as f32, shift as f32, (shift - 1.0) as f32);
    (0..n)
        .map(|i| {
            let s = (n - i) as f32 / nf;
            let s = (shift_f * s) / (1.0 + shift_m1 * s);
            (s, s * nf)
        })
        .unzip()
}

impl DmdSchedule {
    pub fn new(steps: &[i32], shift: f64) -> Self {
        let (sig, ts) = training_table(1000, shift);
        let sigma_for = |t: f64| {
            let mut best = (0usize, f64::INFINITY);
            for (i, &v) in ts.iter().enumerate() {
                let d = (f64::from(v) - t).abs();
                if d < best.1 {
                    best = (i, d);
                }
            }
            f64::from(sig[best.0])
        };
        Self {
            timesteps: steps.to_vec(),
            sigmas: steps
                .iter()
                .map(|&t| sigma_for(f64::from(t)))
                .chain(std::iter::once(0.0))
                .collect(),
            shift,
        }
    }

    /// The LightX2V / History-Forcing 4-step schedule.
    pub fn dmd2() -> Self {
        Self::new(&DMD_STEPS, DMD_TIMESTEP_SHIFT)
    }

    pub fn len(&self) -> usize {
        self.timesteps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }

    /// `(σ_t, σ_next)`; `σ_next` is `None` on the last step (no re-noise).
    pub fn coeffs(&self, i: usize) -> (f32, Option<f32>) {
        let next = (i + 1 < self.len()).then(|| self.sigmas[i + 1] as f32);
        (self.sigmas[i] as f32, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dmd2_sigmas() {
        let s = DmdSchedule::dmd2();
        assert_eq!(s.sigmas[0], 1.0);
        // The nearest-entry lookup maps a timestep onto itself: sigma(t) ≈ t/1000.
        assert!((s.sigmas[1] - 0.75).abs() < 1e-3, "{}", s.sigmas[1]);
        assert!((s.sigmas[2] - 0.5).abs() < 1e-3, "{}", s.sigmas[2]);
        // 250 has no exact table hit (the table steps are 0.001 in u, and
        // sigma(250) falls between two entries 1.6 apart); the nearest entry
        // is 0.2516, so the tolerance is looser here than for exact hits.
        assert!((s.sigmas[3] - 0.25).abs() < 3e-3, "{}", s.sigmas[3]);
        assert_eq!(s.coeffs(3).1, None);
        assert_eq!(s.len(), 4);
    }
}
