//! History-contrast distillation (History-Forcing, arXiv:2611.02782).
//!
//! The teacher is run twice on the same chunk: once **with** the informative
//! early history (the frames the autoregressive rollout has already
//! generated) and once **without** it. The difference between the two
//! predictions is the history signal — what the past actually contributes.
//! The student is trained toward a corrected target:
//!
//! ```text
//! corrected = base_target + w_hist · (teacher_with_history − teacher_without_history)
//! ```
//!
//! where `base_target` is the usual DMD / flow-matching velocity target and
//! `w_hist` is the history-contrast weight. When the history carries no
//! information the two teacher predictions agree, the residual is zero, and
//! the target is unchanged; when the history matters, the residual steers
//! the student to use it, which is what keeps a 60 s rollout consistent
//! instead of drifting.

/// Weighted history-contrast residual: `w · (with − without)`.
pub fn contrast_residual(with_history: &[f32], without_history: &[f32], weight: f32) -> Vec<f32> {
    debug_assert_eq!(with_history.len(), without_history.len());
    with_history
        .iter()
        .zip(without_history)
        .map(|(&w, &wo)| weight * (w - wo))
        .collect()
}

/// The corrected training target for the student.
pub fn corrected_target(
    base_target: &[f32],
    with_history: &[f32],
    without_history: &[f32],
    weight: f32,
) -> Vec<f32> {
    debug_assert_eq!(base_target.len(), with_history.len());
    debug_assert_eq!(base_target.len(), without_history.len());
    base_target
        .iter()
        .zip(with_history)
        .zip(without_history)
        .map(|((&b, &w), &wo)| b + weight * (w - wo))
        .collect()
}

/// The History-Forcing training step's history-contrast term, as a small
/// stateful helper: the weight, plus the residual's magnitude (a diagnostic
/// for how much the history mattered on this chunk).
#[derive(Debug, Clone, Copy)]
pub struct HistoryContrast {
    /// `w_hist`, the history-contrast weight.
    pub weight: f32,
}

impl HistoryContrast {
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }

    pub fn correct(
        &self,
        base_target: &[f32],
        with_history: &[f32],
        without_history: &[f32],
    ) -> Vec<f32> {
        corrected_target(base_target, with_history, without_history, self.weight)
    }

    /// Mean absolute residual — 0 when the history carries no signal.
    pub fn residual_magnitude(&self, with_history: &[f32], without_history: &[f32]) -> f32 {
        debug_assert_eq!(with_history.len(), without_history.len());
        if with_history.is_empty() {
            return 0.0;
        }
        with_history
            .iter()
            .zip(without_history)
            .map(|(&w, &wo)| (w - wo).abs())
            .sum::<f32>()
            / with_history.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weight_or_no_signal_leaves_the_target() {
        let base = vec![1.0f32, 2.0, 3.0];
        let with = vec![5.0f32, 6.0, 7.0];
        let without = vec![4.0f32, 5.0, 6.0];
        assert_eq!(corrected_target(&base, &with, &without, 0.0), base);
        assert_eq!(corrected_target(&base, &with, &with, 2.5), base);
    }

    #[test]
    fn residual_is_the_weighted_difference() {
        let base = vec![0.0f32; 3];
        let with = vec![3.0f32, 1.0, -1.0];
        let without = vec![1.0f32, 1.0, 1.0];
        let out = corrected_target(&base, &with, &without, 0.5);
        assert_eq!(out, vec![1.0, 0.0, -1.0]);
        let hc = HistoryContrast::new(0.5);
        assert!((hc.residual_magnitude(&with, &without) - 4.0 / 3.0).abs() < 1e-6);
        assert_eq!(hc.residual_magnitude(&with, &with), 0.0);
    }
}
