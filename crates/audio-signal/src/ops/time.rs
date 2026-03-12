use crate::{math::db_to_gain, signal::TimeSignal};

impl TimeSignal {
    pub fn normalize_peak(&mut self, peak_level: f64) {
        let max = self.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        if max == 0.0 {
            return;
        }
        let gain = peak_level / max;
        self.iter_mut().for_each(|s| *s *= gain);
    }

    pub fn normalize_peak_db(&mut self, peak_db: f64) {
        self.normalize_peak(db_to_gain(peak_db));
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    use crate::signal::TimeSignal;

    #[test]
    fn normalize_scales_to_requested_peak_linear() {
        let mut signal = TimeSignal::new(arr2(&[[0.25, -0.5, 1.0]]), 48_000.0).unwrap();
        signal.normalize_peak(0.5);

        assert_abs_diff_eq!(signal.channel(0)[0], 0.125, epsilon = 1e-12);
        assert_abs_diff_eq!(signal.channel(0)[1], -0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(signal.channel(0)[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn normalize_scales_to_requested_peak_db() {
        let mut signal = TimeSignal::new(arr2(&[[0.25, -0.5, 1.0]]), 48_000.0).unwrap();
        signal.normalize_peak_db(-6.0);

        let expected_peak = 10.0_f64.powf(-6.0 / 20.0);
        assert_abs_diff_eq!(signal.channel(0)[0], 0.25 * expected_peak, epsilon = 1e-12);
        assert_abs_diff_eq!(signal.channel(0)[1], -0.5 * expected_peak, epsilon = 1e-12);
        assert_abs_diff_eq!(signal.channel(0)[2], expected_peak, epsilon = 1e-12);
    }
}
