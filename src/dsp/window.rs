use std::f64::consts::PI;

use ndarray::Array1;

/// Public window function identifiers used across generic DSP helpers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WindowFn {
    #[default]
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

/// Generate a full window of the requested type.
pub fn generate_window(window_fn: WindowFn, size: usize) -> Array1<f64> {
    if size <= 1 {
        return Array1::ones(size);
    }

    let n = (size - 1) as f64;
    Array1::from_iter((0..size).map(|i| {
        let x = i as f64;
        match window_fn {
            WindowFn::Hann => 0.5 * (1.0 - (2.0 * PI * x / n).cos()),
            WindowFn::Hamming => 0.54 - 0.46 * (2.0 * PI * x / n).cos(),
            WindowFn::Blackman => {
                0.42 - 0.5 * (2.0 * PI * x / n).cos() + 0.08 * (4.0 * PI * x / n).cos()
            }
            WindowFn::Rectangular => 1.0,
        }
    }))
}

/// Apply a full Hann taper to a slice range in-place.
///
/// This is the public replacement for ad-hoc `window_hanning` helpers used in
/// internal prototypes.
pub fn apply_hann(data: &mut [f64], start: usize, len: usize) {
    let start = start.min(data.len());
    let len = len.min(data.len().saturating_sub(start));
    if len == 0 {
        return;
    }

    data.iter_mut()
        .skip(start)
        .take(len)
        .zip(generate_window(WindowFn::Hann, len))
        .for_each(|(sample, weight)| *sample *= weight);
}

/// Apply the rising half of a Hann taper to a slice range in-place.
pub fn apply_hann_left(data: &mut [f64], start: usize, len: usize) {
    let start = start.min(data.len());
    let len = len.min(data.len().saturating_sub(start));
    if len == 0 {
        return;
    }

    data.iter_mut()
        .skip(start)
        .take(len)
        .zip(
            generate_window(WindowFn::Hann, len * 2)
                .into_iter()
                .take(len),
        )
        .for_each(|(sample, weight)| *sample *= weight);
}

/// Apply the falling half of a Hann taper to a slice range in-place.
pub fn apply_hann_right(data: &mut [f64], start: usize, len: usize) {
    let start = start.min(data.len());
    let len = len.min(data.len().saturating_sub(start));
    if len == 0 {
        return;
    }

    data.iter_mut()
        .skip(start)
        .take(len)
        .zip(
            generate_window(WindowFn::Hann, len * 2)
                .into_iter()
                .skip(len),
        )
        .for_each(|(sample, weight)| *sample *= weight);
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn apply_hann_matches_expected_full_window_taper() {
        let mut data = [1.0; 5];
        let len = data.len();

        apply_hann(&mut data, 0, len);

        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        for (sample, expected) in data.into_iter().zip(expected) {
            assert_abs_diff_eq!(sample, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn apply_hann_left_matches_rising_half_window_taper() {
        let mut data = [1.0; 4];
        let len = data.len();

        apply_hann_left(&mut data, 0, len);

        let expected = generate_window(WindowFn::Hann, 8);
        for (sample, expected) in data.into_iter().zip(expected.into_iter().take(4)) {
            assert_abs_diff_eq!(sample, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn apply_hann_right_matches_falling_half_window_taper() {
        let mut data = [1.0; 4];
        let len = data.len();

        apply_hann_right(&mut data, 0, len);

        let expected = generate_window(WindowFn::Hann, 8);
        for (sample, expected) in data.into_iter().zip(expected.into_iter().skip(4)) {
            assert_abs_diff_eq!(sample, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn apply_hann_matches_generated_window_for_subslice() {
        let mut data = [1.0; 8];
        apply_hann(&mut data, 2, 4);

        let expected = generate_window(WindowFn::Hann, 4);
        assert_abs_diff_eq!(data[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(data[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(data[2], expected[0], epsilon = 1e-12);
        assert_abs_diff_eq!(data[3], expected[1], epsilon = 1e-12);
        assert_abs_diff_eq!(data[4], expected[2], epsilon = 1e-12);
        assert_abs_diff_eq!(data[5], expected[3], epsilon = 1e-12);
        assert_abs_diff_eq!(data[6], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(data[7], 1.0, epsilon = 1e-12);
    }
}
