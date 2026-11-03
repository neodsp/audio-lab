use std::f64::consts::PI;

use ndarray::Array1;

/// Public window function identifiers used across generic DSP helpers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WindowFn {
    #[default]
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

/// Generate a symmetric window: both ends have the same value (zero for Hann
/// and Blackman). Use it for FIR filter design, smoothing kernels and fades.
/// For spectral analysis use [`generate_periodic_window`].
pub fn generate_window(window_fn: WindowFn, size: usize) -> Array1<f64> {
    window_with_period(window_fn, size, size.saturating_sub(1))
}

/// Generate a periodic ("DFT-even") window: the first `size` samples of a
/// symmetric window of `size + 1` samples. This is the standard window for
/// FFT-based spectral analysis such as [`stft`](crate::dsp::stft()), the default
/// in scipy, librosa and PyTorch. A periodic Hann window at 50 % or 75 % overlap
/// sums to a constant, and none of these windows is all zeros.
pub fn generate_periodic_window(window_fn: WindowFn, size: usize) -> Array1<f64> {
    window_with_period(window_fn, size, size)
}

/// `size` samples of `window_fn`, where sample `period` would complete one
/// cycle. Windows of 0 or 1 samples are all ones.
fn window_with_period(window_fn: WindowFn, size: usize, period: usize) -> Array1<f64> {
    if size <= 1 {
        return Array1::ones(size);
    }

    let n = period as f64;
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

    #[test]
    fn periodic_window_is_symmetric_window_one_sample_longer_without_the_last() {
        for window_fn in [
            WindowFn::Hann,
            WindowFn::Hamming,
            WindowFn::Blackman,
            WindowFn::Rectangular,
        ] {
            for size in [2, 3, 8, 9, 64] {
                let periodic = generate_periodic_window(window_fn, size);
                let symmetric = generate_window(window_fn, size + 1);
                for (p, s) in periodic.iter().zip(symmetric.iter()) {
                    assert_abs_diff_eq!(*p, *s, epsilon = 1e-12);
                }
            }
        }
        assert_eq!(
            generate_periodic_window(WindowFn::Hann, 1),
            Array1::<f64>::ones(1)
        );
        assert_eq!(
            generate_periodic_window(WindowFn::Hann, 0),
            Array1::<f64>::ones(0)
        );
    }

    #[test]
    fn periodic_hann_sums_to_a_constant_at_50_and_75_percent_overlap() {
        let size = 16;
        let window = generate_periodic_window(WindowFn::Hann, size);
        for hop in [size / 2, size / 4] {
            let expected = size as f64 / (2 * hop) as f64;
            for offset in 0..hop {
                let sum: f64 = window.iter().skip(offset).step_by(hop).sum();
                assert_abs_diff_eq!(sum, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn periodic_windows_are_never_all_zero() {
        // The symmetric 2-sample Hann and Blackman windows are all zeros.
        for window_fn in [WindowFn::Hann, WindowFn::Blackman] {
            assert!(
                generate_window(window_fn, 2)
                    .iter()
                    .all(|w| w.abs() < 1e-12)
            );
            assert!(generate_periodic_window(window_fn, 2).sum() > 0.4);
        }
    }
}
