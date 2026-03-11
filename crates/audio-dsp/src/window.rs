use std::f64::consts::PI;

use ndarray::Array1;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WindowFn {
    #[default]
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

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
    fn hann_window_matches_expected_shape() {
        let window = generate_window(WindowFn::Hann, 5);
        assert_abs_diff_eq!(window[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(window[2], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(window[4], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn half_hann_helpers_taper_correct_side() {
        let mut left = [1.0; 4];
        let mut right = [1.0; 4];

        apply_hann_left(&mut left, 0, 4);
        apply_hann_right(&mut right, 0, 4);

        assert!(left[0] < left[3]);
        assert!(right[0] > right[3]);
        assert_abs_diff_eq!(left[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(right[3], 0.0, epsilon = 1e-12);
    }
}
