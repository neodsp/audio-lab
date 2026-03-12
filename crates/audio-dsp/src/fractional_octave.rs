use audio_signal::signal::FreqSignal;
use ndarray::Array2;
use num::complex::Complex64;

use crate::window::{WindowFn, generate_window};

#[derive(Debug, thiserror::Error)]
pub enum FractionalOctaveSmoothingError {
    #[error("num_fractions must be > 0, got {0}")]
    InvalidNumFractions(f64),
    #[error("signal must have at least 2 frequency bins")]
    TooFewBins,
    #[error(
        "smoothing width given by num_fractions is below the frequency resolution of the signal"
    )]
    BelowFrequencyResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FractionalOctaveSmoothingMode {
    #[default]
    Magnitude,
    MagnitudeZeroPhase,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractionalOctaveSmoothingConfig {
    pub num_fractions: f64,
    pub mode: FractionalOctaveSmoothingMode,
    pub window_fn: WindowFn,
}

impl FractionalOctaveSmoothingConfig {
    pub fn new(num_fractions: f64) -> Self {
        Self {
            num_fractions,
            mode: FractionalOctaveSmoothingMode::Magnitude,
            window_fn: WindowFn::Rectangular,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractionalOctaveSmoothingStats {
    pub window_len: usize,
    pub actual_num_fractions: f64,
}

pub fn smooth_fractional_octave(
    signal: &FreqSignal,
    config: &FractionalOctaveSmoothingConfig,
) -> Result<(FreqSignal, FractionalOctaveSmoothingStats), FractionalOctaveSmoothingError> {
    if config.num_fractions <= 0.0 {
        return Err(FractionalOctaveSmoothingError::InvalidNumFractions(
            config.num_fractions,
        ));
    }

    let num_bins = signal.num_freq_bins();
    if num_bins < 2 {
        return Err(FractionalOctaveSmoothingError::TooFewBins);
    }

    let n_log = logarithmic_positions(num_bins);
    let delta_n = n_log[1].log2();
    let window_len = (2.0 * (1.0 / (config.num_fractions * delta_n * 2.0)).floor() + 1.0) as usize;

    if window_len <= 1 {
        return Err(FractionalOctaveSmoothingError::BelowFrequencyResolution);
    }

    let window = generate_window(config.window_fn, window_len).to_vec();
    let window_sum = window.iter().sum::<f64>();
    let mut smoothed_magnitude = Array2::zeros((signal.num_channels(), num_bins));

    for (mut output_channel, input_channel) in smoothed_magnitude
        .outer_iter_mut()
        .zip(signal.channel_iter())
    {
        let magnitudes = input_channel
            .iter()
            .map(|value| value.norm())
            .collect::<Vec<_>>();
        let warped = interpolate_uniform_catmull_rom(&magnitudes, &n_log);
        let smoothed = smooth_clamped(&warped, &window, window_sum);
        let unwarped = interpolate_samples_catmull_rom(
            &n_log,
            &smoothed,
            &(1..=num_bins).map(|idx| idx as f64).collect::<Vec<_>>(),
        );

        for (out, value) in output_channel.iter_mut().zip(unwarped) {
            *out = value.max(0.0);
        }
    }

    let mut output = signal.clone();
    for (mut output_channel, (input_channel, smoothed_channel)) in output
        .channel_iter_mut()
        .zip(signal.channel_iter().zip(smoothed_magnitude.outer_iter()))
    {
        for ((out_bin, input_bin), &magnitude) in output_channel
            .iter_mut()
            .zip(input_channel.iter())
            .zip(smoothed_channel.iter())
        {
            *out_bin = match config.mode {
                FractionalOctaveSmoothingMode::Magnitude => {
                    let norm = input_bin.norm();
                    if norm > 0.0 {
                        *input_bin * (magnitude / norm)
                    } else {
                        Complex64::new(magnitude, 0.0)
                    }
                }
                FractionalOctaveSmoothingMode::MagnitudeZeroPhase => Complex64::new(magnitude, 0.0),
            };
        }

        force_real_fft_endpoints(&mut output_channel, signal.num_time_steps());
    }

    Ok((
        output,
        FractionalOctaveSmoothingStats {
            window_len,
            actual_num_fractions: 1.0 / (window_len as f64 * delta_n),
        },
    ))
}

fn logarithmic_positions(num_bins: usize) -> Vec<f64> {
    (0..num_bins)
        .map(|idx| (num_bins as f64).powf(idx as f64 / (num_bins - 1) as f64))
        .collect()
}

fn interpolate_uniform_catmull_rom(values: &[f64], query_points: &[f64]) -> Vec<f64> {
    query_points
        .iter()
        .map(|&query| sample_uniform_catmull_rom(values, query))
        .collect()
}

fn interpolate_samples_catmull_rom(
    x_values: &[f64],
    y_values: &[f64],
    query_points: &[f64],
) -> Vec<f64> {
    debug_assert_eq!(x_values.len(), y_values.len());

    query_points
        .iter()
        .map(|&query| sample_nonuniform_catmull_rom(x_values, y_values, query))
        .collect()
}

fn sample_uniform_catmull_rom(values: &[f64], query: f64) -> f64 {
    let num_bins = values.len();
    debug_assert!(num_bins >= 2);

    if query <= 1.0 {
        return values[0];
    }
    if query >= num_bins as f64 {
        return values[num_bins - 1];
    }

    let position = query - 1.0;
    let index = position.floor() as usize;
    let t = position - index as f64;

    let p0 = values[index.saturating_sub(1)];
    let p1 = values[index];
    let p2 = values[(index + 1).min(num_bins - 1)];
    let p3 = values[(index + 2).min(num_bins - 1)];

    catmull_rom(p0, p1, p2, p3, t)
}

fn sample_nonuniform_catmull_rom(x_values: &[f64], y_values: &[f64], query: f64) -> f64 {
    let last = x_values.len() - 1;
    if query <= x_values[0] {
        return y_values[0];
    }
    if query >= x_values[last] {
        return y_values[last];
    }

    let upper = x_values.partition_point(|&x| x < query);
    let index = upper.saturating_sub(1);

    let i0 = index.saturating_sub(1);
    let i1 = index;
    let i2 = (index + 1).min(last);
    let i3 = (index + 2).min(last);

    let x1 = x_values[i1];
    let x2 = x_values[i2];
    let t = if x2 > x1 {
        (query - x1) / (x2 - x1)
    } else {
        0.0
    };

    catmull_rom(y_values[i0], y_values[i1], y_values[i2], y_values[i3], t)
}

fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;

    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn smooth_clamped(values: &[f64], window: &[f64], window_sum: f64) -> Vec<f64> {
    let radius = window.len() / 2;

    (0..values.len())
        .map(|center| {
            let weighted_sum = window
                .iter()
                .enumerate()
                .map(|(offset, weight)| {
                    let sample_index = center as isize + offset as isize - radius as isize;
                    let sample_index = sample_index.clamp(0, values.len() as isize - 1) as usize;
                    values[sample_index] * weight
                })
                .sum::<f64>();

            weighted_sum / window_sum
        })
        .collect()
}

fn force_real_fft_endpoints(
    channel: &mut ndarray::ArrayViewMut1<'_, Complex64>,
    num_time_steps: usize,
) {
    if channel.is_empty() {
        return;
    }

    channel[0] = Complex64::new(channel[0].norm(), 0.0);
    if num_time_steps % 2 == 0 && channel.len() > 1 {
        let last = channel.len() - 1;
        channel[last] = Complex64::new(channel[last].norm(), 0.0);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, arr2};

    use super::*;

    #[test]
    fn fractional_octave_smoothing_rejects_invalid_inputs() {
        let signal = FreqSignal::zeros(1, 8, 48_000.0, Some(14)).unwrap();

        let error = smooth_fractional_octave(&signal, &FractionalOctaveSmoothingConfig::new(0.0))
            .unwrap_err();
        assert!(matches!(
            error,
            FractionalOctaveSmoothingError::InvalidNumFractions(0.0)
        ));

        let error = smooth_fractional_octave(&signal, &FractionalOctaveSmoothingConfig::new(100.0))
            .unwrap_err();
        assert!(matches!(
            error,
            FractionalOctaveSmoothingError::BelowFrequencyResolution
        ));

        let too_short = FreqSignal::zeros(1, 1, 48_000.0, Some(1)).unwrap();
        let error =
            smooth_fractional_octave(&too_short, &FractionalOctaveSmoothingConfig::new(1.0))
                .unwrap_err();
        assert!(matches!(error, FractionalOctaveSmoothingError::TooFewBins));
    }

    #[test]
    fn magnitude_smoothing_preserves_constant_spectrum() {
        let data = Array2::from_shape_fn((1, 17), |(_, index)| match index % 4 {
            0 => Complex64::new(2.0, 0.0),
            1 => Complex64::new(0.0, 2.0),
            2 => Complex64::new(-2.0, 0.0),
            _ => Complex64::new(0.0, -2.0),
        });
        let signal = FreqSignal::new(data, 48_000.0, Some(32)).unwrap();

        let (smoothed, stats) =
            smooth_fractional_octave(&signal, &FractionalOctaveSmoothingConfig::new(1.0)).unwrap();

        assert_eq!(stats.window_len, 3);
        for (index, (actual, expected)) in smoothed
            .channel(0)
            .iter()
            .zip(signal.channel(0).iter())
            .enumerate()
        {
            if index == 0 || index == signal.num_freq_bins() - 1 {
                assert_abs_diff_eq!(actual.im, 0.0, epsilon = 1e-12);
                assert_abs_diff_eq!(actual.re, 2.0, epsilon = 1e-12);
            } else {
                assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
                assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn zero_phase_mode_returns_real_positive_spectrum() {
        let data = Array2::from_shape_fn((1, 17), |(_, index)| match index % 4 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(1.0, 1.0),
            2 => Complex64::new(-1.0, 1.0),
            _ => Complex64::new(-1.0, 0.0),
        });
        let signal = FreqSignal::new(data, 48_000.0, Some(32)).unwrap();

        let mut config = FractionalOctaveSmoothingConfig::new(1.0);
        config.mode = FractionalOctaveSmoothingMode::MagnitudeZeroPhase;

        let (smoothed, _) = smooth_fractional_octave(&signal, &config).unwrap();

        for value in smoothed.channel(0) {
            assert_abs_diff_eq!(value.im, 0.0, epsilon = 1e-12);
            assert!(value.re >= 0.0);
        }
    }

    #[test]
    fn smoothing_spreads_a_narrow_peak() {
        let signal = FreqSignal::new(
            arr2(&[[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]]),
            48_000.0,
            Some(16),
        )
        .unwrap();

        let (smoothed, _) =
            smooth_fractional_octave(&signal, &FractionalOctaveSmoothingConfig::new(1.0)).unwrap();

        assert!(smoothed.channel(0)[3].re < 1.0);
        assert!(smoothed.channel(0)[2].norm() > 0.0 || smoothed.channel(0)[4].norm() > 0.0);
    }

    #[test]
    fn actual_num_fractions_matches_window_length() {
        let signal = FreqSignal::zeros(1, 65, 48_000.0, Some(128)).unwrap();
        let (smoothed, stats) =
            smooth_fractional_octave(&signal, &FractionalOctaveSmoothingConfig::new(3.0)).unwrap();

        assert_eq!(smoothed.num_freq_bins(), signal.num_freq_bins());
        assert_eq!(smoothed.num_time_steps(), signal.num_time_steps());
        assert_eq!(smoothed.sample_rate(), signal.sample_rate());

        let delta_n = (65.0_f64).log2() / 64.0;
        assert_abs_diff_eq!(
            stats.actual_num_fractions,
            1.0 / (stats.window_len as f64 * delta_n),
            epsilon = 1e-12
        );
    }
}
