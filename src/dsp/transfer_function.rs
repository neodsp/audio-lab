use ndarray::{Array2, s};
use num::Complex;
use thiserror::Error;

use crate::signal::{FreqSignal, SignalError, TimeSignal};

#[derive(Debug, Error)]
pub enum TransferFunctionError {
    #[error("sample rates do not match: output has {output} Hz, input has {input} Hz")]
    SampleRateMismatch { output: f64, input: f64 },
    #[error(
        "the input must have 1 channel or as many as the output ({output}), but it has {input}"
    )]
    ChannelMismatch { output: usize, input: usize },
    #[error("lengths do not match: output has {output} samples, input has {input}")]
    LengthMismatch { output: usize, input: usize },
    #[error("segment length must be between 1 and the signal length ({len}), got {segment_len}")]
    InvalidSegmentLength { segment_len: usize, len: usize },
    #[error("regularization must be finite and >= 0, got {0}")]
    InvalidRegularization(f64),
    #[error("signals must not be empty")]
    EmptySignal,
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferFunctionConfig {
    /// Length of the segments that are averaged, `None` for one segment over the whole signal.
    /// Samples after the last whole segment are ignored. With periodic excitation, use the
    /// period: then every segment holds one whole period and the result is exact.
    pub segment_len: Option<usize>,
    /// Added to the input power of every bin, relative to the highest input power, so bins
    /// where the input has almost no energy don't blow up.
    pub regularization: f64,
}

impl Default for TransferFunctionConfig {
    fn default() -> Self {
        Self {
            segment_len: None,
            regularization: 1e-10,
        }
    }
}

impl TransferFunctionConfig {
    pub fn with_segment_len(mut self, segment_len: usize) -> Self {
        self.segment_len = Some(segment_len);
        self
    }

    pub fn with_regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }
}

/// The transfer function from `system_input` to `system_output`, e.g. from the played noise to
/// each mic: `H = Σ Y·X* / (Σ |X|² + ε)` per frequency bin, summed over segments (the H1
/// estimator).
///
/// Unlike [`deconvolve`](crate::dsp::deconvolve), it doesn't zero-pad: each segment is treated
/// as one period. That makes it exact for periodic excitation such as the looped noise of
/// [`NoiseSource`](crate::signal::measure::NoiseSource), as long as the system's impulse
/// response (including the device latency) is shorter than a segment. A delay shows up as a
/// linear phase. With non-periodic excitation it is an estimate, which more segments improve.
///
/// `system_input` has one channel, used for every channel of `system_output`, or as many
/// channels as `system_output`. Both must be equally long. The result has one channel per
/// output channel and `segment_len / 2 + 1` bins.
pub fn transfer_function(
    system_output: &TimeSignal,
    system_input: &TimeSignal,
    config: &TransferFunctionConfig,
) -> Result<FreqSignal, TransferFunctionError> {
    let sample_rate = system_output.sample_rate();
    if sample_rate != system_input.sample_rate() {
        return Err(TransferFunctionError::SampleRateMismatch {
            output: sample_rate,
            input: system_input.sample_rate(),
        });
    }
    let num_outputs = system_output.num_channels();
    let num_inputs = system_input.num_channels();
    if num_inputs != 1 && num_inputs != num_outputs {
        return Err(TransferFunctionError::ChannelMismatch {
            output: num_outputs,
            input: num_inputs,
        });
    }
    let len = system_output.num_time_steps();
    if len != system_input.num_time_steps() {
        return Err(TransferFunctionError::LengthMismatch {
            output: len,
            input: system_input.num_time_steps(),
        });
    }
    if len == 0 || num_outputs == 0 {
        return Err(TransferFunctionError::EmptySignal);
    }
    let segment_len = config.segment_len.unwrap_or(len);
    if segment_len == 0 || segment_len > len {
        return Err(TransferFunctionError::InvalidSegmentLength { segment_len, len });
    }
    if !config.regularization.is_finite() || config.regularization < 0.0 {
        return Err(TransferFunctionError::InvalidRegularization(
            config.regularization,
        ));
    }

    let num_bins = segment_len / 2 + 1;
    let mut cross = Array2::<Complex<f64>>::zeros((num_outputs, num_bins));
    let mut power = Array2::<f64>::zeros((num_inputs, num_bins));
    for segment in 0..len / segment_len {
        let range = segment * segment_len..(segment + 1) * segment_len;
        let x = slice(system_input, range.clone())?.into_freq();
        let y = slice(system_output, range)?.into_freq();
        for (channel, mut cross) in cross.outer_iter_mut().enumerate() {
            let x = x.channel(if num_inputs == 1 { 0 } else { channel });
            ndarray::Zip::from(&mut cross)
                .and(&y.channel(channel))
                .and(&x)
                .for_each(|c, y, x| *c += y * x.conj());
        }
        for (channel, mut power) in power.outer_iter_mut().enumerate() {
            power.zip_mut_with(&x.channel(channel), |p, x| *p += x.norm_sqr());
        }
    }

    let max_power = power.iter().copied().fold(0.0, f64::max);
    let epsilon = config.regularization * max_power;
    let response = Array2::from_shape_fn((num_outputs, num_bins), |(channel, bin)| {
        let p = power[[if num_inputs == 1 { 0 } else { channel }, bin]] + epsilon;
        if p > 0.0 {
            cross[[channel, bin]] / p
        } else {
            Complex::new(0.0, 0.0)
        }
    });

    let mut result = FreqSignal::new(response, sample_rate, Some(segment_len))?;
    for channel in 0..num_outputs {
        result.set_channel_label(channel, system_output.channel_label(channel));
    }
    Ok(result)
}

fn slice(signal: &TimeSignal, range: std::ops::Range<usize>) -> Result<TimeSignal, SignalError> {
    TimeSignal::new(
        signal.time_data().slice(s![.., range]).to_owned(),
        signal.sample_rate(),
    )
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::signal::{NoiseConfig, Spectrum, generate_noise};

    const SAMPLE_RATE: f64 = 48_000.0;

    fn noise(len: usize, spectrum: Spectrum) -> TimeSignal {
        generate_noise(
            len,
            NoiseConfig {
                spectrum,
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// `periods` periods of `signal` through a delay and an FIR filter, in steady state.
    fn periodic_response(
        signal: &TimeSignal,
        delay: usize,
        taps: &[f64],
        periods: usize,
    ) -> Vec<f64> {
        let x = signal.channel(0);
        let n = x.len();
        (0..n * periods)
            .map(|t| {
                taps.iter()
                    .enumerate()
                    .map(|(k, tap)| tap * x[(t + periods * n - delay - k) % n])
                    .sum()
            })
            .collect()
    }

    fn repeat(signal: &TimeSignal, periods: usize) -> TimeSignal {
        let x = signal.channel(0).to_vec();
        let data: Vec<f64> = x.iter().copied().cycle().take(x.len() * periods).collect();
        TimeSignal::new(
            Array2::from_shape_vec((1, data.len()), data).unwrap(),
            SAMPLE_RATE,
        )
        .unwrap()
    }

    fn expected(delay: usize, taps: &[f64], n: usize, bin: usize) -> Complex<f64> {
        taps.iter()
            .enumerate()
            .map(|(k, tap)| {
                Complex::from_polar(*tap, -TAU * bin as f64 * (delay + k) as f64 / n as f64)
            })
            .sum()
    }

    fn assert_response(h: ndarray::ArrayView1<Complex<f64>>, delay: usize, taps: &[f64], n: usize) {
        for (bin, value) in h.iter().enumerate() {
            let expected = expected(delay, taps, n, bin);
            assert_abs_diff_eq!(value.re, expected.re, epsilon = 1e-6);
            assert_abs_diff_eq!(value.im, expected.im, epsilon = 1e-6);
        }
    }

    #[test]
    fn exact_for_periodic_noise() {
        let n = 1024;
        let taps = [0.5, -0.25, 0.125];
        for spectrum in [Spectrum::White, Spectrum::Pink] {
            let x = noise(n, spectrum);
            for periods in [1, 3] {
                let y = periodic_response(&x, 100, &taps, periods);
                let y = TimeSignal::new(
                    Array2::from_shape_vec((1, y.len()), y).unwrap(),
                    SAMPLE_RATE,
                )
                .unwrap();
                let config = TransferFunctionConfig::default()
                    .with_segment_len(n)
                    .with_regularization(0.0);
                let h = transfer_function(&y, &repeat(&x, periods), &config).unwrap();
                assert_eq!(h.num_freq_bins(), n / 2 + 1);
                assert_eq!(h.num_time_steps(), n);
                assert_response(h.channel(0), 100, &taps, n);
            }
        }
    }

    #[test]
    fn mono_input_is_used_for_every_output_channel() {
        let n = 512;
        let x = noise(n, Spectrum::Pink);
        let a = periodic_response(&x, 10, &[1.0], 2);
        let b = periodic_response(&x, 30, &[0.5, 0.5], 2);
        let data = Array2::from_shape_vec((2, 2 * n), [a, b].concat()).unwrap();
        let mut y = TimeSignal::new(data, SAMPLE_RATE).unwrap();
        y.set_channel_label(1, Some("mic 2"));

        let config = TransferFunctionConfig::default()
            .with_segment_len(n)
            .with_regularization(0.0);
        let h = transfer_function(&y, &repeat(&x, 2), &config).unwrap();
        assert_eq!(h.num_channels(), 2);
        assert_eq!(h.channel_label(1), Some("mic 2"));
        assert_response(h.channel(0), 10, &[1.0], n);
        assert_response(h.channel(1), 30, &[0.5, 0.5], n);
    }

    #[test]
    fn leftover_samples_are_ignored() {
        let n = 256;
        let x = repeat(&noise(n, Spectrum::White), 3);
        let trimmed = |s: &TimeSignal| slice(s, 0..2 * n + 100).unwrap();
        let config = TransferFunctionConfig::default()
            .with_segment_len(n)
            .with_regularization(0.0);
        let h = transfer_function(&trimmed(&x), &trimmed(&x), &config).unwrap();
        assert_response(h.channel(0), 0, &[1.0], n);
    }

    #[test]
    fn silent_input_gives_zero_instead_of_nan() {
        let silence = TimeSignal::zeros(1, 64, SAMPLE_RATE).unwrap();
        let h = transfer_function(&silence, &silence, &TransferFunctionConfig::default()).unwrap();
        assert!(h.iter().all(|v| *v == Complex::new(0.0, 0.0)));
    }

    #[test]
    fn invalid_arguments() {
        let mono = TimeSignal::zeros(1, 64, SAMPLE_RATE).unwrap();
        let stereo = TimeSignal::zeros(2, 64, SAMPLE_RATE).unwrap();
        let three = TimeSignal::zeros(3, 64, SAMPLE_RATE).unwrap();
        let other_rate = TimeSignal::zeros(1, 64, 44_100.0).unwrap();
        let shorter = TimeSignal::zeros(1, 32, SAMPLE_RATE).unwrap();
        let empty = TimeSignal::zeros(1, 0, SAMPLE_RATE).unwrap();
        let default = TransferFunctionConfig::default();
        let tf = |y, x, c: TransferFunctionConfig| transfer_function(y, x, &c).unwrap_err();

        assert!(matches!(
            tf(&mono, &other_rate, default.clone()),
            TransferFunctionError::SampleRateMismatch { .. }
        ));
        assert!(matches!(
            tf(&stereo, &three, default.clone()),
            TransferFunctionError::ChannelMismatch {
                output: 2,
                input: 3
            }
        ));
        assert!(matches!(
            tf(&mono, &shorter, default.clone()),
            TransferFunctionError::LengthMismatch {
                output: 64,
                input: 32
            }
        ));
        assert!(matches!(
            tf(&empty, &empty, default.clone()),
            TransferFunctionError::EmptySignal
        ));
        for segment_len in [0, 65] {
            assert!(matches!(
                tf(&mono, &mono, default.clone().with_segment_len(segment_len)),
                TransferFunctionError::InvalidSegmentLength { len: 64, .. }
            ));
        }
        assert!(matches!(
            tf(&mono, &mono, default.with_regularization(-1.0)),
            TransferFunctionError::InvalidRegularization(_)
        ));
    }
}
