use std::{f64::consts::PI, ops::Range};

use ndarray::Array1;
use num::complex::Complex64;
use thiserror::Error;

use audio_signal::signal::{FreqSignal, SignalError, TimeSignal};

use crate::{PadMode, pad_zeros};

#[derive(Debug, Error)]
pub enum DeconvolutionError {
    #[error("sample rates do not match: output has {output} Hz, input has {input} Hz")]
    SampleRateMismatch { output: f64, input: f64 },
    #[error(
        "channel counts do not match: output has {output} channels, input has {input} channels"
    )]
    ChannelMismatch { output: usize, input: usize },
    #[error(
        "fft length ({fft_len}) is shorter than signal length (output: {output_len}, input: {input_len})"
    )]
    FftLengthTooShort {
        fft_len: usize,
        output_len: usize,
        input_len: usize,
    },
    #[error(
        "invalid frequency range: {start} Hz to {end} Hz (must satisfy 0 <= start < end <= {nyquist} Hz)"
    )]
    InvalidFrequencyRange { start: f64, end: f64, nyquist: f64 },
    #[error("signals must not be empty")]
    EmptySignal,
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeconvolutionConfig {
    fft_len: Option<usize>,
    regularized_band: Option<Range<f64>>,
    regularization_outside_band: f64,
    regularization_inside_band: f64,
}

impl Default for DeconvolutionConfig {
    fn default() -> Self {
        Self {
            fft_len: None,
            regularized_band: None,
            regularization_outside_band: 1.0,
            regularization_inside_band: 10_f64.powf(-200.0 / 20.0),
        }
    }
}

impl DeconvolutionConfig {
    pub fn with_fft_len(mut self, fft_len: usize) -> Self {
        self.fft_len = Some(fft_len);
        self
    }

    pub fn with_regularized_band(mut self, regularized_band: Range<f64>) -> Self {
        self.regularized_band = Some(regularized_band);
        self
    }

    pub fn with_frequency_range(self, frequency_range: Range<f64>) -> Self {
        self.with_regularized_band(frequency_range)
    }

    pub fn with_regularization(mut self, outside_band: f64, inside_band: f64) -> Self {
        self.regularization_outside_band = outside_band;
        self.regularization_inside_band = inside_band;
        self
    }
}

pub fn deconvolve(
    system_output: &TimeSignal,
    system_input: &TimeSignal,
    config: &DeconvolutionConfig,
) -> Result<TimeSignal, DeconvolutionError> {
    if system_output.sample_rate() != system_input.sample_rate() {
        return Err(DeconvolutionError::SampleRateMismatch {
            output: system_output.sample_rate(),
            input: system_input.sample_rate(),
        });
    }

    if system_output.num_channels() != system_input.num_channels() {
        return Err(DeconvolutionError::ChannelMismatch {
            output: system_output.num_channels(),
            input: system_input.num_channels(),
        });
    }

    let output_len = system_output.num_time_steps();
    let input_len = system_input.num_time_steps();
    if output_len == 0 || input_len == 0 {
        return Err(DeconvolutionError::EmptySignal);
    }

    let fft_len = config.fft_len.unwrap_or(output_len.max(input_len));
    if fft_len < output_len || fft_len < input_len {
        return Err(DeconvolutionError::FftLengthTooShort {
            fft_len,
            output_len,
            input_len,
        });
    }

    let padded_output = pad_zeros(system_output, fft_len - output_len, PadMode::After)?;
    let padded_input = pad_zeros(system_input, fft_len - input_len, PadMode::After)?;

    let output_freq = padded_output.into_freq();
    let inverse_input = regularized_spectrum_inversion(&padded_input.into_freq(), config)?;
    let mut response_freq = FreqSignal::zeros(
        output_freq.num_channels(),
        output_freq.num_freq_bins(),
        output_freq.sample_rate(),
        Some(fft_len),
    )?;

    for ((mut out_channel, output_channel), inverse_channel) in response_freq
        .channel_iter_mut()
        .zip(output_freq.channel_iter())
        .zip(inverse_input.channel_iter())
    {
        out_channel.assign(&(&output_channel * &inverse_channel));
    }

    Ok(response_freq.into_time())
}

fn regularized_spectrum_inversion(
    signal: &FreqSignal,
    config: &DeconvolutionConfig,
) -> Result<FreqSignal, DeconvolutionError> {
    let nyquist = signal.nyquist();
    let regularized_band = config.regularized_band.clone().unwrap_or(0.0..nyquist);

    validate_frequency_range(&regularized_band, nyquist)?;

    let mut regularization =
        Array1::from_elem(signal.num_freq_bins(), config.regularization_inside_band);

    let outside = Array1::from_elem(signal.num_freq_bins(), config.regularization_outside_band);
    let lower_crossfade_start = signal.nearest_freq_index(regularized_band.start / 2_f64.sqrt());
    let lower_crossfade_end = signal.nearest_freq_index(regularized_band.start);
    regularization = crossfade(
        &outside,
        &regularization,
        lower_crossfade_start..lower_crossfade_end,
    );

    if regularized_band.end < nyquist {
        let upper_crossfade_start = signal.nearest_freq_index(regularized_band.end);
        let upper_crossfade_end =
            signal.nearest_freq_index((regularized_band.end * 2_f64.sqrt()).min(nyquist));
        regularization = crossfade(
            &regularization,
            &outside,
            upper_crossfade_start..upper_crossfade_end,
        );
    }

    let max_squared_magnitude = max_squared_magnitude(signal);
    regularization *= max_squared_magnitude;

    let mut inverse = FreqSignal::zeros(
        signal.num_channels(),
        signal.num_freq_bins(),
        signal.sample_rate(),
        Some(signal.num_time_steps()),
    )?;

    for ((mut out_channel, in_channel), reg_channel) in inverse
        .channel_iter_mut()
        .zip(signal.channel_iter())
        .zip(std::iter::repeat(&regularization))
    {
        for ((out_bin, in_bin), reg) in out_channel
            .iter_mut()
            .zip(in_channel.iter())
            .zip(reg_channel.iter())
        {
            *out_bin = in_bin.conj() / (in_bin.conj() * *in_bin + Complex64::from(*reg));
        }
    }

    Ok(inverse)
}

fn validate_frequency_range(range: &Range<f64>, nyquist: f64) -> Result<(), DeconvolutionError> {
    if range.start < 0.0 || range.start >= range.end || range.end > nyquist {
        return Err(DeconvolutionError::InvalidFrequencyRange {
            start: range.start,
            end: range.end,
            nyquist,
        });
    }

    Ok(())
}

fn max_squared_magnitude(signal: &FreqSignal) -> f64 {
    signal
        .freq_data()
        .iter()
        .map(|value| value.norm_sqr())
        .fold(0.0_f64, f64::max)
}

fn crossfade(first: &Array1<f64>, second: &Array1<f64>, range: Range<usize>) -> Array1<f64> {
    assert_eq!(first.len(), second.len());
    assert!(range.start <= first.len());
    assert!(range.end <= first.len());

    let len = first.len();
    let crossfade_len = range.end.saturating_sub(range.start);
    let mut output = Array1::zeros(len);

    for i in 0..range.start {
        output[i] = first[i];
    }

    if crossfade_len > 0 {
        for step in 0..crossfade_len {
            let position = (step + 1) as f64 / (crossfade_len + 1) as f64;
            let second_weight = 0.5 - 0.5 * (PI * position).cos();
            let first_weight = 1.0 - second_weight;
            let idx = range.start + step;
            output[idx] = first[idx] * first_weight + second[idx] * second_weight;
        }
    }

    for i in range.end..len {
        output[i] = second[i];
    }

    output
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};

    use super::*;

    #[test]
    fn deconvolve_recovers_known_response_for_impulse_input() {
        let system_input = TimeSignal::new(arr2(&[[1.0, 0.0, 0.0, 0.0]]), 48_000.0).unwrap();
        let expected_response =
            TimeSignal::new(arr2(&[[0.5, -0.25, 0.125, 0.0]]), 48_000.0).unwrap();

        let recovered = deconvolve(
            &expected_response,
            &system_input,
            &DeconvolutionConfig::default(),
        )
        .unwrap();

        assert_abs_diff_eq!(
            recovered.channel(0),
            expected_response.channel(0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn deconvolve_rejects_sample_rate_mismatch() {
        let system_output = TimeSignal::zeros(1, 16, 44_100.0).unwrap();
        let system_input = TimeSignal::zeros(1, 16, 48_000.0).unwrap();

        let result = deconvolve(
            &system_output,
            &system_input,
            &DeconvolutionConfig::default(),
        );

        assert!(matches!(
            result,
            Err(DeconvolutionError::SampleRateMismatch {
                output: 44_100.0,
                input: 48_000.0
            })
        ));
    }

    #[test]
    fn deconvolve_rejects_channel_mismatch() {
        let system_output = TimeSignal::zeros(2, 16, 48_000.0).unwrap();
        let system_input = TimeSignal::zeros(1, 16, 48_000.0).unwrap();

        let result = deconvolve(
            &system_output,
            &system_input,
            &DeconvolutionConfig::default(),
        );

        assert!(matches!(
            result,
            Err(DeconvolutionError::ChannelMismatch {
                output: 2,
                input: 1
            })
        ));
    }

    #[test]
    fn deconvolve_rejects_fft_lengths_shorter_than_inputs() {
        let system_output = TimeSignal::zeros(1, 16, 48_000.0).unwrap();
        let system_input = TimeSignal::zeros(1, 8, 48_000.0).unwrap();

        let result = deconvolve(
            &system_output,
            &system_input,
            &DeconvolutionConfig::default().with_fft_len(15),
        );

        assert!(matches!(
            result,
            Err(DeconvolutionError::FftLengthTooShort {
                fft_len: 15,
                output_len: 16,
                input_len: 8
            })
        ));
    }

    #[test]
    fn deconvolve_rejects_invalid_frequency_ranges() {
        let system_output = TimeSignal::zeros(1, 16, 48_000.0).unwrap();
        let system_input = TimeSignal::zeros(1, 16, 48_000.0).unwrap();

        let result = deconvolve(
            &system_output,
            &system_input,
            &DeconvolutionConfig::default().with_regularized_band(2_000.0..1_000.0),
        );
        assert!(matches!(
            result,
            Err(DeconvolutionError::InvalidFrequencyRange { .. })
        ));

        let result = deconvolve(
            &system_output,
            &system_input,
            &DeconvolutionConfig::default().with_regularized_band(100.0..30_000.0),
        );
        assert!(matches!(
            result,
            Err(DeconvolutionError::InvalidFrequencyRange { .. })
        ));
    }

    #[test]
    fn crossfade_transitions_between_arrays() {
        let first = arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let second = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0]);

        let output = crossfade(&first, &second, 1..4);

        assert_abs_diff_eq!(output[0], 1.0, epsilon = 1e-12);
        assert!(output[1] < 1.0 && output[1] > output[2]);
        assert_abs_diff_eq!(output[2], 0.5, epsilon = 1e-12);
        assert!(output[3] > 0.0 && output[3] < output[2]);
        assert_abs_diff_eq!(output[4], 0.0, epsilon = 1e-12);
    }
}
