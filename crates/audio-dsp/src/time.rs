use std::time::Duration;

use ndarray::s;
use rubato::{Fft, FixedSync, ResampleError, Resampler, ResamplerConstructionError};

use audio_signal::math::db_to_gain;
use audio_signal::signal::{SignalError, TimeSignal};
use audioadapter_buffers::direct::SequentialSlice;

use crate::apply_hann_right;

#[derive(Debug, thiserror::Error)]
pub enum TimeError {
    #[error("invalid trim range: start={start}, end={end}, signal_len={signal_len}")]
    InvalidTrimRange {
        start: usize,
        end: usize,
        signal_len: usize,
    },
    #[error("invalid analysis threshold: {threshold_db} dB")]
    InvalidThreshold { threshold_db: f64 },
    #[error(
        "impulse response start could not be detected in channel {channel}: peak power {peak_power} below threshold {required_peak_power} from noise floor {noise_power}"
    )]
    LowSnr {
        channel: usize,
        peak_power: f64,
        noise_power: f64,
        required_peak_power: f64,
    },
    #[error("trimmed window duration must map to at least one sample, got {duration_seconds} s")]
    InvalidWindowDuration { duration_seconds: f64 },
    #[error("invalid sample rate for resampling: {sample_rate} Hz")]
    InvalidResampleRate { sample_rate: f64 },
    #[error("shift vector length {got} does not match channel count {expected}")]
    InvalidShiftCount { expected: usize, got: usize },
    #[error("linear shift magnitude {shift} exceeds signal length {signal_len}")]
    ShiftExceedsSignal { shift: isize, signal_len: usize },
    #[error(transparent)]
    ResamplerConstruction(#[from] ResamplerConstructionError),
    #[error(transparent)]
    Resampler(#[from] ResampleError),
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeShiftMode {
    Linear,
    Cyclic,
}

pub fn trim_samples(
    signal: &TimeSignal,
    start_sample: usize,
    end_sample: i64,
) -> Result<TimeSignal, TimeError> {
    let signal_len = signal.num_time_steps();
    let end_sample = if end_sample < 0 {
        signal_len.saturating_sub((-end_sample) as usize)
    } else {
        end_sample as usize
    };

    if start_sample >= end_sample || end_sample > signal_len {
        return Err(TimeError::InvalidTrimRange {
            start: start_sample,
            end: end_sample,
            signal_len,
        });
    }

    Ok(TimeSignal::new(
        signal
            .time_data()
            .slice(s![.., start_sample..end_sample])
            .to_owned(),
        signal.sample_rate(),
    )?)
}

pub fn trim_duration(
    signal: &TimeSignal,
    start_duration: Duration,
    end_duration: Duration,
) -> Result<TimeSignal, TimeError> {
    let start_sample = (signal.sample_rate() * start_duration.as_secs_f64()).round() as usize;
    let end_sample = (signal.sample_rate() * end_duration.as_secs_f64()).round() as i64;
    trim_samples(signal, start_sample, end_sample)
}

pub fn apply_gain(signal: &TimeSignal, gain: f64) -> Result<TimeSignal, TimeError> {
    Ok(TimeSignal::new(
        signal.time_data().mapv(|sample| sample * gain),
        signal.sample_rate(),
    )?)
}

pub fn apply_gain_db(signal: &TimeSignal, gain_db: f64) -> Result<TimeSignal, TimeError> {
    apply_gain(signal, db_to_gain(gain_db))
}

pub fn time_shift(
    signal: &TimeSignal,
    shift_samples: isize,
    mode: TimeShiftMode,
    pad_value: f64,
) -> Result<TimeSignal, TimeError> {
    let shifts = vec![shift_samples; signal.num_channels()];
    time_shift_per_channel(signal, &shifts, mode, pad_value)
}

pub fn time_shift_per_channel(
    signal: &TimeSignal,
    shift_samples: &[isize],
    mode: TimeShiftMode,
    pad_value: f64,
) -> Result<TimeSignal, TimeError> {
    if shift_samples.len() != signal.num_channels() {
        return Err(TimeError::InvalidShiftCount {
            expected: signal.num_channels(),
            got: shift_samples.len(),
        });
    }

    let signal_len = signal.num_time_steps();
    if let Some(shift) = shift_samples
        .iter()
        .copied()
        .find(|&shift| matches!(mode, TimeShiftMode::Linear) && shift.unsigned_abs() > signal_len)
    {
        return Err(TimeError::ShiftExceedsSignal { shift, signal_len });
    }

    let mut shifted = signal.clone();
    for (channel_index, mut channel) in shifted.channel_iter_mut().enumerate() {
        let shift = shift_samples[channel_index];
        rotate_channel(
            channel.as_slice_mut().expect("channel view is contiguous"),
            shift,
        );

        if matches!(mode, TimeShiftMode::Linear) {
            if shift > 0 {
                channel.slice_mut(s![..shift as usize]).fill(pad_value);
            } else if shift < 0 {
                let start = signal_len - shift.unsigned_abs();
                channel.slice_mut(s![start..]).fill(pad_value);
            }
        }
    }

    Ok(shifted)
}

pub fn find_impulse_response_start(
    impulse_response: &TimeSignal,
    threshold_db: Option<f64>,
) -> Result<Vec<usize>, TimeError> {
    let threshold_db = threshold_db.unwrap_or(20.0);
    if !threshold_db.is_finite() || threshold_db < 0.0 {
        return Err(TimeError::InvalidThreshold { threshold_db });
    }

    let num_time_steps = impulse_response.num_time_steps();
    let mask_start = ((0.9 * num_time_steps as f64) as usize).min(num_time_steps);
    let analysis_len = num_time_steps.saturating_sub(mask_start);
    let snr_ratio = 10.0_f64.powf(threshold_db / 10.0);
    let relative_threshold = 10.0_f64.powf(-threshold_db / 10.0);

    let mut start_samples = Vec::with_capacity(impulse_response.num_channels());
    for (channel_index, channel) in impulse_response.channel_iter().enumerate() {
        let power: Vec<f64> = channel.iter().map(|sample| sample.abs().powi(2)).collect();
        let noise_power = if analysis_len == 0 {
            0.0
        } else {
            power.iter().skip(mask_start).sum::<f64>() / analysis_len as f64
        };

        let (peak_index, peak_power) = power
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));

        let required_peak_power = noise_power * snr_ratio;
        if peak_power < required_peak_power || peak_index > mask_start {
            return Err(TimeError::LowSnr {
                channel: channel_index,
                peak_power,
                noise_power,
                required_peak_power,
            });
        }

        let first_above_threshold =
            power
                .iter()
                .take(peak_index + 1)
                .enumerate()
                .find_map(|(index, value)| {
                    if peak_power > 0.0 && (*value / peak_power) >= relative_threshold {
                        Some(index)
                    } else {
                        None
                    }
                });

        let start_sample = first_above_threshold
            .map(|index| index.saturating_sub(1))
            .unwrap_or(0);
        start_samples.push(start_sample);
    }

    Ok(start_samples)
}

pub fn window_and_trim(
    impulse_response: &TimeSignal,
    duration_seconds: f64,
) -> Result<(TimeSignal, Vec<usize>), TimeError> {
    let len = (duration_seconds * impulse_response.sample_rate()).ceil() as usize;
    if len == 0 {
        return Err(TimeError::InvalidWindowDuration { duration_seconds });
    }

    let starts = find_impulse_response_start(impulse_response, None)?;
    let mut windowed = TimeSignal::zeros(
        impulse_response.num_channels(),
        len,
        impulse_response.sample_rate(),
    )?;

    for ((mut output_channel, input_channel), &start) in windowed
        .channel_iter_mut()
        .zip(impulse_response.channel_iter())
        .zip(starts.iter())
    {
        output_channel
            .iter_mut()
            .zip(input_channel.iter().skip(start).take(len))
            .for_each(|(dst, src)| *dst = *src);

        if let Some(slice) = output_channel.as_slice_mut() {
            apply_hann_right(slice, 0, len);
        }
    }

    Ok((windowed, starts))
}

pub fn resample(signal: &TimeSignal, new_sample_rate: f64) -> Result<TimeSignal, TimeError> {
    if (new_sample_rate - signal.sample_rate()).abs() < 1e-9 {
        return Ok(signal.clone());
    }

    let sample_rate_in = validated_resample_rate(signal.sample_rate())?;
    let sample_rate_out = validated_resample_rate(new_sample_rate)?;
    let num_channels = signal.num_channels();
    let num_input_frames = signal.num_time_steps();

    let mut resampler = Fft::<f64>::new(
        sample_rate_in,
        sample_rate_out,
        1024,
        2,
        num_channels,
        FixedSync::Both,
    )?;

    let input_sequential: Vec<f64> = signal.time_data().iter().copied().collect();
    let buffer_in = SequentialSlice::new(&input_sequential, num_channels, num_input_frames)
        .expect("sequential input length matches channel/frame dimensions");

    let num_output_frames = resampler.process_all_needed_output_len(num_input_frames);
    let mut output_sequential = vec![0.0; num_output_frames * num_channels];
    let mut buffer_out =
        SequentialSlice::new_mut(&mut output_sequential, num_channels, num_output_frames)
            .expect("allocated output length matches channel/frame dimensions");

    let (_, actual_output_frames) =
        resampler.process_all_into_buffer(&buffer_in, &mut buffer_out, num_input_frames, None)?;

    output_sequential.truncate(actual_output_frames * num_channels);
    let data =
        ndarray::Array2::from_shape_vec((num_channels, actual_output_frames), output_sequential)
            .expect("resampler output length matches channel/frame dimensions");

    TimeSignal::new(data, new_sample_rate).map_err(TimeError::from)
}

fn validated_resample_rate(sample_rate: f64) -> Result<usize, TimeError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(TimeError::InvalidResampleRate { sample_rate });
    }

    let rounded = sample_rate.round();
    if (sample_rate - rounded).abs() > 1e-9 {
        return Err(TimeError::InvalidResampleRate { sample_rate });
    }

    Ok(rounded as usize)
}

fn rotate_channel(channel: &mut [f64], shift: isize) {
    if channel.is_empty() {
        return;
    }

    let normalized_shift = shift.rem_euclid(channel.len() as isize) as usize;
    channel.rotate_right(normalized_shift);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use approx::assert_abs_diff_eq;
    use audio_signal::{SineConfig, generate_sine};
    use ndarray::arr2;

    use super::*;

    #[test]
    fn trim_samples_accepts_positive_and_negative_end() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0, 3.0, 4.0]]), 10.0).unwrap();

        let trimmed = trim_samples(&signal, 1, 4).unwrap();
        assert_eq!(trimmed.num_time_steps(), 3);
        assert_abs_diff_eq!(trimmed.channel(0)[0], 1.0, epsilon = 1e-12);

        let trimmed = trim_samples(&signal, 1, -1).unwrap();
        assert_eq!(trimmed.num_time_steps(), 3);
        assert_abs_diff_eq!(trimmed.channel(0)[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn trim_duration_maps_to_sample_positions() {
        let signal = TimeSignal::new(arr2(&[[0.0; 20]]), 10.0).unwrap();
        let trimmed = trim_duration(
            &signal,
            Duration::from_secs_f64(0.2),
            Duration::from_secs_f64(1.1),
        )
        .unwrap();

        assert_eq!(trimmed.num_time_steps(), 9);
    }

    #[test]
    fn trim_rejects_invalid_ranges() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0]]), 1.0).unwrap();
        let result = trim_samples(&signal, 2, 2);

        assert!(matches!(result, Err(TimeError::InvalidTrimRange { .. })));
    }

    #[test]
    fn apply_gain_scales_samples_linearly() {
        let signal = TimeSignal::new(arr2(&[[0.25, -0.5, 1.0]]), 48_000.0).unwrap();

        let gained = apply_gain(&signal, 2.0).unwrap();

        assert_abs_diff_eq!(gained.channel(0)[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(gained.channel(0)[1], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(gained.channel(0)[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn apply_gain_db_converts_db_to_linear_gain() {
        let signal = TimeSignal::new(arr2(&[[0.25, -0.5, 1.0]]), 48_000.0).unwrap();

        let gained = apply_gain_db(&signal, -6.0).unwrap();
        let expected_gain = db_to_gain(-6.0);

        assert_abs_diff_eq!(gained.channel(0)[0], 0.25 * expected_gain, epsilon = 1e-12);
        assert_abs_diff_eq!(gained.channel(0)[1], -0.5 * expected_gain, epsilon = 1e-12);
        assert_abs_diff_eq!(gained.channel(0)[2], expected_gain, epsilon = 1e-12);
    }

    #[test]
    fn time_shift_cyclic_wraps_signal() {
        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0, 4.0]]), 48_000.0).unwrap();

        let shifted = time_shift(&signal, 1, TimeShiftMode::Cyclic, 0.0).unwrap();

        assert_abs_diff_eq!(
            shifted.channel(0),
            ndarray::arr1(&[4.0, 1.0, 2.0, 3.0]),
            epsilon = 1e-12
        );
    }

    #[test]
    fn time_shift_linear_pads_exposed_samples() {
        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0, 4.0]]), 48_000.0).unwrap();

        let shifted = time_shift(&signal, -2, TimeShiftMode::Linear, 0.0).unwrap();

        assert_abs_diff_eq!(
            shifted.channel(0),
            ndarray::arr1(&[3.0, 4.0, 0.0, 0.0]),
            epsilon = 1e-12
        );
    }

    #[test]
    fn time_shift_accepts_per_channel_shifts() {
        let signal =
            TimeSignal::new(arr2(&[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]), 48_000.0).unwrap();

        let shifted =
            time_shift_per_channel(&signal, &[1, -1], TimeShiftMode::Linear, -1.0).unwrap();

        assert_abs_diff_eq!(
            shifted.channel(0),
            ndarray::arr1(&[-1.0, 1.0, 2.0]),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            shifted.channel(1),
            ndarray::arr1(&[20.0, 30.0, -1.0]),
            epsilon = 1e-12
        );
    }

    #[test]
    fn find_impulse_response_start_detects_onset_per_channel() {
        let signal = TimeSignal::new(
            arr2(&[
                [0.0, 0.0, 0.1, 0.4, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.2, 0.6, 1.0, 0.2, 0.0, 0.0, 0.0],
            ]),
            10.0,
        )
        .unwrap();

        let starts = find_impulse_response_start(&signal, Some(20.0)).unwrap();

        assert_eq!(starts, vec![1, 2]);
    }

    #[test]
    fn find_impulse_response_start_rejects_low_snr() {
        let signal = TimeSignal::new(
            arr2(&[[0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01]]),
            10.0,
        )
        .unwrap();

        let result = find_impulse_response_start(&signal, Some(20.0));

        assert!(matches!(result, Err(TimeError::LowSnr { channel: 0, .. })));
    }

    #[test]
    fn window_and_trim_uses_detected_onset_and_applies_fade_out() {
        let signal = TimeSignal::new(
            arr2(&[[0.0, 0.0, 0.1, 0.4, 1.0, 0.5, 0.25, 0.0, 0.0, 0.0]]),
            10.0,
        )
        .unwrap();

        let (trimmed, starts) = window_and_trim(&signal, 0.5).unwrap();

        assert_eq!(starts, vec![1]);
        assert_eq!(trimmed.num_time_steps(), 5);
        assert_abs_diff_eq!(trimmed.channel(0)[0], 0.0, epsilon = 1e-12);
        assert!(trimmed.channel(0)[1] > 0.0);
        assert!(trimmed.channel(0)[4].abs() < trimmed.channel(0)[3].abs());
    }

    #[test]
    fn window_and_trim_rejects_zero_duration() {
        let signal = TimeSignal::new(arr2(&[[1.0, 0.0, 0.0]]), 10.0).unwrap();

        let result = window_and_trim(&signal, 0.0);

        assert!(matches!(
            result,
            Err(TimeError::InvalidWindowDuration { .. })
        ));
    }

    #[test]
    fn resample_preserves_duration_and_frequency() {
        let signal = generate_sine(
            4_410,
            440.0,
            &SineConfig {
                amplitude: 0.5,
                sample_rate: 44_100.0,
                num_channels: 1,
            },
        )
        .unwrap();

        let resampled = resample(&signal, 48_000.0).unwrap();

        assert_eq!(resampled.sample_rate(), 48_000.0);
        assert_abs_diff_eq!(
            resampled.length_in_seconds(),
            signal.length_in_seconds(),
            epsilon = 1e-3
        );

        let (peak_freq, _) = resampled.into_freq().to_magnitude().max_per_channel()[0];
        assert_abs_diff_eq!(peak_freq, 440.0, epsilon = 1.0);
    }

    #[test]
    fn resample_rejects_non_integer_target_rate() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 0.0]]), 48_000.0).unwrap();

        let result = resample(&signal, 48_000.5);

        assert!(matches!(
            result,
            Err(TimeError::InvalidResampleRate {
                sample_rate: 48_000.5
            })
        ));
    }

    #[test]
    fn resample_short_circuits_when_sample_rate_is_unchanged() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 0.0, -1.0]]), 48_000.0).unwrap();

        let resampled = resample(&signal, 48_000.0).unwrap();

        assert_eq!(resampled, signal);
    }
}
