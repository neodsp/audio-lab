use ndarray::s;
use thiserror::Error;

use audio_signal::signal::{FreqSignal, SignalError, TimeSignal};

use crate::{PadMode, pad_zeros};

#[derive(Debug, Error)]
pub enum ConvolveError {
    #[error("sample rates do not match: {left} Hz vs {right} Hz")]
    SampleRateMismatch { left: f64, right: f64 },
    #[error("channel counts do not match: {left} vs {right}")]
    ChannelMismatch { left: usize, right: usize },
    #[error("signals must not be empty")]
    EmptySignal,
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolveMode {
    Full,
    Cut,
    Cyclic,
}

pub fn convolve(
    signal1: &TimeSignal,
    signal2: &TimeSignal,
    mode: ConvolveMode,
) -> Result<TimeSignal, ConvolveError> {
    if signal1.sample_rate() != signal2.sample_rate() {
        return Err(ConvolveError::SampleRateMismatch {
            left: signal1.sample_rate(),
            right: signal2.sample_rate(),
        });
    }

    if signal1.num_channels() != signal2.num_channels() {
        return Err(ConvolveError::ChannelMismatch {
            left: signal1.num_channels(),
            right: signal2.num_channels(),
        });
    }

    let n1 = signal1.num_time_steps();
    let n2 = signal2.num_time_steps();
    if n1 == 0 || n2 == 0 {
        return Err(ConvolveError::EmptySignal);
    }

    let conv_len = n1 + n2 - 1;
    let padded1 = pad_zeros(signal1, conv_len - n1, PadMode::After)?;
    let padded2 = pad_zeros(signal2, conv_len - n2, PadMode::After)?;

    let freq1 = padded1.into_freq();
    let freq2 = padded2.into_freq();
    let mut result_freq = FreqSignal::zeros(
        freq1.num_channels(),
        freq1.num_freq_bins(),
        freq1.sample_rate(),
        Some(conv_len),
    )?;

    for ((mut out_channel, left_channel), right_channel) in result_freq
        .channel_iter_mut()
        .zip(freq1.channel_iter())
        .zip(freq2.channel_iter())
    {
        out_channel.assign(&(&left_channel * &right_channel));
    }

    let result = result_freq.into_time();

    match mode {
        ConvolveMode::Full => Ok(result),
        ConvolveMode::Cut => truncate(&result, n1.max(n2)),
        ConvolveMode::Cyclic => {
            let mut wrapped = result;
            let wrap_len = n1.min(n2) - 1;

            if wrap_len > 0 {
                for mut channel in wrapped.channel_iter_mut() {
                    let tail = channel.slice(s![conv_len - wrap_len..]).to_owned();
                    for (sample, wrapped_tail) in channel.iter_mut().take(wrap_len).zip(tail.iter())
                    {
                        *sample += wrapped_tail;
                    }
                }
            }

            truncate(&wrapped, n1.max(n2))
        }
    }
}

fn truncate(signal: &TimeSignal, len: usize) -> Result<TimeSignal, ConvolveError> {
    Ok(TimeSignal::new(
        signal.time_data().slice(s![.., ..len]).to_owned(),
        signal.sample_rate(),
    )?)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use audio_signal::{ImpulseConfig, generate_impulse};
    use ndarray::{arr1, arr2};

    use super::*;

    #[test]
    fn convolve_full_matches_known_result() {
        let s1 = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0]]), 1.0).unwrap();
        let s2 = TimeSignal::new(arr2(&[[1.0, 1.0]]), 1.0).unwrap();

        let result = convolve(&s1, &s2, ConvolveMode::Full).unwrap();

        assert_eq!(result.num_time_steps(), 4);
        assert_abs_diff_eq!(
            result.channel(0),
            arr1(&[1.0, 3.0, 5.0, 3.0]),
            epsilon = 1e-10
        );
    }

    #[test]
    fn convolve_cut_truncates_to_longer_input() {
        let s1 = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0]]), 1.0).unwrap();
        let s2 = TimeSignal::new(arr2(&[[1.0, 1.0]]), 1.0).unwrap();

        let result = convolve(&s1, &s2, ConvolveMode::Cut).unwrap();

        assert_eq!(result.num_time_steps(), 3);
        assert_abs_diff_eq!(result.channel(0), arr1(&[1.0, 3.0, 5.0]), epsilon = 1e-10);
    }

    #[test]
    fn convolve_cyclic_wraps_tail_onto_start() {
        let s1 = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0]]), 1.0).unwrap();
        let s2 = TimeSignal::new(arr2(&[[1.0, 1.0]]), 1.0).unwrap();

        let result = convolve(&s1, &s2, ConvolveMode::Cyclic).unwrap();

        assert_eq!(result.num_time_steps(), 3);
        assert_abs_diff_eq!(result.channel(0), arr1(&[4.0, 3.0, 5.0]), epsilon = 1e-10);
    }

    #[test]
    fn convolve_rejects_sample_rate_mismatch() {
        let s1 = TimeSignal::new(arr2(&[[1.0, 2.0]]), 44_100.0).unwrap();
        let s2 = TimeSignal::new(arr2(&[[1.0, 1.0]]), 48_000.0).unwrap();

        let result = convolve(&s1, &s2, ConvolveMode::Full);

        assert!(matches!(
            result,
            Err(ConvolveError::SampleRateMismatch {
                left: 44_100.0,
                right: 48_000.0
            })
        ));
    }

    #[test]
    fn convolve_rejects_channel_mismatch() {
        let mono = TimeSignal::new(arr2(&[[1.0, 2.0]]), 1.0).unwrap();
        let stereo = TimeSignal::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]), 1.0).unwrap();

        let result = convolve(&mono, &stereo, ConvolveMode::Full);

        assert!(matches!(
            result,
            Err(ConvolveError::ChannelMismatch { left: 1, right: 2 })
        ));
    }

    #[test]
    fn convolve_applies_same_channel_processing_for_multichannel_input() {
        let s1 = TimeSignal::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]), 1.0).unwrap();
        let s2 = TimeSignal::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), 1.0).unwrap();

        let result = convolve(&s1, &s2, ConvolveMode::Full).unwrap();

        assert_eq!(result.num_channels(), 2);
        assert_eq!(result.num_time_steps(), 3);
        assert_abs_diff_eq!(result.channel(0), arr1(&[1.0, 2.0, 0.0]), epsilon = 1e-10);
        assert_abs_diff_eq!(result.channel(1), arr1(&[0.0, 3.0, 4.0]), epsilon = 1e-10);
    }

    #[test]
    fn convolve_rejects_empty_input() {
        let empty = TimeSignal::zeros(1, 0, 1.0).unwrap();
        let signal = TimeSignal::new(arr2(&[[1.0, 2.0]]), 1.0).unwrap();

        let result = convolve(&empty, &signal, ConvolveMode::Full);

        assert!(matches!(result, Err(ConvolveError::EmptySignal)));
    }

    #[test]
    fn convolving_with_generated_delta_impulse_is_identity() {
        let signal = TimeSignal::new(arr2(&[[0.25, -0.5, 1.0, 0.75, -0.125]]), 48_000.0).unwrap();
        let impulse = generate_impulse(
            1,
            ImpulseConfig {
                sample_rate: 48_000.0,
                ..Default::default()
            },
        )
        .unwrap();

        let result = convolve(&signal, &impulse, ConvolveMode::Full).unwrap();

        assert_eq!(result.num_time_steps(), signal.num_time_steps());
        assert_abs_diff_eq!(result.channel(0), signal.channel(0), epsilon = 1e-10);
    }
}
