use ndarray::{Array2, s};
use ndrustfft::{R2cFftHandler, ndfft_r2c, ndifft_r2c};
use num::complex::Complex64;
use thiserror::Error;

use crate::signal::{SignalError, TimeSignal};

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
    // Power-of-two FFT lengths are much faster than arbitrary (possibly prime)
    // lengths. The extra zero padding does not change the linear convolution.
    let fft_len = conv_len.next_power_of_two();
    let num_freq_bins = fft_len / 2 + 1;
    let num_channels = signal1.num_channels();
    let fft_handler = R2cFftHandler::<f64>::new(fft_len);

    let mut buffer = Array2::<f64>::zeros((num_channels, fft_len));
    let mut spectrum1 = Array2::<Complex64>::zeros((num_channels, num_freq_bins));
    let mut spectrum2 = Array2::<Complex64>::zeros((num_channels, num_freq_bins));

    buffer.slice_mut(s![.., ..n1]).assign(&signal1.time_data());
    ndfft_r2c(&buffer.view(), &mut spectrum1.view_mut(), &fft_handler, 1);

    buffer.slice_mut(s![.., ..n1]).fill(0.0);
    buffer.slice_mut(s![.., ..n2]).assign(&signal2.time_data());
    ndfft_r2c(&buffer.view(), &mut spectrum2.view_mut(), &fft_handler, 1);

    spectrum1 *= &spectrum2;
    ndifft_r2c(&spectrum1.view(), &mut buffer.view_mut(), &fft_handler, 1);

    let out_len = match mode {
        ConvolveMode::Full => conv_len,
        ConvolveMode::Cut | ConvolveMode::Cyclic => n1.max(n2),
    };

    if mode == ConvolveMode::Cyclic {
        // Wrap the tail (out_len..conv_len) back onto the start.
        let wrap_len = conv_len - out_len;
        for mut channel in buffer.outer_iter_mut() {
            for i in 0..wrap_len {
                channel[i] += channel[out_len + i];
            }
        }
    }

    Ok(TimeSignal::new(
        buffer.slice(s![.., ..out_len]).to_owned(),
        signal1.sample_rate(),
    )?)
}

#[cfg(test)]
mod tests {
    use crate::signal::{ImpulseConfig, generate_impulse};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, arr1, arr2};

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
    fn convolve_matches_direct_convolution_for_non_power_of_two_lengths() {
        let n1 = 37;
        let n2 = 13;
        let x: Vec<f64> = (0..n1).map(|i| ((i * 7 % 11) as f64) - 5.0).collect();
        let h: Vec<f64> = (0..n2).map(|i| ((i * 3 % 5) as f64) * 0.5 - 1.0).collect();

        let mut full = vec![0.0; n1 + n2 - 1];
        for (i, &xi) in x.iter().enumerate() {
            for (j, &hj) in h.iter().enumerate() {
                full[i + j] += xi * hj;
            }
        }
        let mut cyclic = full[..n1].to_vec();
        for (i, &tail) in full[n1..].iter().enumerate() {
            cyclic[i] += tail;
        }

        let s1 = TimeSignal::new(Array2::from_shape_vec((1, n1), x).unwrap(), 1.0).unwrap();
        let s2 = TimeSignal::new(Array2::from_shape_vec((1, n2), h).unwrap(), 1.0).unwrap();

        let result = convolve(&s1, &s2, ConvolveMode::Full).unwrap();
        assert_abs_diff_eq!(
            result.channel(0),
            Array1::from(full.clone()),
            epsilon = 1e-9
        );

        let result = convolve(&s1, &s2, ConvolveMode::Cut).unwrap();
        assert_abs_diff_eq!(
            result.channel(0),
            Array1::from(full[..n1].to_vec()),
            epsilon = 1e-9
        );

        let result = convolve(&s2, &s1, ConvolveMode::Cyclic).unwrap();
        assert_abs_diff_eq!(result.channel(0), Array1::from(cyclic), epsilon = 1e-9);
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
