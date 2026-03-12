use audio_signal::signal::{Spectrogram, SpectrogramNormalization, TimeSignal};
use ndarray::prelude::*;
use ndrustfft::{R2cFftHandler, ndfft_r2c};
use num::complex::Complex64;

pub use crate::window::WindowFn;
use crate::window::generate_window;

#[derive(Debug, thiserror::Error)]
pub enum StftError {
    #[error("window_size must be > 0")]
    WindowSizeZero,
    #[error("hop_size must be > 0")]
    HopSizeZero,
    #[error("window_size ({window_size}) must be <= signal length ({signal_length})")]
    WindowLargerThanSignal {
        window_size: usize,
        signal_length: usize,
    },
}

#[derive(Debug, Clone)]
pub struct StftConfig {
    /// FFT window length in samples.
    pub window_size: usize,
    /// Step between successive frames in samples (default: `window_size / 4`, i.e. 75% overlap).
    pub hop_size: usize,
    /// Tapering window applied to each frame before the FFT.
    pub window_fn: WindowFn,
}

impl StftConfig {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            hop_size: window_size / 4,
            window_fn: WindowFn::Hann,
        }
    }
}

impl Default for StftConfig {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Compute the short-time Fourier transform of a [`TimeSignal`].
///
/// Returns a [`Spectrogram`] with shape `(num_channels, num_frames, num_freq_bins)`
/// where `num_freq_bins = window_size / 2 + 1`.
///
/// The number of frames is `(num_samples - window_size) / hop_size + 1` (no padding).
pub fn stft(signal: &TimeSignal, config: &StftConfig) -> Result<Spectrogram, StftError> {
    if config.window_size == 0 {
        return Err(StftError::WindowSizeZero);
    }
    if config.hop_size == 0 {
        return Err(StftError::HopSizeZero);
    }
    let num_samples = signal.num_time_steps();
    if config.window_size > num_samples {
        return Err(StftError::WindowLargerThanSignal {
            window_size: config.window_size,
            signal_length: num_samples,
        });
    }

    let num_channels = signal.num_channels();
    let window_size = config.window_size;
    let hop_size = config.hop_size;
    let num_frames = (num_samples - window_size) / hop_size + 1;
    let num_freq_bins = window_size / 2 + 1;
    let sample_rate = signal.sample_rate();

    let window = generate_window(config.window_fn, window_size);
    let coherent_gain = window.sum() / window_size as f64;
    let window_energy = window.iter().map(|w| w * w).sum();

    let frame_times = Array1::from_iter(
        (0..num_frames).map(|f| (f * hop_size + window_size / 2) as f64 / sample_rate),
    );

    let end_freq = (num_freq_bins - 1) as f64 * sample_rate / window_size as f64;
    let freq_bins = Array1::linspace(0.0, end_freq, num_freq_bins);

    let fft_handler = R2cFftHandler::<f64>::new(window_size);
    let mut data: Array3<Complex64> = Array3::zeros((num_channels, num_frames, num_freq_bins));

    // Reusable buffers
    let mut windowed_frame: Array2<f64> = Array2::zeros((num_channels, window_size));
    let mut frame_freq: Array2<Complex64> = Array2::zeros((num_channels, num_freq_bins));

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        for ch in 0..num_channels {
            let ch_data = signal.channel(ch);
            let wf_row = windowed_frame.row_mut(ch);
            azip!((wf in wf_row, &s in ch_data.slice(s![start..start + window_size]), &w in &window) {
                *wf = s * w;
            });
        }

        ndfft_r2c(
            &windowed_frame.view(),
            &mut frame_freq.view_mut(),
            &fft_handler,
            1,
        );

        data.slice_mut(s![.., frame_idx, ..]).assign(&frame_freq);
    }

    Ok(Spectrogram::new(
        data,
        frame_times,
        freq_bins,
        sample_rate,
        window_size,
        hop_size,
        SpectrogramNormalization::new(coherent_gain, window_energy),
    ))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use audio_signal::signal::TimeSignal;
    use ndarray::arr2;

    use super::*;

    #[test]
    fn stft_errors() {
        let signal = TimeSignal::new(arr2(&[[0.0; 16]]), 16.0).unwrap();

        assert!(matches!(
            stft(
                &signal,
                &StftConfig {
                    window_size: 0,
                    hop_size: 4,
                    window_fn: WindowFn::Hann
                }
            ),
            Err(StftError::WindowSizeZero)
        ));
        assert!(matches!(
            stft(
                &signal,
                &StftConfig {
                    window_size: 8,
                    hop_size: 0,
                    window_fn: WindowFn::Hann
                }
            ),
            Err(StftError::HopSizeZero)
        ));
        assert!(matches!(
            stft(
                &signal,
                &StftConfig {
                    window_size: 32,
                    hop_size: 8,
                    window_fn: WindowFn::Hann
                }
            ),
            Err(StftError::WindowLargerThanSignal { .. })
        ));
    }

    #[test]
    fn stft_dimensions() {
        // 32 samples, window=8, hop=4 → (32-8)/4+1 = 7 frames, 5 freq bins
        let signal = TimeSignal::new(arr2(&[[1.0; 32]]), 16.0).unwrap();
        let config = StftConfig {
            window_size: 8,
            hop_size: 4,
            window_fn: WindowFn::Rectangular,
        };
        let spec = stft(&signal, &config).unwrap();

        assert_eq!(spec.num_channels(), 1);
        assert_eq!(spec.num_frames(), 7);
        assert_eq!(spec.num_freq_bins(), 5);
        assert_eq!(spec.sample_rate(), 16.0);
        assert_eq!(spec.window_size(), 8);
        assert_eq!(spec.hop_size(), 4);
    }

    #[test]
    fn stft_frame_times_and_freq_bins() {
        let signal = TimeSignal::new(arr2(&[[0.0; 16]]), 8.0).unwrap();
        // window=8, hop=4 → (16-8)/4+1 = 3 frames
        let config = StftConfig {
            window_size: 8,
            hop_size: 4,
            window_fn: WindowFn::Rectangular,
        };
        let spec = stft(&signal, &config).unwrap();

        // frame centers: (0*4+4)/8, (1*4+4)/8, (2*4+4)/8 = 0.5, 1.0, 1.5 s
        assert_abs_diff_eq!(spec.frame_times()[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(spec.frame_times()[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spec.frame_times()[2], 1.5, epsilon = 1e-10);

        // freq bins: 0, 1, 2, 3, 4 Hz  (sr=8, N=8 → bin spacing = 8/8 = 1 Hz)
        assert_abs_diff_eq!(spec.freq_bins()[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spec.freq_bins()[4], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn stft_dc_signal() {
        // A constant signal of amplitude 1 with a rectangular window should give
        // DC bin = window_size and all other bins ≈ 0.
        let window_size = 8;
        let signal = TimeSignal::new(arr2(&[[1.0; 32]]), 16.0).unwrap();
        let config = StftConfig {
            window_size,
            hop_size: 4,
            window_fn: WindowFn::Rectangular,
        };
        let spec = stft(&signal, &config).unwrap();

        for frame in 0..spec.num_frames() {
            let ch_data = spec.channel(0);
            let dc = ch_data[[frame, 0]];
            assert_abs_diff_eq!(dc.re, window_size as f64, epsilon = 1e-10);
            assert_abs_diff_eq!(dc.im, 0.0, epsilon = 1e-10);
            for bin in 1..spec.num_freq_bins() {
                assert_abs_diff_eq!(ch_data[[frame, bin]].norm(), 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn stft_multi_channel() {
        let data = arr2(&[[1.0; 32], [2.0; 32]]);
        let signal = TimeSignal::new(data, 16.0).unwrap();
        let config = StftConfig {
            window_size: 8,
            hop_size: 4,
            window_fn: WindowFn::Rectangular,
        };
        let spec = stft(&signal, &config).unwrap();

        assert_eq!(spec.num_channels(), 2);
        // Channel 1 DC should be 2x channel 0 DC
        for frame in 0..spec.num_frames() {
            let dc_ch0 = spec.channel(0)[[frame, 0]].re;
            let dc_ch1 = spec.channel(1)[[frame, 0]].re;
            assert_abs_diff_eq!(dc_ch1, 2.0 * dc_ch0, epsilon = 1e-10);
        }
    }

    #[test]
    fn calibrated_spectra_for_bin_centered_sine() {
        let sample_rate = 8.0;
        let window_size = 8;
        let signal = TimeSignal::new(
            arr2(&[std::array::from_fn::<f64, 8, _>(|n| {
                (std::f64::consts::TAU * 2.0 * n as f64 / window_size as f64).sin()
            })]),
            sample_rate,
        )
        .unwrap();
        let spec = stft(
            &signal,
            &StftConfig {
                window_size,
                hop_size: window_size,
                window_fn: WindowFn::Rectangular,
            },
        )
        .unwrap();

        assert_abs_diff_eq!(spec.raw_magnitude()[[0, 0, 2]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spec.amplitude_spectrum()[[0, 0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(spec.power_spectrum()[[0, 0, 2]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(
            spec.power_spectral_density()[[0, 0, 2]],
            0.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn calibrated_dc_amplitude_is_window_corrected() {
        let signal = TimeSignal::new(arr2(&[[1.0; 8]]), 8.0).unwrap();
        let spec = stft(
            &signal,
            &StftConfig {
                window_size: 8,
                hop_size: 8,
                window_fn: WindowFn::Hann,
            },
        )
        .unwrap();

        assert_abs_diff_eq!(spec.amplitude_spectrum()[[0, 0, 0]], 1.0, epsilon = 1e-10);
    }
}
