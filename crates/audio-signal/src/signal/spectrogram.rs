use ndarray::prelude::*;
use num::complex::Complex64;

/// A short-time Fourier transform (STFT) result.
///
/// Data layout: `(num_channels, num_frames, num_freq_bins)`.
/// Use [`channel`](Spectrogram::channel) for per-channel access and
/// [`frame`](Spectrogram::frame) for per-frame access.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Spectrogram {
    /// Shape: (num_channels, num_frames, num_freq_bins)
    data: Array3<Complex64>,
    /// Center time of each frame in seconds
    frame_times: Array1<f64>,
    /// Frequency bins in Hz
    freq_bins: Array1<f64>,
    sample_rate: f64,
    window_size: usize,
    hop_size: usize,
}

impl Spectrogram {
    pub fn new(
        data: Array3<Complex64>,
        frame_times: Array1<f64>,
        freq_bins: Array1<f64>,
        sample_rate: f64,
        window_size: usize,
        hop_size: usize,
    ) -> Self {
        Self {
            data,
            frame_times,
            freq_bins,
            sample_rate,
            window_size,
            hop_size,
        }
    }

    pub fn num_channels(&self) -> usize {
        self.data.shape()[0]
    }

    pub fn num_frames(&self) -> usize {
        self.data.shape()[1]
    }

    pub fn num_freq_bins(&self) -> usize {
        self.data.shape()[2]
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Center time of each STFT frame in seconds.
    pub fn frame_times(&self) -> ArrayView1<'_, f64> {
        self.frame_times.view()
    }

    /// Frequency bin centers in Hz.
    pub fn freq_bins(&self) -> ArrayView1<'_, f64> {
        self.freq_bins.view()
    }

    /// Raw complex STFT data, shape `(num_channels, num_frames, num_freq_bins)`.
    pub fn data(&self) -> ArrayView3<'_, Complex64> {
        self.data.view()
    }

    /// Complex STFT data for a single channel, shape `(num_frames, num_freq_bins)`.
    pub fn channel(&self, ch: usize) -> ArrayView2<'_, Complex64> {
        self.data.slice(s![ch, .., ..])
    }

    /// Complex STFT data for a single frame, shape `(num_channels, num_freq_bins)`.
    pub fn frame(&self, t: usize) -> ArrayView2<'_, Complex64> {
        self.data.slice(s![.., t, ..])
    }

    /// Linear magnitude spectrogram, shape `(num_channels, num_frames, num_freq_bins)`.
    pub fn magnitude(&self) -> Array3<f64> {
        self.data.map(|c| c.norm())
    }

    /// Magnitude spectrogram in dB with a noise floor, shape `(num_channels, num_frames, num_freq_bins)`.
    pub fn magnitude_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 20.0);
        self.data.map(|c| 20.0 * c.norm().max(floor_linear).log10())
    }
}

impl std::fmt::Display for Spectrogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Spectrogram with {} channels, {} frames, {} freq bins at {} Hz (window={}, hop={}).",
            self.num_channels(),
            self.num_frames(),
            self.num_freq_bins(),
            self.sample_rate(),
            self.window_size(),
            self.hop_size(),
        )
    }
}
