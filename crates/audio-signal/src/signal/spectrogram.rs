use ndarray::{prelude::*, s};
use num::complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectrogramNormalization {
    /// Mean window value, used to correct coherent tone amplitude.
    coherent_gain: f64,
    /// Sum of squared window values, used for power / PSD estimates.
    window_energy: f64,
}

impl SpectrogramNormalization {
    pub fn new(coherent_gain: f64, window_energy: f64) -> Self {
        Self {
            coherent_gain,
            window_energy,
        }
    }

    pub fn coherent_gain(&self) -> f64 {
        self.coherent_gain
    }

    pub fn window_energy(&self) -> f64 {
        self.window_energy
    }
}

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
    normalization: SpectrogramNormalization,
    channel_labels: Vec<Option<String>>,
}

impl Spectrogram {
    pub fn new(
        data: Array3<Complex64>,
        frame_times: Array1<f64>,
        freq_bins: Array1<f64>,
        sample_rate: f64,
        window_size: usize,
        hop_size: usize,
        normalization: SpectrogramNormalization,
    ) -> Self {
        let num_channels = data.shape()[0];
        Self {
            data,
            frame_times,
            freq_bins,
            sample_rate,
            window_size,
            hop_size,
            normalization,
            channel_labels: vec![None; num_channels],
        }
    }

    pub fn channel_label(&self, channel: usize) -> Option<&str> {
        self.channel_labels.get(channel).and_then(|s| s.as_deref())
    }

    pub fn set_channel_label(&mut self, channel: usize, label: Option<&str>) {
        if let Some(slot) = self.channel_labels.get_mut(channel) {
            *slot = label.map(Into::into);
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

    pub fn normalization(&self) -> SpectrogramNormalization {
        self.normalization
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
