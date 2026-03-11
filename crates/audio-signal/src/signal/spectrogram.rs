use ndarray::prelude::*;
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
        Self {
            data,
            frame_times,
            freq_bins,
            sample_rate,
            window_size,
            hop_size,
            normalization,
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

    fn one_sided_scale(&self, bin: usize) -> f64 {
        let last_bin = self.num_freq_bins().saturating_sub(1);
        if bin == 0 || (self.window_size.is_multiple_of(2) && bin == last_bin) {
            1.0
        } else {
            2.0
        }
    }

    /// Raw FFT magnitude of the windowed STFT frames.
    ///
    /// This is not normalized for FFT size, window attenuation, or one-sided
    /// spectrum interpretation. Use the calibrated helpers below for
    /// measurement-oriented analysis.
    pub fn raw_magnitude(&self) -> Array3<f64> {
        self.data.map(|c| c.norm())
    }

    /// Raw FFT magnitude in dB with a floor, preserving the legacy display semantics.
    pub fn raw_magnitude_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 20.0);
        self.data.map(|c| 20.0 * c.norm().max(floor_linear).log10())
    }

    /// One-sided amplitude spectrum corrected for FFT size and window coherent gain.
    pub fn amplitude_spectrum(&self) -> Array3<f64> {
        let coherent_sum = self.normalization.coherent_gain * self.window_size as f64;
        Array3::from_shape_fn(self.data.raw_dim(), |(ch, frame, bin)| {
            self.data[[ch, frame, bin]].norm() * self.one_sided_scale(bin) / coherent_sum
        })
    }

    pub fn amplitude_spectrum_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 20.0);
        self.amplitude_spectrum()
            .map(|v| 20.0 * v.max(floor_linear).log10())
    }

    /// One-sided power spectrum in units of signal^2 per FFT bin.
    pub fn power_spectrum(&self) -> Array3<f64> {
        let denom = self.window_size as f64 * self.normalization.window_energy;
        Array3::from_shape_fn(self.data.raw_dim(), |(ch, frame, bin)| {
            let mag2 = self.data[[ch, frame, bin]].norm_sqr();
            mag2 * self.one_sided_scale(bin) / denom
        })
    }

    pub fn power_spectrum_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 10.0);
        self.power_spectrum()
            .map(|v| 10.0 * v.max(floor_linear).log10())
    }

    /// One-sided power spectral density estimate in units of signal^2/Hz.
    pub fn power_spectral_density(&self) -> Array3<f64> {
        let denom = self.sample_rate * self.normalization.window_energy;
        Array3::from_shape_fn(self.data.raw_dim(), |(ch, frame, bin)| {
            let mag2 = self.data[[ch, frame, bin]].norm_sqr();
            mag2 * self.one_sided_scale(bin) / denom
        })
    }

    pub fn power_spectral_density_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 10.0);
        self.power_spectral_density()
            .map(|v| 10.0 * v.max(floor_linear).log10())
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
