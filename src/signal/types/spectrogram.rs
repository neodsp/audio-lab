use ndarray::{prelude::*, s};
use num::complex::Complex64;

use super::{SignalError, validate_sample_rate};
use crate::dsp::window::{WindowFn, generate_window};
use crate::signal::data::XAxis;

/// Window gains used to calibrate spectra. Always computed from the window, see
/// [`Spectrogram::normalization`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrogramNormalization {
    /// Mean window value, used to correct coherent tone amplitude.
    coherent_gain: f64,
    /// Sum of squared window values, used for power / PSD estimates.
    window_energy: f64,
}

impl SpectrogramNormalization {
    /// Gains of `window_fn` at `window_size` samples, or `None` if that window
    /// is all zeros (a Hann or Blackman window of 2 samples). Calibrated
    /// spectra divide by these gains, so a zero window cannot be used.
    pub(crate) fn of_window(window_fn: WindowFn, window_size: usize) -> Option<Self> {
        let window = generate_window(window_fn, window_size);
        let coherent_gain = window.sum() / window_size as f64;
        let window_energy = window.iter().map(|w| w * w).sum();
        // Blackman's zeros come out as about -1.4e-17, so compare with a tolerance.
        (coherent_gain > f64::EPSILON).then_some(Self {
            coherent_gain,
            window_energy,
        })
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
///
/// Frame times and frequency bins are derived from the sample rate, window
/// size and hop size, so they always match the data.
#[derive(Clone, Debug, PartialEq)]
pub struct Spectrogram {
    /// Shape: (num_channels, num_frames, num_freq_bins)
    data: Array3<Complex64>,
    /// Center time of each frame in seconds
    frame_axis: XAxis,
    /// Frequency bins in Hz
    freq_axis: XAxis,
    sample_rate: f64,
    window_size: usize,
    hop_size: usize,
    window_fn: WindowFn,
    /// Derived from `window_fn` and `window_size`, cached.
    normalization: SpectrogramNormalization,
    channel_labels: Vec<Option<String>>,
}

impl Spectrogram {
    /// Wrap STFT data, shape `(num_channels, num_frames, num_freq_bins)`.
    ///
    /// Frame `f` is centred at `(f * hop_size + window_size / 2) / sample_rate`
    /// seconds and bin `k` sits at `k * sample_rate / window_size` Hz. The data
    /// must have `window_size / 2 + 1` bins, as a real FFT of that size gives.
    /// The calibration gains are computed from `window_fn`, the window the
    /// frames were tapered with.
    pub fn new(
        data: Array3<Complex64>,
        sample_rate: f64,
        window_size: usize,
        hop_size: usize,
        window_fn: WindowFn,
    ) -> Result<Self, SignalError> {
        validate_sample_rate(sample_rate)?;
        if window_size == 0 {
            return Err(SignalError::WindowSizeZero);
        }
        if hop_size == 0 {
            return Err(SignalError::HopSizeZero);
        }
        let (num_channels, num_frames, num_freq_bins) = data.dim();
        let expected_bins = window_size / 2 + 1;
        if num_freq_bins != expected_bins {
            return Err(SignalError::SpectrogramBinMismatch {
                window_size,
                expected: expected_bins,
                actual: num_freq_bins,
            });
        }
        let normalization = SpectrogramNormalization::of_window(window_fn, window_size).ok_or(
            SignalError::DegenerateWindow {
                window_fn,
                window_size,
            },
        )?;

        let frame_axis = XAxis::uniform(
            (window_size / 2) as f64 / sample_rate,
            hop_size as f64 / sample_rate,
            num_frames,
        )
        .expect("validated sample rate and hop size give a finite, positive step");
        let freq_axis = XAxis::uniform(0.0, sample_rate / window_size as f64, num_freq_bins)
            .expect("validated sample rate and window size give a finite, positive step");

        Ok(Self {
            data,
            frame_axis,
            freq_axis,
            sample_rate,
            window_size,
            hop_size,
            window_fn,
            normalization,
            channel_labels: vec![None; num_channels],
        })
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

    /// The window the frames were tapered with.
    pub fn window_fn(&self) -> WindowFn {
        self.window_fn
    }

    /// Calibration gains of [`window_fn`](Self::window_fn) at
    /// [`window_size`](Self::window_size) samples.
    pub fn normalization(&self) -> SpectrogramNormalization {
        self.normalization
    }

    /// Center time of each STFT frame in seconds. Computed on each call; use
    /// [`frame_axis`](Self::frame_axis) for single values.
    pub fn frame_times(&self) -> CowArray<'_, f64, Ix1> {
        self.frame_axis.to_cow_array()
    }

    /// Frame time axis in seconds, without allocating.
    pub fn frame_axis(&self) -> &XAxis {
        &self.frame_axis
    }

    /// Frequency bin centers in Hz. Computed on each call; use
    /// [`freq_axis`](Self::freq_axis) for single values.
    pub fn freq_bins(&self) -> CowArray<'_, f64, Ix1> {
        self.freq_axis.to_cow_array()
    }

    /// Frequency axis in Hz, without allocating.
    pub fn freq_axis(&self) -> &XAxis {
        &self.freq_axis
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

/// Stored form: the data plus what cannot be derived from it. Frame times,
/// frequency bins and calibration gains are rebuilt when reading, through
/// [`Spectrogram::new`].
#[cfg(feature = "serde")]
impl serde::Serialize for Spectrogram {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        struct Stored<'a> {
            data: ArrayView3<'a, Complex64>,
            sample_rate: f64,
            window_size: usize,
            hop_size: usize,
            window_fn: WindowFn,
            channel_labels: &'a [Option<String>],
        }

        Stored {
            data: self.data(),
            sample_rate: self.sample_rate,
            window_size: self.window_size,
            hop_size: self.hop_size,
            window_fn: self.window_fn,
            channel_labels: &self.channel_labels,
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Spectrogram {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        struct Stored {
            data: Array3<Complex64>,
            sample_rate: f64,
            window_size: usize,
            hop_size: usize,
            window_fn: WindowFn,
            channel_labels: Vec<Option<String>>,
        }

        let stored = Stored::deserialize(deserializer)?;
        let mut spectrogram = Spectrogram::new(
            stored.data,
            stored.sample_rate,
            stored.window_size,
            stored.hop_size,
            stored.window_fn,
        )
        .map_err(Error::custom)?;
        if stored.channel_labels.len() != spectrogram.num_channels() {
            return Err(Error::custom(
                crate::signal::data::DataError::ChannelLabelMismatch {
                    labels: stored.channel_labels.len(),
                    channels: spectrogram.num_channels(),
                },
            ));
        }
        spectrogram.channel_labels = stored.channel_labels;
        Ok(spectrogram)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn data(window_size: usize, num_frames: usize) -> Array3<Complex64> {
        Array3::from_shape_fn((2, num_frames, window_size / 2 + 1), |(ch, f, b)| {
            Complex64::new((ch + 1) as f64 * b as f64, f as f64)
        })
    }

    #[test]
    fn new_rejects_inconsistent_parameters() {
        let result = Spectrogram::new(data(8, 3), 0.0, 8, 4, WindowFn::Hann);
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));

        let result = Spectrogram::new(data(8, 3), 8.0, 0, 4, WindowFn::Hann);
        assert!(matches!(result, Err(SignalError::WindowSizeZero)));

        let result = Spectrogram::new(data(8, 3), 8.0, 8, 0, WindowFn::Hann);
        assert!(matches!(result, Err(SignalError::HopSizeZero)));

        // 5 bins fit windows of 8 or 9 samples, not 16.
        assert!(Spectrogram::new(data(8, 3), 8.0, 9, 4, WindowFn::Hann).is_ok());
        let result = Spectrogram::new(data(8, 3), 8.0, 16, 4, WindowFn::Hann);
        assert!(matches!(
            result,
            Err(SignalError::SpectrogramBinMismatch {
                window_size: 16,
                expected: 9,
                actual: 5
            })
        ));
    }

    #[test]
    fn normalization_is_computed_from_the_window() {
        for window_fn in [
            WindowFn::Hann,
            WindowFn::Hamming,
            WindowFn::Blackman,
            WindowFn::Rectangular,
        ] {
            let spec = Spectrogram::new(data(8, 3), 8.0, 8, 4, window_fn).unwrap();
            let window = generate_window(window_fn, 8);
            assert_eq!(spec.window_fn(), window_fn);
            approx::assert_relative_eq!(
                spec.normalization().coherent_gain(),
                window.sum() / 8.0,
                max_relative = 1e-15
            );
            approx::assert_relative_eq!(
                spec.normalization().window_energy(),
                window.iter().map(|w| w * w).sum::<f64>(),
                max_relative = 1e-15
            );
        }
    }

    #[test]
    fn new_rejects_all_zero_windows() {
        // 2-sample Hann and Blackman windows are all zeros (Blackman up to rounding).
        for window_fn in [WindowFn::Hann, WindowFn::Blackman] {
            let result = Spectrogram::new(data(2, 3), 8.0, 2, 1, window_fn);
            assert!(matches!(
                result,
                Err(SignalError::DegenerateWindow { window_size: 2, .. })
            ));
        }
        // Hamming and rectangular are fine at 2 samples; 1 sample is all ones.
        assert!(Spectrogram::new(data(2, 3), 8.0, 2, 1, WindowFn::Hamming).is_ok());
        assert!(Spectrogram::new(data(2, 3), 8.0, 2, 1, WindowFn::Rectangular).is_ok());
        assert!(Spectrogram::new(data(1, 3), 8.0, 1, 1, WindowFn::Hann).is_ok());
    }

    #[test]
    fn axes_follow_from_sample_rate_window_and_hop() {
        let (sample_rate, window_size, hop_size, num_frames) = (48_000.0, 1024, 256, 40);
        let spec = Spectrogram::new(
            data(window_size, num_frames),
            sample_rate,
            window_size,
            hop_size,
            WindowFn::Hann,
        )
        .unwrap();

        assert_eq!(spec.frame_axis().len(), num_frames);
        assert_eq!(spec.freq_axis().len(), window_size / 2 + 1);
        // Same values the STFT used to compute and pass in.
        for (f, time) in spec.frame_axis().iter().enumerate() {
            let expected = (f * hop_size + window_size / 2) as f64 / sample_rate;
            approx::assert_relative_eq!(time, expected, max_relative = 1e-14);
        }
        for (k, freq) in spec.freq_axis().iter().enumerate() {
            approx::assert_relative_eq!(
                freq,
                k as f64 * sample_rate / window_size as f64,
                max_relative = 1e-14
            );
        }
        assert_eq!(spec.frame_times()[0], spec.frame_axis().get(0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_stores_parameters_and_validates_on_read() {
        use serde_json::json;

        let mut spec = Spectrogram::new(data(8, 3), 8.0, 8, 4, WindowFn::Hann).unwrap();
        spec.set_channel_label(1, Some("right"));

        let json = serde_json::to_value(&spec).unwrap();
        let mut keys: Vec<_> = json.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "channel_labels",
                "data",
                "hop_size",
                "sample_rate",
                "window_fn",
                "window_size"
            ]
        );

        let back: Spectrogram = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(back, spec);
        assert_eq!(back.channel_label(1), Some("right"));

        let read_edited = |key: &str, value: serde_json::Value| {
            let mut edited = json.clone();
            edited[key] = value;
            serde_json::from_value::<Spectrogram>(edited)
                .unwrap_err()
                .to_string()
        };
        assert!(read_edited("sample_rate", json!(-1.0)).contains("Sample rate"));
        assert!(read_edited("hop_size", json!(0)).contains("Hop size"));
        assert!(read_edited("window_size", json!(32)).contains("but the data has 5"));
        assert!(read_edited("channel_labels", json!([null])).contains("1 channel labels for 2"));
        assert!(read_edited("window_fn", json!("Kaiser")).contains("unknown variant"));
    }
}
