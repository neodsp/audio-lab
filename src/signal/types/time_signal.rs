use ndarray::prelude::*;
use ndrustfft::{R2cFftHandler, ndfft_r2c};

use super::{SignalError, freq_signal::FreqSignal, time_axis, utils, validate_sample_rate};
use crate::signal::data::{RealData, XAxis};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "TimeSignalFields")
)]
pub struct TimeSignal {
    data: RealData,
    sample_rate: f64,
}

/// Deserialized fields, checked by `TryFrom` before they become a `TimeSignal`.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct TimeSignalFields {
    data: RealData,
    sample_rate: f64,
}

#[cfg(feature = "serde")]
impl TryFrom<TimeSignalFields> for TimeSignal {
    type Error = SignalError;

    fn try_from(fields: TimeSignalFields) -> Result<Self, Self::Error> {
        validate_sample_rate(fields.sample_rate)?;
        let expected_axis = time_axis(fields.data.num_data_points(), fields.sample_rate);
        if fields.data.x_axis() != &expected_axis {
            return Err(SignalError::AxisMismatch);
        }
        Ok(Self {
            data: fields.data,
            sample_rate: fields.sample_rate,
        })
    }
}

impl TimeSignal {
    pub fn new(data: Array2<f64>, sample_rate: f64) -> Result<Self, SignalError> {
        validate_sample_rate(sample_rate)?;
        Ok(Self {
            data: RealData::from_axis(time_axis(data.ncols(), sample_rate), data)
                .expect("axis length matches the number of samples"),
            sample_rate,
        })
    }

    pub fn zeros(
        num_channels: usize,
        num_time_steps: usize,
        sample_rate: f64,
    ) -> Result<Self, SignalError> {
        validate_sample_rate(sample_rate)?;
        Ok(Self {
            data: RealData::from_axis(
                time_axis(num_time_steps, sample_rate),
                Array2::zeros((num_channels, num_time_steps)),
            )
            .expect("axis length matches the number of samples"),
            sample_rate,
        })
    }

    pub fn from_real_data(data: RealData, sample_rate: f64) -> Result<Self, SignalError> {
        Self::new(data.into_y_data(), sample_rate)
    }

    pub fn comment(&self) -> Option<&str> {
        self.data.comment()
    }

    pub fn set_comment(&mut self, comment: Option<&str>) {
        self.data.set_comment(comment);
    }

    pub fn channel_label(&self, channel: usize) -> Option<&str> {
        self.data.channel_label(channel)
    }

    pub fn set_channel_label(&mut self, channel: usize, label: Option<&str>) {
        self.data.set_channel_label(channel, label);
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn num_channels(&self) -> usize {
        self.data.num_channels()
    }

    pub fn num_time_steps(&self) -> usize {
        self.data.num_data_points()
    }

    pub fn num_freq_bins(&self) -> usize {
        utils::f_from_t(self.num_time_steps())
    }

    pub fn length_in_seconds(&self) -> f64 {
        self.num_time_steps().saturating_sub(1) as f64 / self.sample_rate
    }

    /// Time of each sample in seconds. See [`RealData::x_data`] for when this
    /// allocates. Use [`time_axis`](Self::time_axis) for single values.
    pub fn time_steps(&self) -> CowArray<'_, f64, Ix1> {
        self.data.x_data()
    }

    /// Time axis in seconds, without allocating.
    pub fn time_axis(&self) -> &XAxis {
        self.data.x_axis()
    }

    pub fn time_data(&self) -> ArrayView2<'_, f64> {
        self.data.y_data()
    }

    pub fn time_data_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        self.data.y_data_mut()
    }

    pub fn data(&self) -> &RealData {
        &self.data
    }

    pub fn channel(&self, ch: usize) -> ArrayView1<'_, f64> {
        self.data.channel(ch)
    }

    pub fn channel_mut(&mut self, ch: usize) -> ArrayViewMut1<'_, f64> {
        self.data.channel_mut(ch)
    }

    pub fn channel_iter(&self) -> ndarray::iter::AxisIter<'_, f64, Ix1> {
        self.data.channel_iter()
    }

    pub fn channel_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, f64, Ix1> {
        self.data.channel_iter_mut()
    }

    pub fn iter(&self) -> ndarray::iter::Iter<'_, f64, Ix2> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, f64, Ix2> {
        self.data.iter_mut()
    }

    pub fn into_time(self) -> TimeSignal {
        self
    }

    /// Transform to the frequency domain. Equivalent to [`TimeSignal::to_freq`].
    pub fn into_freq(self) -> FreqSignal {
        self.to_freq()
    }

    /// Transform to the frequency domain without consuming `self`.
    pub fn to_freq(&self) -> FreqSignal {
        let num_time_steps = self.num_time_steps();
        let fft_handler = R2cFftHandler::<f64>::new(num_time_steps);
        let mut freq_signal = FreqSignal::zeros(
            self.num_channels(),
            self.num_freq_bins(),
            self.sample_rate,
            Some(num_time_steps),
        )
        .expect("generate_freq_steps produced invalid data");
        freq_signal.set_comment(self.comment());
        for ch in 0..self.num_channels() {
            freq_signal.set_channel_label(ch, self.channel_label(ch));
        }
        ndfft_r2c(
            &self.time_data(),
            &mut freq_signal.freq_data_mut(),
            &fft_handler,
            1,
        );
        freq_signal
    }
}

impl std::fmt::Display for TimeSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base_info = format!(
            "Time domain signal with {} channels and {} samples at {} Hz sampling rate.",
            self.num_channels(),
            self.num_time_steps(),
            self.sample_rate()
        );

        if let Some(comment) = self.comment() {
            write!(f, "{}\nComment: {:?}", base_info, comment)
        } else {
            write!(f, "{}", base_info)
        }
    }
}

impl approx::AbsDiffEq for TimeSignal {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.sample_rate != other.sample_rate {
            return false;
        }

        if !self.data().abs_diff_eq(other.data(), epsilon) {
            return false;
        }

        true
    }
}

impl<'a> IntoIterator for &'a TimeSignal {
    type Item = &'a f64;
    type IntoIter = ndarray::iter::Iter<'a, f64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut TimeSignal {
    type Item = &'a mut f64;
    type IntoIter = ndarray::iter::IterMut<'a, f64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl IntoIterator for TimeSignal {
    type Item = f64;
    type IntoIter = ndarray::iter::IntoIter<f64, Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use num::complex::Complex64;

    use super::*;

    #[test]
    fn time_signal_error() {
        let result = TimeSignal::new(arr2(&[[0.0], [0.0]]), -10.0);
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));
        let result = TimeSignal::new(arr2(&[[0.0], [0.0]]), 0.0);
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));

        let result = TimeSignal::zeros(2, 10, -10.0);
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));
        let result = TimeSignal::zeros(2, 10, 0.0);
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));
    }

    #[test]
    fn time_signal() -> Result<(), SignalError> {
        let signal = TimeSignal::zeros(0, 0, 1.0).unwrap();
        assert_eq!(signal.time_data(), Array2::<f64>::zeros((0, 0)));

        let signal = TimeSignal::zeros(2, 10, 1.0).unwrap();
        assert_eq!(signal.time_data(), arr2(&[[0.0; 10], [0.0; 10]]));

        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]), 8.0)?;
        let signal = signal.into_time();
        let mut signal = signal.clone();

        assert_eq!(signal.comment(), None);
        signal.set_comment(Some("Hello"));
        assert_eq!(signal.comment(), Some("Hello"));

        assert_eq!(signal.sample_rate(), 8.0);
        assert_eq!(signal.num_channels(), 2);

        assert_eq!(signal.channel(0), arr1(&[0.0, 1.0, 2.0, 3.0]));
        assert_eq!(signal.channel(1), arr1(&[4.0, 5.0, 6.0, 7.0]));
        assert_eq!(signal.channel_mut(0), arr1(&[0.0, 1.0, 2.0, 3.0]));
        assert_eq!(signal.channel_mut(1), arr1(&[4.0, 5.0, 6.0, 7.0]));

        for (i, channel) in signal.channel_iter().enumerate() {
            match i {
                0 => assert_eq!(channel, arr1(&[0.0, 1.0, 2.0, 3.0])),
                1 => assert_eq!(channel, arr1(&[4.0, 5.0, 6.0, 7.0])),
                _ => panic!("Signal should only have two channels"),
            }
        }

        for (i, channel) in signal.channel_iter_mut().enumerate() {
            match i {
                0 => assert_eq!(channel, arr1(&[0.0, 1.0, 2.0, 3.0])),
                1 => assert_eq!(channel, arr1(&[4.0, 5.0, 6.0, 7.0])),
                _ => panic!("Signal should only have two channels"),
            }
        }

        assert_eq!(signal.num_time_steps(), 4);
        assert_eq!(signal.num_freq_bins(), 3);
        assert_eq!(signal.length_in_seconds(), 0.375);
        assert_eq!(signal.time_steps(), arr1(&[0.0, 0.125, 0.25, 0.375]));
        assert_eq!(
            signal.time_data(),
            arr2(&[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        );
        assert_eq!(
            signal.time_data_mut(),
            arr2(&[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        );

        let freq_signal = signal.into_freq();
        assert_eq!(freq_signal.channel(0)[0], Complex64::new(6.0, 0.0));
        assert_eq!(freq_signal.channel(0)[1], Complex64::new(-2.0, 2.0));
        assert_eq!(freq_signal.channel(0)[2], Complex64::new(-2.0, 0.0));

        assert_eq!(freq_signal.channel(1)[0], Complex64::new(22.0, 0.0));
        assert_eq!(freq_signal.channel(1)[1], Complex64::new(-2.0, 2.0));
        assert_eq!(freq_signal.channel(1)[2], Complex64::new(-2.0, 0.0));

        assert_eq!(freq_signal.sample_rate(), 8.0);
        assert_eq!(freq_signal.num_channels(), 2);

        assert_eq!(freq_signal.num_time_steps(), 4);
        assert_eq!(freq_signal.num_freq_bins(), 3);
        assert_eq!(freq_signal.length_in_seconds(), 0.375);
        assert_eq!(freq_signal.freq_bins(), arr1(&[0.0, 2.0, 4.0]));

        let time_signal = freq_signal.into_time();
        assert_eq!(time_signal.channel(0), arr1(&[0.0, 1.0, 2.0, 3.0]));
        assert_eq!(time_signal.channel(1), arr1(&[4.0, 5.0, 6.0, 7.0]));
        Ok(())
    }

    #[test]
    fn signals_use_uniform_axes() {
        let signal = TimeSignal::zeros(2, 6, 4.0).unwrap();
        assert_eq!(signal.time_axis().as_uniform(), Some((0.0, 0.25)));
        assert_eq!(
            signal.time_steps(),
            arr1(&[0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
        );

        let freq = signal.to_freq();
        assert_eq!(freq.freq_axis().as_uniform(), Some((0.0, 4.0 / 6.0)));
        assert_eq!(freq.freq_axis().len(), 4);

        // Matches the previously stored linspace values. Time values are
        // identical; frequency values can differ in the last bit because the
        // spacing is now sample_rate / N instead of end / (bins - 1).
        for sample_rate in [44_100.0, 48_000.0] {
            let signal = TimeSignal::zeros(1, 1001, sample_rate).unwrap();
            assert_eq!(
                signal.time_steps(),
                utils::generate_time_steps(1001, sample_rate)
            );
            let expected = utils::generate_freq_steps(501, sample_rate, 1001);
            for (actual, expected) in signal.to_freq().freq_axis().iter().zip(expected) {
                approx::assert_relative_eq!(actual, expected, max_relative = 1e-15);
            }
        }
    }

    #[test]
    fn rejects_non_finite_sample_rates() {
        for sample_rate in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                TimeSignal::zeros(1, 4, sample_rate),
                Err(SignalError::SampleRateZeroOrNeg)
            ));
            assert!(matches!(
                FreqSignal::zeros(1, 3, sample_rate, None),
                Err(SignalError::SampleRateZeroOrNeg)
            ));
        }
    }

    #[test]
    fn constructs_from_owned_data_containers() {
        use crate::signal::data::{ComplexData, RealData};

        let y = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let time = TimeSignal::from_real_data(RealData::from_y_data(y.clone()), 10.0).unwrap();
        assert_eq!(time.time_data(), y);
        assert_eq!(time.time_steps(), arr1(&[0.0, 0.1, 0.2]));

        let spectrum = arr2(&[[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]]);
        let freq = FreqSignal::from_complex_data(
            ComplexData::from_y_data(spectrum.clone()),
            10.0,
            Some(2),
        )
        .unwrap();
        assert_eq!(freq.freq_data(), spectrum);
        assert_eq!(freq.freq_bins(), arr1(&[0.0, 5.0]));
    }

    #[test]
    fn borrowing_conversions_match_consuming_conversions() {
        let mut signal = TimeSignal::new(arr2(&[[1.0, -2.0, 3.0, 0.5, 4.0]]), 10.0).unwrap();
        signal.set_comment(Some("comment"));
        signal.set_channel_label(0, Some("left"));

        let freq = signal.to_freq();
        assert_eq!(freq, signal.clone().into_freq());
        assert_eq!(freq.channel_label(0), Some("left"));

        let time = freq.to_time();
        assert_eq!(time, freq.clone().into_time());
        assert_eq!(time.comment(), Some("comment"));
    }

    #[test]
    fn odd_length_round_trip() {
        use approx::assert_abs_diff_eq;

        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0]]), 10.0).unwrap();
        assert_eq!(signal.num_time_steps(), 5);
        assert_eq!(signal.num_freq_bins(), 3);

        let freq = signal.into_freq();
        assert_eq!(freq.num_time_steps(), 5);
        assert_eq!(freq.num_freq_bins(), 3);
        assert_eq!(freq.num_time_steps(), 5);

        let recovered = freq.into_time();
        assert_eq!(recovered.num_time_steps(), 5);
        assert_abs_diff_eq!(
            recovered.channel(0),
            arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            epsilon = 1e-10
        );
    }
}
