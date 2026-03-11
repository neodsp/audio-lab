use ndarray::prelude::*;
use ndrustfft::{R2cFftHandler, ndfft_r2c};

use super::{SignalError, freq_signal::FreqSignal, utils};
use crate::data::real_data::RealData;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeSignal {
    data: RealData,
    sample_rate: f64,
}

impl TimeSignal {
    pub fn new(data: Array2<f64>, sample_rate: f64) -> Result<Self, SignalError> {
        if sample_rate <= 0.0 {
            return Err(SignalError::SampleRateZeroOrNeg);
        }
        Ok(Self {
            data: RealData::new(
                utils::generate_time_steps(data.shape()[1], sample_rate),
                data,
            )
            .unwrap(),
            sample_rate,
        })
    }

    pub fn zeros(
        num_channels: usize,
        num_time_steps: usize,
        sample_rate: f64,
    ) -> Result<Self, SignalError> {
        if sample_rate <= 0.0 {
            return Err(SignalError::SampleRateZeroOrNeg);
        }
        Ok(Self {
            data: RealData::new(
                utils::generate_time_steps(num_time_steps, sample_rate),
                Array2::zeros((num_channels, num_time_steps)),
            )
            .unwrap(),
            sample_rate,
        })
    }

    pub fn comment(&self) -> Option<&str> {
        self.data.comment()
    }

    pub fn set_comment(&mut self, comment: Option<impl Into<String>>) {
        self.data.set_comment(comment);
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

    pub fn into_freq(self) -> FreqSignal {
        let n_samples = self.num_time_steps();
        let fft_handler = R2cFftHandler::<f64>::new(n_samples);
        let mut freq_signal = FreqSignal::zeros(
            self.num_channels(),
            self.num_freq_bins(),
            self.sample_rate,
            Some(n_samples),
        )
        .unwrap();
        freq_signal.set_comment(self.comment());
        ndfft_r2c(
            &self.time_data(),
            &mut freq_signal.freq_data_mut(),
            &fft_handler,
            1,
        );
        freq_signal
    }

    pub fn data(&self) -> &RealData {
        &self.data
    }

    pub fn time_steps(&self) -> ArrayView1<'_, f64> {
        self.data.x_data()
    }

    pub fn time_data(&self) -> ArrayView2<'_, f64> {
        self.data.y_data()
    }

    pub fn time_data_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        self.data.y_data_mut()
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

// Immutable IntoIterator
impl<'a> IntoIterator for &'a TimeSignal {
    type Item = &'a f64;
    type IntoIter = ndarray::iter::Iter<'a, f64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Mutable IntoIterator
impl<'a> IntoIterator for &'a mut TimeSignal {
    type Item = &'a mut f64;
    type IntoIter = ndarray::iter::IterMut<'a, f64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// Owned IntoIterator
impl IntoIterator for TimeSignal {
    type Item = f64;
    type IntoIter = ndarray::iter::IntoIter<f64, Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
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
        assert_eq!(signal.time_data(), Array2::zeros((0, 0)));

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
    fn odd_length_round_trip() {
        use approx::assert_abs_diff_eq;

        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0]]), 10.0).unwrap();
        assert_eq!(signal.num_time_steps(), 5);
        assert_eq!(signal.num_freq_bins(), 3);

        let freq = signal.into_freq();
        assert_eq!(freq.num_time_steps(), 5);
        assert_eq!(freq.num_freq_bins(), 3);
        assert_eq!(freq.n_samples(), 5);

        let recovered = freq.into_time();
        assert_eq!(recovered.num_time_steps(), 5);
        assert_abs_diff_eq!(
            recovered.channel(0),
            arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            epsilon = 1e-10
        );
    }
}
