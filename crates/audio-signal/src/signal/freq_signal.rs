use ndarray::prelude::*;
use ndrustfft::{R2cFftHandler, ndifft_r2c};
use num::complex::Complex64;

use super::{SignalError, TimeSignal, utils};
use crate::data::complex_data::ComplexData;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FreqSignal {
    data: ComplexData,
    sample_rate: f64,
    n_samples: usize,
}

impl FreqSignal {
    pub fn new(
        data: Array2<Complex64>,
        sample_rate: f64,
        n_samples: Option<usize>,
    ) -> Result<Self, SignalError> {
        if sample_rate <= 0.0 {
            return Err(SignalError::SampleRateZeroOrNeg);
        }
        let n_samples = n_samples.unwrap_or_else(|| utils::t_from_f(data.ncols()));
        Ok(Self {
            data: ComplexData::new(
                utils::generate_freq_steps(data.ncols(), sample_rate, n_samples),
                data,
            )
            .expect("generate_freq_steps produced invalid data"),
            sample_rate,
            n_samples,
        })
    }

    pub fn zeros(
        num_channels: usize,
        num_freq_bins: usize,
        sample_rate: f64,
        n_samples: Option<usize>,
    ) -> Result<Self, SignalError> {
        if sample_rate <= 0.0 {
            return Err(SignalError::SampleRateZeroOrNeg);
        }
        let n_samples = n_samples.unwrap_or_else(|| utils::t_from_f(num_freq_bins));
        Ok(Self {
            data: ComplexData::new(
                utils::generate_freq_steps(num_freq_bins, sample_rate, n_samples),
                Array2::zeros((num_channels, num_freq_bins)),
            )
            .expect("generate_freq_steps produced invalid data"),
            sample_rate,
            n_samples,
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

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn num_time_steps(&self) -> usize {
        self.n_samples
    }

    pub fn num_freq_bins(&self) -> usize {
        self.data.num_data_points()
    }

    pub fn length_in_seconds(&self) -> f64 {
        self.num_time_steps().saturating_sub(1) as f64 / self.sample_rate
    }

    pub fn channel(&self, ch: usize) -> ArrayView1<'_, Complex64> {
        self.data.channel(ch)
    }

    pub fn channel_mut(&mut self, ch: usize) -> ArrayViewMut1<'_, Complex64> {
        self.data.channel_mut(ch)
    }

    pub fn channel_iter(&self) -> ndarray::iter::AxisIter<'_, Complex64, Ix1> {
        self.data.channel_iter()
    }

    pub fn channel_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, Complex64, Ix1> {
        self.data.channel_iter_mut()
    }

    pub fn iter(&self) -> ndarray::iter::Iter<'_, Complex64, Ix2> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, Complex64, Ix2> {
        self.data.iter_mut()
    }

    pub fn into_time(self) -> TimeSignal {
        let fft_handler = R2cFftHandler::<f64>::new(self.num_time_steps());
        let mut time_signal =
            TimeSignal::zeros(self.num_channels(), self.num_time_steps(), self.sample_rate)
                .expect("generate_time_steps produced invalid data");
        time_signal.set_comment(self.comment());
        ndifft_r2c(
            &self.freq_data(),
            &mut time_signal.time_data_mut(),
            &fft_handler,
            1,
        );
        time_signal
    }

    pub fn into_freq(self) -> FreqSignal {
        self
    }

    pub fn data(&self) -> &ComplexData {
        &self.data
    }

    pub fn freq_bins(&self) -> ArrayView1<'_, f64> {
        self.data.x_data()
    }

    pub fn freq_data(&self) -> ArrayView2<'_, Complex64> {
        self.data.y_data()
    }

    pub fn freq_data_mut(&mut self) -> ArrayViewMut2<'_, Complex64> {
        self.data.y_data_mut()
    }
}

impl std::fmt::Display for FreqSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base_info = format!(
            "Frequency domain signal with {} channels and {} bins at {} Hz sampling rate.",
            self.num_channels(),
            self.num_freq_bins(),
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
impl<'a> IntoIterator for &'a FreqSignal {
    type Item = &'a Complex64;
    type IntoIter = ndarray::iter::Iter<'a, Complex64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Mutable IntoIterator
impl<'a> IntoIterator for &'a mut FreqSignal {
    type Item = &'a mut Complex64;
    type IntoIter = ndarray::iter::IterMut<'a, Complex64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// Owned IntoIterator
impl IntoIterator for FreqSignal {
    type Item = Complex64;
    type IntoIter = ndarray::iter::IntoIter<Complex64, Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl approx::AbsDiffEq for FreqSignal {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if self.sample_rate != other.sample_rate || self.n_samples != other.n_samples {
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
    use num::Zero;

    use super::*;

    #[test]
    fn freq_signal_error() {
        let result = FreqSignal::new(
            arr2(&[[Complex64::from(0.0)], [Complex64::from(0.0)]]),
            -10.0,
            None,
        );
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));
        let result = FreqSignal::new(
            arr2(&[[Complex64::from(0.0)], [Complex64::from(0.0)]]),
            0.0,
            None,
        );
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));

        let result = FreqSignal::zeros(2, 10, -10.0, None);
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));
        let result = FreqSignal::zeros(2, 10, 0.0, None);
        assert!(matches!(result, Err(SignalError::SampleRateZeroOrNeg)));
    }

    #[test]
    fn freq_signal() -> Result<(), SignalError> {
        let signal = FreqSignal::zeros(0, 0, 1.0, None).unwrap();
        assert_eq!(signal.freq_data(), Array2::zeros((0, 0)));

        let signal = FreqSignal::zeros(2, 10, 1.0, None).unwrap();
        assert_eq!(
            signal.freq_data(),
            arr2(&[[Complex64::zero(); 10], [Complex64::zero(); 10]])
        );

        let signal = FreqSignal::new(
            arr2(&[
                [
                    Complex64::new(6.0, 0.0),
                    Complex64::new(-2.0, 2.0),
                    Complex64::new(-2.0, 0.0),
                ],
                [
                    Complex64::new(22.0, 0.0),
                    Complex64::new(-2.0, 2.0),
                    Complex64::new(-2.0, 0.0),
                ],
            ]),
            8.0,
            Some(4),
        )?;
        let signal = signal.into_freq();
        let mut signal = signal.clone();

        assert_eq!(signal.comment(), None);
        signal.set_comment(Some("Hello"));
        assert_eq!(signal.comment(), Some("Hello"));

        assert_eq!(signal.sample_rate(), 8.0);
        assert_eq!(signal.num_channels(), 2);

        assert_eq!(
            signal.channel(0),
            arr1(&[
                Complex64::new(6.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ],)
        );
        assert_eq!(
            signal.channel(1),
            arr1(&[
                Complex64::new(22.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ],)
        );

        assert_eq!(
            signal.channel_mut(0),
            arr1(&[
                Complex64::new(6.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ],)
        );
        assert_eq!(
            signal.channel_mut(1),
            arr1(&[
                Complex64::new(22.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ],)
        );

        for (i, channel) in signal.channel_iter().enumerate() {
            match i {
                0 => assert_eq!(
                    channel,
                    arr1(&[
                        Complex64::new(6.0, 0.0),
                        Complex64::new(-2.0, 2.0),
                        Complex64::new(-2.0, 0.0),
                    ],)
                ),
                1 => assert_eq!(
                    channel,
                    arr1(&[
                        Complex64::new(22.0, 0.0),
                        Complex64::new(-2.0, 2.0),
                        Complex64::new(-2.0, 0.0),
                    ],)
                ),
                _ => panic!("Signal should only have two channels"),
            }
        }

        for (i, channel) in signal.channel_iter_mut().enumerate() {
            match i {
                0 => assert_eq!(
                    channel,
                    arr1(&[
                        Complex64::new(6.0, 0.0),
                        Complex64::new(-2.0, 2.0),
                        Complex64::new(-2.0, 0.0),
                    ],)
                ),
                1 => assert_eq!(
                    channel,
                    arr1(&[
                        Complex64::new(22.0, 0.0),
                        Complex64::new(-2.0, 2.0),
                        Complex64::new(-2.0, 0.0),
                    ],)
                ),
                _ => panic!("Signal should only have two channels"),
            }
        }

        assert_eq!(signal.num_time_steps(), 4);
        assert_eq!(signal.num_freq_bins(), 3);
        assert_eq!(signal.length_in_seconds(), 0.375);
        assert_eq!(signal.freq_bins(), arr1(&[0.0, 2.0, 4.0]));
        assert_eq!(
            signal.freq_data(),
            arr2(&[
                [
                    Complex64::new(6.0, 0.0),
                    Complex64::new(-2.0, 2.0),
                    Complex64::new(-2.0, 0.0),
                ],
                [
                    Complex64::new(22.0, 0.0),
                    Complex64::new(-2.0, 2.0),
                    Complex64::new(-2.0, 0.0),
                ],
            ])
        );

        assert_eq!(
            signal.freq_data_mut(),
            arr2(&[
                [
                    Complex64::new(6.0, 0.0),
                    Complex64::new(-2.0, 2.0),
                    Complex64::new(-2.0, 0.0),
                ],
                [
                    Complex64::new(22.0, 0.0),
                    Complex64::new(-2.0, 2.0),
                    Complex64::new(-2.0, 0.0),
                ],
            ])
        );

        let freq_signal = signal.into_time();
        assert_eq!(freq_signal.channel(0)[0], 0.0);
        assert_eq!(freq_signal.channel(0)[1], 1.0);
        assert_eq!(freq_signal.channel(0)[2], 2.0);
        assert_eq!(freq_signal.channel(0)[3], 3.0);

        assert_eq!(freq_signal.channel(1)[0], 4.0);
        assert_eq!(freq_signal.channel(1)[1], 5.0);
        assert_eq!(freq_signal.channel(1)[2], 6.0);

        assert_eq!(freq_signal.sample_rate(), 8.0);
        assert_eq!(freq_signal.num_channels(), 2);

        assert_eq!(freq_signal.num_time_steps(), 4);
        assert_eq!(freq_signal.num_freq_bins(), 3);
        assert_eq!(freq_signal.length_in_seconds(), 0.375);
        assert_eq!(freq_signal.time_steps(), arr1(&[0.0, 0.125, 0.25, 0.375]));

        let time_signal = freq_signal.into_freq();
        assert_eq!(
            time_signal.channel(0),
            arr1(&[
                Complex64::new(6.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ])
        );
        assert_eq!(
            time_signal.channel(1),
            arr1(&[
                Complex64::new(22.0, 0.0),
                Complex64::new(-2.0, 2.0),
                Complex64::new(-2.0, 0.0),
            ])
        );
        Ok(())
    }
}
