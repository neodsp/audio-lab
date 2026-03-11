use ndarray::prelude::*;
use num::complex::Complex64;

use super::DataError;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComplexData {
    x_data: Array1<f64>,
    y_data: Array2<Complex64>,
    comment: Option<String>,
}

impl ComplexData {
    pub fn new(x_data: Array1<f64>, y_data: Array2<Complex64>) -> Result<Self, DataError> {
        let num_samples = y_data.ncols();
        if num_samples != x_data.len() {
            return Err(DataError::NotMatching);
        }
        if !x_data
            .windows(2)
            .into_iter()
            .all(|window| window[1] > window[0])
        {
            return Err(DataError::NotIncreasing);
        }
        Ok(Self {
            x_data,
            y_data,
            comment: None,
        })
    }

    pub fn zeros(num_channels: usize, num_data_points: usize) -> Self {
        Self {
            x_data: Array1::from_iter((0..num_data_points).map(|i| i as f64)),
            y_data: Array2::zeros((num_channels, num_data_points)),
            comment: None,
        }
    }

    pub fn from_x_data(x_data: Array1<f64>, num_channels: usize) -> Result<Self, DataError> {
        if !x_data
            .windows(2)
            .into_iter()
            .all(|window| window[1] > window[0])
        {
            return Err(DataError::NotIncreasing);
        }
        Ok(Self {
            y_data: Array2::zeros((num_channels, x_data.len())),
            x_data,
            comment: None,
        })
    }

    pub fn from_y_data(y_data: Array2<Complex64>) -> Self {
        Self {
            x_data: Array1::from_iter((0..y_data.ncols()).map(|i| i as f64)),
            y_data,
            comment: None,
        }
    }

    pub fn comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    pub fn set_comment(&mut self, comment: Option<&str>) {
        self.comment = comment.map(Into::into);
    }

    pub fn num_channels(&self) -> usize {
        self.y_data.nrows()
    }

    pub fn num_data_points(&self) -> usize {
        self.y_data.ncols()
    }

    pub fn x_data(&self) -> ArrayView1<'_, f64> {
        self.x_data.view()
    }

    pub fn x_data_mut(&mut self) -> ArrayViewMut1<'_, f64> {
        self.x_data.view_mut()
    }

    pub fn y_data(&self) -> ArrayView2<'_, Complex64> {
        self.y_data.view()
    }

    pub fn y_data_mut(&mut self) -> ArrayViewMut2<'_, Complex64> {
        self.y_data.view_mut()
    }

    pub fn channel(&self, ch: usize) -> ArrayView1<'_, Complex64> {
        self.y_data.row(ch)
    }

    pub fn channel_mut(&mut self, ch: usize) -> ArrayViewMut1<'_, Complex64> {
        self.y_data.row_mut(ch)
    }

    pub fn channel_iter(&self) -> ndarray::iter::AxisIter<'_, Complex64, Ix1> {
        self.y_data.outer_iter()
    }

    pub fn channel_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, Complex64, Ix1> {
        self.y_data.outer_iter_mut()
    }

    pub fn iter(&self) -> ndarray::iter::Iter<'_, Complex64, Ix2> {
        self.y_data.iter()
    }

    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, Complex64, Ix2> {
        self.y_data.iter_mut()
    }
}

impl std::fmt::Display for ComplexData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base_info = format!(
            "Complex data with {} channels and {} data points.",
            self.num_channels(),
            self.num_data_points()
        );

        if let Some(comment) = self.comment() {
            write!(f, "{}\nComment: {:?}", base_info, comment)
        } else {
            write!(f, "{}", base_info)
        }
    }
}

impl PartialEq for ComplexData {
    fn eq(&self, other: &Self) -> bool {
        self.x_data == other.x_data && self.y_data == other.y_data
    }
}

impl approx::AbsDiffEq for ComplexData {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if !self.x_data().abs_diff_eq(&other.x_data(), epsilon) {
            return false;
        }

        // Compare y_data
        if !self.y_data().abs_diff_eq(&other.y_data(), epsilon) {
            return false;
        }

        true
    }
}

// Immutable IntoIterator
impl<'a> IntoIterator for &'a ComplexData {
    type Item = &'a Complex64;
    type IntoIter = ndarray::iter::Iter<'a, Complex64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Mutable IntoIterator
impl<'a> IntoIterator for &'a mut ComplexData {
    type Item = &'a mut Complex64;
    type IntoIter = ndarray::iter::IterMut<'a, Complex64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// Owned IntoIterator
impl IntoIterator for ComplexData {
    type Item = Complex64;
    type IntoIter = ndarray::iter::IntoIter<Complex64, Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.y_data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_data_errors() {
        let error = ComplexData::new(
            array![0.0, 1.0, 2.0, 5.0],
            arr2(&[[Complex64::from(0.0); 3]; 2]),
        )
        .err();
        assert!(matches!(error, Some(DataError::NotMatching)));

        let error =
            ComplexData::new(array![0.0, 0.0, 0.0], arr2(&[[Complex64::from(0.0); 3]; 2])).err();
        assert!(matches!(error, Some(DataError::NotIncreasing)));

        let error = ComplexData::from_x_data(array![0.0, 0.0, 0.0], 3).err();
        assert!(matches!(error, Some(DataError::NotIncreasing)));
    }

    #[test]
    fn test_complex_data() -> Result<(), DataError> {
        let data = ComplexData::zeros(2, 4);
        assert_eq!(data.x_data(), arr1(&[0.0, 1.0, 2.0, 3.0]));
        assert_eq!(data.y_data(), arr2(&[[Complex64::from(0.0); 4]; 2]),);

        let data = ComplexData::from_x_data(arr1(&[0.0, 1.0, 2.0, 5.0]), 2).unwrap();
        assert_eq!(data.x_data(), arr1(&[0.0, 1.0, 2.0, 5.0]));
        assert_eq!(data.y_data(), arr2(&[[Complex64::from(0.0); 4]; 2]),);

        let mut i = 0;
        let mut data = ComplexData::new(
            Array1::from_iter((0..10).map(|v| v as f64 / 10.0)),
            Array2::from_shape_fn((2, 10), |_| {
                let value = Complex64::from(i as f64);
                i += 1;
                value
            }),
        )?;

        assert_eq!(data.comment(), None);

        data.set_comment(Some("This is just test data"));

        assert_eq!(data.comment(), Some("This is just test data"));

        assert_eq!(data.num_channels(), 2);
        assert_eq!(data.num_data_points(), 10);

        assert_eq!(
            data.x_data(),
            arr1(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        );

        assert_eq!(
            data.x_data_mut(),
            arr1(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        );

        assert_eq!(
            data.y_data(),
            arr2(&[
                [
                    Complex64::from(0.0),
                    Complex64::from(1.0),
                    Complex64::from(2.0),
                    Complex64::from(3.0),
                    Complex64::from(4.0),
                    Complex64::from(5.0),
                    Complex64::from(6.0),
                    Complex64::from(7.0),
                    Complex64::from(8.0),
                    Complex64::from(9.0)
                ],
                [
                    Complex64::from(10.0),
                    Complex64::from(11.0),
                    Complex64::from(12.0),
                    Complex64::from(13.0),
                    Complex64::from(14.0),
                    Complex64::from(15.0),
                    Complex64::from(16.0),
                    Complex64::from(17.0),
                    Complex64::from(18.0),
                    Complex64::from(19.0),
                ]
            ])
        );

        assert_eq!(
            data.y_data_mut(),
            arr2(&[
                [
                    Complex64::from(0.0),
                    Complex64::from(1.0),
                    Complex64::from(2.0),
                    Complex64::from(3.0),
                    Complex64::from(4.0),
                    Complex64::from(5.0),
                    Complex64::from(6.0),
                    Complex64::from(7.0),
                    Complex64::from(8.0),
                    Complex64::from(9.0)
                ],
                [
                    Complex64::from(10.0),
                    Complex64::from(11.0),
                    Complex64::from(12.0),
                    Complex64::from(13.0),
                    Complex64::from(14.0),
                    Complex64::from(15.0),
                    Complex64::from(16.0),
                    Complex64::from(17.0),
                    Complex64::from(18.0),
                    Complex64::from(19.0),
                ]
            ])
        );

        let a: Vec<Complex64> = data.clone().into_iter().collect();
        assert_eq!(
            a,
            (0..20)
                .map(|v| Complex64::from(v as f64))
                .collect::<Vec<_>>()
        );

        assert_eq!(
            data.channel(0),
            arr1(&[
                Complex64::from(0.0),
                Complex64::from(1.0),
                Complex64::from(2.0),
                Complex64::from(3.0),
                Complex64::from(4.0),
                Complex64::from(5.0),
                Complex64::from(6.0),
                Complex64::from(7.0),
                Complex64::from(8.0),
                Complex64::from(9.0)
            ])
        );

        assert_eq!(
            data.channel_mut(0),
            arr1(&[
                Complex64::from(0.0),
                Complex64::from(1.0),
                Complex64::from(2.0),
                Complex64::from(3.0),
                Complex64::from(4.0),
                Complex64::from(5.0),
                Complex64::from(6.0),
                Complex64::from(7.0),
                Complex64::from(8.0),
                Complex64::from(9.0)
            ])
        );

        assert_eq!(
            data.channel(1),
            arr1(&[
                Complex64::from(10.0),
                Complex64::from(11.0),
                Complex64::from(12.0),
                Complex64::from(13.0),
                Complex64::from(14.0),
                Complex64::from(15.0),
                Complex64::from(16.0),
                Complex64::from(17.0),
                Complex64::from(18.0),
                Complex64::from(19.0)
            ])
        );

        assert_eq!(
            data.channel_mut(1),
            arr1(&[
                Complex64::from(10.0),
                Complex64::from(11.0),
                Complex64::from(12.0),
                Complex64::from(13.0),
                Complex64::from(14.0),
                Complex64::from(15.0),
                Complex64::from(16.0),
                Complex64::from(17.0),
                Complex64::from(18.0),
                Complex64::from(19.0)
            ])
        );

        for (i, ch) in data.channel_iter().enumerate() {
            match i {
                0 => assert_eq!(
                    ch,
                    arr1(&[
                        Complex64::from(0.0),
                        Complex64::from(1.0),
                        Complex64::from(2.0),
                        Complex64::from(3.0),
                        Complex64::from(4.0),
                        Complex64::from(5.0),
                        Complex64::from(6.0),
                        Complex64::from(7.0),
                        Complex64::from(8.0),
                        Complex64::from(9.0)
                    ])
                ),
                1 => assert_eq!(
                    ch,
                    arr1(&[
                        Complex64::from(10.0),
                        Complex64::from(11.0),
                        Complex64::from(12.0),
                        Complex64::from(13.0),
                        Complex64::from(14.0),
                        Complex64::from(15.0),
                        Complex64::from(16.0),
                        Complex64::from(17.0),
                        Complex64::from(18.0),
                        Complex64::from(19.0)
                    ])
                ),
                _ => panic!("Should only have two channels"),
            }
        }

        for (i, ch) in data.channel_iter_mut().enumerate() {
            match i {
                0 => assert_eq!(
                    ch,
                    arr1(&[
                        Complex64::from(0.0),
                        Complex64::from(1.0),
                        Complex64::from(2.0),
                        Complex64::from(3.0),
                        Complex64::from(4.0),
                        Complex64::from(5.0),
                        Complex64::from(6.0),
                        Complex64::from(7.0),
                        Complex64::from(8.0),
                        Complex64::from(9.0)
                    ])
                ),
                1 => assert_eq!(
                    ch,
                    arr1(&[
                        Complex64::from(10.0),
                        Complex64::from(11.0),
                        Complex64::from(12.0),
                        Complex64::from(13.0),
                        Complex64::from(14.0),
                        Complex64::from(15.0),
                        Complex64::from(16.0),
                        Complex64::from(17.0),
                        Complex64::from(18.0),
                        Complex64::from(19.0)
                    ])
                ),
                _ => panic!("Should only have two channels"),
            }
        }

        Ok(())
    }
}
