use ndarray::prelude::*;
use num::complex::Complex64;

use super::{DataError, XAxis};

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "ComplexDataFields")
)]
pub struct ComplexData {
    x_axis: XAxis,
    y_data: Array2<Complex64>,
    comment: Option<String>,
    channel_labels: Vec<Option<String>>,
}

/// Deserialized fields, checked by `TryFrom` before they become a `ComplexData`.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct ComplexDataFields {
    x_axis: XAxis,
    y_data: Array2<Complex64>,
    comment: Option<String>,
    channel_labels: Vec<Option<String>>,
}

#[cfg(feature = "serde")]
impl TryFrom<ComplexDataFields> for ComplexData {
    type Error = DataError;

    fn try_from(fields: ComplexDataFields) -> Result<Self, Self::Error> {
        if fields.channel_labels.len() != fields.y_data.nrows() {
            return Err(DataError::ChannelLabelMismatch {
                labels: fields.channel_labels.len(),
                channels: fields.y_data.nrows(),
            });
        }
        let mut data = Self::from_axis(fields.x_axis, fields.y_data)?;
        data.comment = fields.comment;
        data.channel_labels = fields.channel_labels;
        Ok(data)
    }
}

impl ComplexData {
    pub fn new(x_data: Array1<f64>, y_data: Array2<Complex64>) -> Result<Self, DataError> {
        if y_data.ncols() != x_data.len() {
            return Err(DataError::NotMatching);
        }
        Self::from_axis(XAxis::explicit(x_data)?, y_data)
    }

    /// Create data from an existing axis. Only the length is checked, since
    /// an [`XAxis`] is always valid.
    pub fn from_axis(x_axis: XAxis, y_data: Array2<Complex64>) -> Result<Self, DataError> {
        if y_data.ncols() != x_axis.len() {
            return Err(DataError::NotMatching);
        }
        let num_channels = y_data.nrows();
        Ok(Self {
            x_axis,
            y_data,
            comment: None,
            channel_labels: vec![None; num_channels],
        })
    }

    pub fn zeros(num_channels: usize, num_data_points: usize) -> Self {
        Self {
            x_axis: XAxis::indices(num_data_points),
            y_data: Array2::zeros((num_channels, num_data_points)),
            comment: None,
            channel_labels: vec![None; num_channels],
        }
    }

    pub fn from_x_data(x_data: Array1<f64>, num_channels: usize) -> Result<Self, DataError> {
        let x_axis = XAxis::explicit(x_data)?;
        Ok(Self {
            y_data: Array2::zeros((num_channels, x_axis.len())),
            x_axis,
            comment: None,
            channel_labels: vec![None; num_channels],
        })
    }

    pub fn from_y_data(y_data: Array2<Complex64>) -> Self {
        let num_channels = y_data.nrows();
        Self {
            x_axis: XAxis::indices(y_data.ncols()),
            y_data,
            comment: None,
            channel_labels: vec![None; num_channels],
        }
    }

    pub fn comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    pub fn set_comment(&mut self, comment: Option<&str>) {
        self.comment = comment.map(Into::into);
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
        self.y_data.nrows()
    }

    pub fn num_data_points(&self) -> usize {
        self.y_data.ncols()
    }

    /// The x axis. Prefer this over [`x_data`](Self::x_data) for single
    /// values ([`XAxis::get`]) or iteration, because it never allocates.
    pub fn x_axis(&self) -> &XAxis {
        &self.x_axis
    }

    /// The x values as an array. Borrowed for an explicit axis, computed
    /// (one allocation) for a uniform one.
    pub fn x_data(&self) -> CowArray<'_, f64, Ix1> {
        self.x_axis.to_cow_array()
    }

    /// Replace the x axis. It must have one value per data point.
    pub fn set_x_axis(&mut self, x_axis: XAxis) -> Result<(), DataError> {
        if x_axis.len() != self.num_data_points() {
            return Err(DataError::NotMatching);
        }
        self.x_axis = x_axis;
        Ok(())
    }

    pub fn y_data(&self) -> ArrayView2<'_, Complex64> {
        self.y_data.view()
    }

    pub fn y_data_mut(&mut self) -> ArrayViewMut2<'_, Complex64> {
        self.y_data.view_mut()
    }

    /// Consume the data and return the owned y values without copying.
    pub fn into_y_data(self) -> Array2<Complex64> {
        self.y_data
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

    /// Index of the x value closest to `x`. On a tie the lower index wins.
    ///
    /// O(1) for a uniform axis, a binary search otherwise.
    pub fn nearest_x_index(&self, x: f64) -> usize {
        self.x_axis.nearest_index(x)
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
        self.x_axis == other.x_axis && self.y_data == other.y_data
    }
}

impl approx::AbsDiffEq for ComplexData {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        if !self.x_axis.abs_diff_eq(&other.x_axis, epsilon) {
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
    fn nearest_x_index_matches_linear_scan() {
        let data = ComplexData::from_x_data(array![0.0, 1.0, 2.5, 4.0, 10.0], 1).unwrap();
        let linear_scan = |x: f64| {
            data.x_data()
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (*a - x)
                        .abs()
                        .partial_cmp(&(*b - x).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index)
                .unwrap()
        };

        // Exact hits, midpoints (ties go to the lower index), values outside
        // the range, and NaN.
        let queries = [
            -5.0,
            0.0,
            0.5,
            0.6,
            1.75,
            2.5,
            3.25,
            3.3,
            7.0,
            7.1,
            10.0,
            42.0,
            f64::NAN,
        ];
        for x in queries {
            assert_eq!(data.nearest_x_index(x), linear_scan(x), "x = {x}");
        }

        assert_eq!(ComplexData::zeros(1, 0).nearest_x_index(1.0), 0);
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
