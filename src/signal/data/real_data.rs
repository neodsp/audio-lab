use ndarray::prelude::*;

use super::{DataError, XAxis};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RealData {
    x_axis: XAxis,
    y_data: Array2<f64>,
    comment: Option<String>,
    channel_labels: Vec<Option<String>>,
}

impl RealData {
    pub fn new(x_data: Array1<f64>, y_data: Array2<f64>) -> Result<Self, DataError> {
        if y_data.ncols() != x_data.len() {
            return Err(DataError::NotMatching);
        }
        Self::from_axis(XAxis::explicit(x_data)?, y_data)
    }

    /// Create data from an existing axis. Only the length is checked, since
    /// an [`XAxis`] is always valid.
    pub fn from_axis(x_axis: XAxis, y_data: Array2<f64>) -> Result<Self, DataError> {
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

    pub fn from_y_data(y_data: Array2<f64>) -> Self {
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

    /// Mutable access to the x values. Turns a uniform axis into an explicit
    /// one. The values must stay strictly increasing.
    pub fn x_data_mut(&mut self) -> ArrayViewMut1<'_, f64> {
        if self.x_axis.as_uniform().is_some() {
            self.x_axis = XAxis::explicit(self.x_axis.to_array())
                .expect("values of a valid axis are increasing");
        }
        self.x_axis.explicit_values_mut()
    }

    pub fn y_data(&self) -> ArrayView2<'_, f64> {
        self.y_data.view()
    }

    pub fn y_data_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        self.y_data.view_mut()
    }

    /// Consume the data and return the owned y values without copying.
    pub fn into_y_data(self) -> Array2<f64> {
        self.y_data
    }

    pub fn channel(&self, ch: usize) -> ArrayView1<'_, f64> {
        self.y_data.row(ch)
    }

    pub fn channel_mut(&mut self, ch: usize) -> ArrayViewMut1<'_, f64> {
        self.y_data.row_mut(ch)
    }

    pub fn channel_iter(&self) -> ndarray::iter::AxisIter<'_, f64, Ix1> {
        self.y_data.outer_iter()
    }

    pub fn channel_iter_mut(&mut self) -> ndarray::iter::AxisIterMut<'_, f64, Ix1> {
        self.y_data.outer_iter_mut()
    }

    pub fn iter(&self) -> ndarray::iter::Iter<'_, f64, Ix2> {
        self.y_data.iter()
    }

    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, f64, Ix2> {
        self.y_data.iter_mut()
    }
}

impl std::fmt::Display for RealData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base_info = format!(
            "Real data with {} channels and {} data points.",
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

impl PartialEq for RealData {
    fn eq(&self, other: &Self) -> bool {
        self.x_axis == other.x_axis && self.y_data == other.y_data
    }
}

impl approx::AbsDiffEq for RealData {
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
impl<'a> IntoIterator for &'a RealData {
    type Item = &'a f64;
    type IntoIter = ndarray::iter::Iter<'a, f64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Mutable IntoIterator
impl<'a> IntoIterator for &'a mut RealData {
    type Item = &'a mut f64;
    type IntoIter = ndarray::iter::IterMut<'a, f64, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// Owned IntoIterator
impl IntoIterator for RealData {
    type Item = f64;
    type IntoIter = ndarray::iter::IntoIter<f64, Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.y_data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_axis_checks_length_and_keeps_axis() {
        let axis = XAxis::uniform(0.0, 0.5, 3).unwrap();
        let data = RealData::from_axis(axis.clone(), array![[1.0, 2.0, 3.0]]).unwrap();
        assert_eq!(data.x_axis(), &axis);
        assert_eq!(data.x_data(), arr1(&[0.0, 0.5, 1.0]));

        let error = RealData::from_axis(axis, array![[1.0, 2.0]]).err();
        assert!(matches!(error, Some(DataError::NotMatching)));
    }

    #[test]
    fn x_data_mut_turns_uniform_axis_into_explicit_values() {
        let mut data = RealData::from_y_data(array![[1.0, 2.0, 3.0]]);
        assert_eq!(data.x_axis().as_uniform(), Some((0.0, 1.0)));

        data.x_data_mut()[2] = 5.0;

        assert_eq!(data.x_axis().as_uniform(), None);
        assert_eq!(data.x_data(), arr1(&[0.0, 1.0, 5.0]));
    }

    #[test]
    fn time_data_errors() {
        let error = RealData::new(
            array![0.0, 1.0, 2.0, 5.0],
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        .err();
        assert!(matches!(error, Some(DataError::NotMatching)));

        let error = RealData::new(
            array![0.0, 0.0, 0.0],
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        .err();
        assert!(matches!(error, Some(DataError::NotIncreasing)));

        let error = RealData::from_x_data(array![0.0, 0.0, 0.0], 3).err();
        assert!(matches!(error, Some(DataError::NotIncreasing)));
    }

    #[test]
    fn time_data() -> Result<(), DataError> {
        let data = RealData::zeros(2, 4);
        assert_eq!(data.x_data(), arr1(&[0.0, 1.0, 2.0, 3.0]));
        assert_eq!(data.y_data(), arr2(&[[0.0; 4]; 2]),);

        let data = RealData::from_x_data(arr1(&[0.0, 1.0, 2.0, 3.0]), 2).unwrap();
        assert_eq!(data.x_data(), arr1(&[0.0, 1.0, 2.0, 3.0]));
        assert_eq!(data.y_data(), arr2(&[[0.0; 4]; 2]),);

        let mut i = 0;
        let mut data = RealData::new(
            Array1::from_iter((0..10).map(|v| v as f64 / 10.0)),
            Array2::from_shape_fn((2, 10), |_| {
                let value = i as f64;
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
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
            ])
        );

        assert_eq!(
            data.y_data_mut(),
            arr2(&[
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
            ])
        );

        assert_eq!(
            data.channel(0),
            arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        );
        assert_eq!(
            data.channel(1),
            arr1(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        );

        assert_eq!(
            data.channel_mut(0),
            arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        );
        assert_eq!(
            data.channel_mut(1),
            arr1(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        );

        for (i, ch) in data.channel_iter().enumerate() {
            match i {
                0 => assert_eq!(
                    ch,
                    arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                ),
                1 => assert_eq!(
                    ch,
                    arr1(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
                ),
                _ => panic!("Should only have two channels"),
            }
        }

        for (i, ch) in data.channel_iter_mut().enumerate() {
            match i {
                0 => assert_eq!(
                    ch,
                    arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                ),
                1 => assert_eq!(
                    ch,
                    arr1(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
                ),
                _ => panic!("Should only have two channels"),
            }
        }

        Ok(())
    }
}
