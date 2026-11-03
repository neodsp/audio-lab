use ndarray::{Array1, CowArray, Ix1};

use super::DataError;

/// The x axis of [`RealData`](super::RealData) and
/// [`ComplexData`](super::ComplexData): time steps, frequency bins, or any
/// other strictly increasing positions.
///
/// A uniform axis stores only `start`, `step` and the length, and computes
/// values on demand. An explicit axis stores every value. Both kinds are
/// strictly increasing, which the constructors check.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct XAxis(Repr);

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Repr {
    /// `x[i] = start + i * step`
    Uniform {
        start: f64,
        step: f64,
        len: usize,
    },
    Explicit(Array1<f64>),
}

impl XAxis {
    /// A uniform axis `x[i] = start + i * step` with `len` points.
    ///
    /// `start` must be finite and `step` must be finite and positive.
    pub fn uniform(start: f64, step: f64, len: usize) -> Result<Self, DataError> {
        if !start.is_finite() || !step.is_finite() || step <= 0.0 {
            return Err(DataError::InvalidUniformAxis { start, step });
        }
        Ok(Self(Repr::Uniform { start, step, len }))
    }

    /// An axis with the given values, which must be strictly increasing.
    pub fn explicit(values: Array1<f64>) -> Result<Self, DataError> {
        if !values.windows(2).into_iter().all(|pair| pair[1] > pair[0]) {
            return Err(DataError::NotIncreasing);
        }
        Ok(Self(Repr::Explicit(values)))
    }

    /// The axis `0, 1, 2, …, len - 1`.
    pub fn indices(len: usize) -> Self {
        Self(Repr::Uniform {
            start: 0.0,
            step: 1.0,
            len,
        })
    }

    pub fn len(&self) -> usize {
        match &self.0 {
            Repr::Uniform { len, .. } => *len,
            Repr::Explicit(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `Some((start, step))` if the axis is uniform.
    pub fn as_uniform(&self) -> Option<(f64, f64)> {
        match &self.0 {
            Repr::Uniform { start, step, .. } => Some((*start, *step)),
            Repr::Explicit(_) => None,
        }
    }

    /// The value at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    pub fn get(&self, index: usize) -> f64 {
        match &self.0 {
            Repr::Uniform { start, step, len } => {
                assert!(
                    index < *len,
                    "index {index} out of bounds for axis of length {len}"
                );
                start + index as f64 * step
            }
            Repr::Explicit(values) => values[index],
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        (0..self.len()).map(move |index| match &self.0 {
            Repr::Uniform { start, step, .. } => start + index as f64 * step,
            Repr::Explicit(values) => values[index],
        })
    }

    /// The values as an array. Borrowed for an explicit axis, computed for a
    /// uniform one.
    pub fn to_cow_array(&self) -> CowArray<'_, f64, Ix1> {
        match &self.0 {
            Repr::Uniform { .. } => CowArray::from(self.to_array()),
            Repr::Explicit(values) => CowArray::from(values.view()),
        }
    }

    /// The values as an owned array.
    pub fn to_array(&self) -> Array1<f64> {
        match &self.0 {
            Repr::Uniform { .. } => Array1::from_iter(self.iter()),
            Repr::Explicit(values) => values.clone(),
        }
    }

    /// Mutable view of an explicit axis, for `x_data_mut`.
    ///
    /// # Panics
    ///
    /// Panics on a uniform axis.
    pub(super) fn explicit_values_mut(&mut self) -> ndarray::ArrayViewMut1<'_, f64> {
        match &mut self.0 {
            Repr::Explicit(values) => values.view_mut(),
            Repr::Uniform { .. } => panic!("uniform axis has no stored values"),
        }
    }

    /// Index of the value closest to `x`. On a tie the lower index wins.
    /// Returns `0` for an empty axis or a NaN `x`.
    pub fn nearest_index(&self, x: f64) -> usize {
        let len = self.len();
        if len == 0 || x.is_nan() {
            return 0;
        }

        let upper = match &self.0 {
            Repr::Uniform { start, step, .. } => {
                // First index whose value is not below x, then fix up for rounding.
                let mut upper = ((x - start) / step).ceil().clamp(0.0, len as f64) as usize;
                while upper > 0 && self.get(upper - 1) >= x {
                    upper -= 1;
                }
                while upper < len && self.get(upper) < x {
                    upper += 1;
                }
                upper
            }
            Repr::Explicit(values) => values
                .as_slice()
                .expect("owned 1-D array is contiguous")
                .partition_point(|&value| value < x),
        };

        if upper == 0 {
            return 0;
        }
        if upper == len {
            return len - 1;
        }

        let lower = upper - 1;
        if x - self.get(lower) <= self.get(upper) - x {
            lower
        } else {
            upper
        }
    }
}

impl PartialEq for XAxis {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if let (
            Repr::Uniform {
                start: a_start,
                step: a_step,
                len,
            },
            Repr::Uniform {
                start: b_start,
                step: b_step,
                ..
            },
        ) = (&self.0, &other.0)
        {
            // With 0 or 1 points the step does not affect any value.
            return *len == 0 || (a_start == b_start && (*len == 1 || a_step == b_step));
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl approx::AbsDiffEq for XAxis {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| a.abs_diff_eq(&b, epsilon))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn constructors_validate_input() {
        assert!(XAxis::uniform(0.0, 0.5, 4).is_ok());
        assert!(XAxis::uniform(0.0, 0.0, 4).is_err());
        assert!(XAxis::uniform(0.0, -1.0, 4).is_err());
        assert!(XAxis::uniform(f64::NAN, 1.0, 4).is_err());
        assert!(XAxis::uniform(0.0, f64::INFINITY, 4).is_err());

        assert!(XAxis::explicit(array![0.0, 1.0, 5.0]).is_ok());
        assert!(matches!(
            XAxis::explicit(array![0.0, 1.0, 1.0]),
            Err(DataError::NotIncreasing)
        ));
    }

    #[test]
    fn uniform_and_explicit_axes_expose_the_same_values() {
        let uniform = XAxis::uniform(1.0, 0.25, 5).unwrap();
        let explicit = XAxis::explicit(array![1.0, 1.25, 1.5, 1.75, 2.0]).unwrap();

        assert_eq!(uniform.len(), 5);
        assert_eq!(uniform.as_uniform(), Some((1.0, 0.25)));
        assert_eq!(explicit.as_uniform(), None);
        assert_eq!(uniform.get(3), 1.75);
        assert_eq!(
            uniform.iter().collect::<Vec<_>>(),
            explicit.iter().collect::<Vec<_>>()
        );
        assert_eq!(uniform.to_array(), explicit.to_array());
        assert_eq!(uniform.to_cow_array(), explicit.to_cow_array());
        assert_eq!(uniform, explicit);
        assert_eq!(
            XAxis::indices(3),
            XAxis::explicit(array![0.0, 1.0, 2.0]).unwrap()
        );
    }

    #[test]
    fn equality_compares_values() {
        let a = XAxis::uniform(0.0, 1.0, 3).unwrap();
        assert_ne!(a, XAxis::uniform(0.0, 2.0, 3).unwrap());
        assert_ne!(a, XAxis::uniform(0.0, 1.0, 4).unwrap());
        // Step does not matter for a single point.
        assert_eq!(
            XAxis::uniform(2.0, 1.0, 1).unwrap(),
            XAxis::uniform(2.0, 5.0, 1).unwrap()
        );
        assert_eq!(
            XAxis::uniform(2.0, 1.0, 0).unwrap(),
            XAxis::explicit(Array1::zeros(0)).unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn uniform_get_panics_out_of_bounds() {
        XAxis::uniform(0.0, 1.0, 3).unwrap().get(3);
    }

    #[test]
    fn nearest_index_matches_linear_scan_for_both_kinds() {
        let linear_scan = |axis: &XAxis, x: f64| {
            axis.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a - x)
                        .abs()
                        .partial_cmp(&(b - x).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index)
                .unwrap()
        };

        let axes = [
            XAxis::uniform(0.0, 0.1, 11).unwrap(),
            XAxis::uniform(-3.0, 1.5, 7).unwrap(),
            XAxis::explicit(array![0.0, 1.0, 2.5, 4.0, 10.0]).unwrap(),
        ];
        for axis in &axes {
            let lo = axis.get(0) - 2.0;
            let hi = axis.get(axis.len() - 1) + 2.0;
            for i in 0..=200 {
                let x = lo + (hi - lo) * i as f64 / 200.0;
                assert_eq!(axis.nearest_index(x), linear_scan(axis, x), "x = {x}");
            }
            for i in 0..axis.len() {
                // Exact hits and midpoints between neighbours.
                assert_eq!(axis.nearest_index(axis.get(i)), i);
                if i + 1 < axis.len() {
                    let mid = 0.5 * (axis.get(i) + axis.get(i + 1));
                    assert_eq!(axis.nearest_index(mid), linear_scan(axis, mid));
                }
            }
            assert_eq!(axis.nearest_index(f64::NAN), 0);
            assert_eq!(axis.nearest_index(f64::INFINITY), axis.len() - 1);
            assert_eq!(axis.nearest_index(f64::NEG_INFINITY), 0);
        }
        assert_eq!(XAxis::indices(0).nearest_index(1.0), 0);
    }
}
