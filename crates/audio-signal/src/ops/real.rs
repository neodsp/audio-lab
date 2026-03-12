use crate::{
    data::RealData,
    math::{db_to_gain, gain_to_db},
};

impl RealData {
    pub fn max_per_channel(&self) -> Vec<(f64, f64)> {
        self.y_data()
            .outer_iter()
            .map(|channel| {
                let (max_index, max_value) = channel
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("RealData channels always have consistent lengths");

                (self.x_data()[max_index], *max_value)
            })
            .collect()
    }

    pub fn max_overall(&self) -> (f64, f64) {
        *self
            .max_per_channel()
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("RealData must contain at least one sample")
    }

    pub fn max_abs_per_channel(&self) -> Vec<(f64, f64)> {
        self.y_data()
            .outer_iter()
            .map(|channel| {
                let (max_index, max_abs_value) = channel
                    .iter()
                    .enumerate()
                    .map(|(idx, &value)| (idx, value.abs()))
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("RealData channels always have consistent lengths");

                (self.x_data()[max_index], max_abs_value)
            })
            .collect()
    }

    pub fn max_abs_overall(&self) -> (f64, f64) {
        *self
            .max_abs_per_channel()
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("RealData must contain at least one sample")
    }

    pub fn min_max_per_channel(&self) -> Vec<((f64, f64), (f64, f64))> {
        self.y_data()
            .outer_iter()
            .map(|channel| {
                let (min_index, min_value) = channel
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("RealData channels always have consistent lengths");
                let (max_index, max_value) = channel
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("RealData channels always have consistent lengths");

                (
                    (self.x_data()[min_index], *min_value),
                    (self.x_data()[max_index], *max_value),
                )
            })
            .collect()
    }

    pub fn min_max_overall(&self) -> ((f64, f64), (f64, f64)) {
        let per_channel = self.min_max_per_channel();

        let min = per_channel
            .iter()
            .map(|&(min, _)| min)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("RealData must contain at least one sample");
        let max = per_channel
            .iter()
            .map(|&(_, max)| max)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("RealData must contain at least one sample");

        (min, max)
    }

    pub fn to_decibels(&self) -> RealData {
        RealData::new(self.x_data().to_owned(), self.y_data().mapv(gain_to_db))
            .expect("x/y data originated from RealData")
    }

    pub fn to_gain(&self) -> RealData {
        RealData::new(self.x_data().to_owned(), self.y_data().mapv(db_to_gain))
            .expect("x/y data originated from RealData")
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, aview2};

    use crate::data::RealData;

    #[test]
    fn extrema_helpers() {
        let data = RealData::new(
            ndarray::arr1(&[0.0, 1.0, 2.0, 3.0, 4.0]),
            arr2(&[[0.0, 1.0, 5.0, -11.0, 1.0], [0.0, -10.0, 0.0, 6.0, 1.0]]),
        )
        .unwrap();

        assert_eq!(data.max_per_channel(), vec![(2.0, 5.0), (3.0, 6.0)]);
        assert_eq!(data.max_overall(), (3.0, 6.0));
        assert_eq!(data.max_abs_per_channel(), vec![(3.0, 11.0), (1.0, 10.0)]);
        assert_eq!(data.max_abs_overall(), (3.0, 11.0));
        assert_eq!(
            data.min_max_per_channel(),
            vec![((3.0, -11.0), (2.0, 5.0)), ((1.0, -10.0), (3.0, 6.0))]
        );
        assert_eq!(data.min_max_overall(), ((3.0, -11.0), (3.0, 6.0)));
    }

    #[test]
    fn db_gain_conversion_helpers() {
        let gain = RealData::new(ndarray::arr1(&[1.0, 2.0]), arr2(&[[1.0, 10.0]])).unwrap();
        let db = gain.to_decibels();
        assert_abs_diff_eq!(db.channel(0)[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(db.channel(0)[1], 20.0, epsilon = 1e-12);

        let roundtrip = db.to_gain();
        assert!(
            roundtrip
                .y_data()
                .abs_diff_eq(&aview2(&[[1.0, 10.0]]), 1e-12)
        );
    }
}
