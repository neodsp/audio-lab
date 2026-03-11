use crate::{
    data::RealData,
    math::{db_to_gain, gain_to_db},
};

pub fn max_per_channel(data: &RealData) -> Vec<(f64, f64)> {
    data.y_data()
        .outer_iter()
        .map(|channel| {
            let (max_index, max_value) = channel
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("RealData channels always have consistent lengths");

            (data.x_data()[max_index], *max_value)
        })
        .collect()
}

pub fn max_overall(data: &RealData) -> (f64, f64) {
    *max_per_channel(data)
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("RealData must contain at least one sample")
}

pub fn max_abs_per_channel(data: &RealData) -> Vec<(f64, f64)> {
    data.y_data()
        .outer_iter()
        .map(|channel| {
            let (max_index, max_abs_value) = channel
                .iter()
                .enumerate()
                .map(|(idx, &value)| (idx, value.abs()))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("RealData channels always have consistent lengths");

            (data.x_data()[max_index], max_abs_value)
        })
        .collect()
}

pub fn max_abs_overall(data: &RealData) -> (f64, f64) {
    *max_abs_per_channel(data)
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("RealData must contain at least one sample")
}

pub fn min_max_per_channel(data: &RealData) -> Vec<((f64, f64), (f64, f64))> {
    data.y_data()
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
                (data.x_data()[min_index], *min_value),
                (data.x_data()[max_index], *max_value),
            )
        })
        .collect()
}

pub fn min_max_overall(data: &RealData) -> ((f64, f64), (f64, f64)) {
    let per_channel = min_max_per_channel(data);

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

pub fn to_decibels(data: &RealData) -> RealData {
    RealData::new(data.x_data().to_owned(), data.y_data().mapv(gain_to_db))
        .expect("x/y data originated from RealData")
}

pub fn to_gain(data: &RealData) -> RealData {
    RealData::new(data.x_data().to_owned(), data.y_data().mapv(db_to_gain))
        .expect("x/y data originated from RealData")
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, aview2};

    use super::*;

    #[test]
    fn extrema_helpers() {
        let data = RealData::new(
            ndarray::arr1(&[0.0, 1.0, 2.0, 3.0, 4.0]),
            arr2(&[[0.0, 1.0, 5.0, -11.0, 1.0], [0.0, -10.0, 0.0, 6.0, 1.0]]),
        )
        .unwrap();

        assert_eq!(max_per_channel(&data), vec![(2.0, 5.0), (3.0, 6.0)]);
        assert_eq!(max_overall(&data), (3.0, 6.0));
        assert_eq!(max_abs_per_channel(&data), vec![(3.0, 11.0), (1.0, 10.0)]);
        assert_eq!(max_abs_overall(&data), (3.0, 11.0));
        assert_eq!(
            min_max_per_channel(&data),
            vec![((3.0, -11.0), (2.0, 5.0)), ((1.0, -10.0), (3.0, 6.0))]
        );
        assert_eq!(min_max_overall(&data), ((3.0, -11.0), (3.0, 6.0)));
    }

    #[test]
    fn db_gain_conversion_helpers() {
        let gain = RealData::new(ndarray::arr1(&[1.0, 2.0]), arr2(&[[1.0, 10.0]])).unwrap();
        let db = to_decibels(&gain);
        assert_abs_diff_eq!(db.channel(0)[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(db.channel(0)[1], 20.0, epsilon = 1e-12);

        let roundtrip = to_gain(&db);
        assert!(
            roundtrip
                .y_data()
                .abs_diff_eq(&aview2(&[[1.0, 10.0]]), 1e-12)
        );
    }
}
