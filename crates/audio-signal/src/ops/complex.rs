use crate::{
    data::{ComplexData, RealData},
    math::gain_to_db,
};

const NEG_INF_DB: f64 = -100.0;

pub fn to_magnitude(data: &ComplexData) -> RealData {
    RealData::new(
        data.x_data().to_owned(),
        data.y_data().mapv(|value| value.norm()),
    )
    .expect("x/y data originated from ComplexData")
}

pub fn to_magnitude_db(data: &ComplexData) -> RealData {
    RealData::new(
        data.x_data().to_owned(),
        data.y_data()
            .mapv(|value| gain_to_db(value.norm()).max(NEG_INF_DB)),
    )
    .expect("x/y data originated from ComplexData")
}

pub fn to_phase(data: &ComplexData) -> RealData {
    RealData::new(
        data.x_data().to_owned(),
        data.y_data().mapv(|value| value.arg()),
    )
    .expect("x/y data originated from ComplexData")
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};
    use num::complex::Complex64;

    use super::*;

    #[test]
    fn complex_data_conversions() {
        let data = ComplexData::new(
            arr1(&[10.0, 20.0, 30.0]),
            arr2(&[[
                Complex64::new(3.0, 4.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-1.0, 0.0),
            ]]),
        )
        .unwrap();

        let magnitude = to_magnitude(&data);
        assert_abs_diff_eq!(magnitude.channel(0)[0], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(magnitude.channel(0)[1], 1.0, epsilon = 1e-12);

        let magnitude_db = to_magnitude_db(&data);
        assert_abs_diff_eq!(
            magnitude_db.channel(0)[0],
            20.0 * 5.0_f64.log10(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(magnitude_db.channel(0)[1], 0.0, epsilon = 1e-12);

        let phase = to_phase(&data);
        assert_abs_diff_eq!(phase.channel(0)[0], (4.0_f64).atan2(3.0), epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[1], FRAC_PI_2, epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[2], PI, epsilon = 1e-12);
    }

    #[test]
    fn complex_data_phase_quadrants() {
        let data = ComplexData::new(
            arr1(&[0.0, 1.0, 2.0, 3.0]),
            arr2(&[[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, -1.0),
            ]]),
        )
        .unwrap();

        let phase = to_phase(&data);
        assert_abs_diff_eq!(phase.channel(0)[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[1], FRAC_PI_2, epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[2], FRAC_PI_4, epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[3], -FRAC_PI_4, epsilon = 1e-12);
    }
}
