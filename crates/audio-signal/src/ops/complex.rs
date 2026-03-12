use crate::{
    data::{ComplexData, RealData},
    math::{gain_to_db, wrap_phase_2pi},
    signal::FreqSignal,
};

const NEG_INF_DB: f64 = -100.0;

impl ComplexData {
    pub fn to_magnitude(&self) -> RealData {
        RealData::new(
            self.x_data().to_owned(),
            self.y_data().mapv(|value| value.norm()),
        )
        .expect("x/y data originated from ComplexData")
    }

    pub fn to_magnitude_db(&self) -> RealData {
        RealData::new(
            self.x_data().to_owned(),
            self.y_data()
                .mapv(|value| gain_to_db(value.norm()).max(NEG_INF_DB)),
        )
        .expect("x/y data originated from ComplexData")
    }

    pub fn to_phase(&self) -> RealData {
        RealData::new(
            self.x_data().to_owned(),
            self.y_data().mapv(|value| value.arg()),
        )
        .expect("x/y data originated from ComplexData")
    }

    pub fn to_phase_unwrapped(&self) -> RealData {
        let mut phase = self.y_data().mapv(|value| value.arg());

        for mut channel in phase.outer_iter_mut() {
            unwrap_phase_in_place(channel.as_slice_mut().expect("channel view is contiguous"));
        }

        RealData::new(self.x_data().to_owned(), phase)
            .expect("x/y data originated from ComplexData")
    }

    pub fn to_phase_2pi(&self) -> RealData {
        RealData::new(
            self.x_data().to_owned(),
            self.y_data().mapv(|value| wrap_phase_2pi(value.arg())),
        )
        .expect("x/y data originated from ComplexData")
    }

    pub fn to_phase_degrees(&self) -> RealData {
        RealData::new(
            self.x_data().to_owned(),
            self.y_data().mapv(|value| value.arg().to_degrees()),
        )
        .expect("x/y data originated from ComplexData")
    }

    pub fn to_phase_degrees_unwrapped(&self) -> RealData {
        let mut phase = self.y_data().mapv(|value| value.arg());

        for mut channel in phase.outer_iter_mut() {
            unwrap_phase_in_place(channel.as_slice_mut().expect("channel view is contiguous"));
        }

        RealData::new(self.x_data().to_owned(), phase.mapv(f64::to_degrees))
            .expect("x/y data originated from ComplexData")
    }
}

impl FreqSignal {
    pub fn to_magnitude(&self) -> RealData {
        self.data().to_magnitude()
    }

    pub fn to_magnitude_db(&self) -> RealData {
        self.data().to_magnitude_db()
    }

    pub fn to_phase(&self) -> RealData {
        self.data().to_phase()
    }

    pub fn to_phase_unwrapped(&self) -> RealData {
        self.data().to_phase_unwrapped()
    }

    pub fn to_phase_2pi(&self) -> RealData {
        self.data().to_phase_2pi()
    }

    pub fn to_phase_degrees(&self) -> RealData {
        self.data().to_phase_degrees()
    }

    pub fn to_phase_degrees_unwrapped(&self) -> RealData {
        self.data().to_phase_degrees_unwrapped()
    }
}

fn unwrap_phase_in_place(phase: &mut [f64]) {
    if phase.len() < 2 {
        return;
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut offset = 0.0;
    let mut previous = phase[0];

    for index in 1..phase.len() {
        let current = phase[index];
        let delta = current - previous;
        if delta > std::f64::consts::PI {
            offset -= two_pi;
        } else if delta < -std::f64::consts::PI {
            offset += two_pi;
        }
        phase[index] = current + offset;
        previous = current;
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};
    use num::complex::Complex64;

    use crate::data::ComplexData;

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

        let magnitude = data.to_magnitude();
        assert_abs_diff_eq!(magnitude.channel(0)[0], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(magnitude.channel(0)[1], 1.0, epsilon = 1e-12);

        let magnitude_db = data.to_magnitude_db();
        assert_abs_diff_eq!(
            magnitude_db.channel(0)[0],
            20.0 * 5.0_f64.log10(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(magnitude_db.channel(0)[1], 0.0, epsilon = 1e-12);

        let phase = data.to_phase();
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

        let phase = data.to_phase();
        assert_abs_diff_eq!(phase.channel(0)[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[1], FRAC_PI_2, epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[2], FRAC_PI_4, epsilon = 1e-12);
        assert_abs_diff_eq!(phase.channel(0)[3], -FRAC_PI_4, epsilon = 1e-12);
    }

    #[test]
    fn complex_data_phase_variants() {
        let data = ComplexData::new(
            arr1(&[0.0, 1.0, 2.0]),
            arr2(&[[
                Complex64::from_polar(1.0, 3.0 * PI / 4.0),
                Complex64::from_polar(1.0, -3.0 * PI / 4.0),
                Complex64::from_polar(1.0, -PI / 2.0),
            ]]),
        )
        .unwrap();

        let unwrapped = data.to_phase_unwrapped();
        let wrapped_2pi = data.to_phase_2pi();
        let degrees = data.to_phase_degrees();
        let degrees_unwrapped = data.to_phase_degrees_unwrapped();

        assert_abs_diff_eq!(unwrapped.channel(0)[0], 3.0 * PI / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(unwrapped.channel(0)[1], 5.0 * PI / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(unwrapped.channel(0)[2], 3.0 * PI / 2.0, epsilon = 1e-12);

        assert_abs_diff_eq!(wrapped_2pi.channel(0)[1], 5.0 * PI / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(wrapped_2pi.channel(0)[2], 3.0 * PI / 2.0, epsilon = 1e-12);

        assert_abs_diff_eq!(degrees.channel(0)[0], 135.0, epsilon = 1e-12);
        assert_abs_diff_eq!(degrees.channel(0)[1], -135.0, epsilon = 1e-12);
        assert_abs_diff_eq!(degrees_unwrapped.channel(0)[1], 225.0, epsilon = 1e-12);
        assert_abs_diff_eq!(degrees_unwrapped.channel(0)[2], 270.0, epsilon = 1e-12);
    }
}
