#[inline(always)]
pub fn db_to_gain(decibels: f64) -> f64 {
    10.0_f64.powf(decibels * 0.05)
}

#[inline(always)]
pub fn gain_to_db(gain: f64) -> f64 {
    gain.log10() * 20.0
}

#[inline(always)]
pub fn wrap_phase_2pi(phase: f64) -> f64 {
    let wrapped = phase.rem_euclid(2.0 * std::f64::consts::PI);
    if phase > 0.0 && wrapped == 0.0 {
        2.0 * std::f64::consts::PI
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn decibel_gain_roundtrip_examples() {
        assert_eq!(db_to_gain(0.0), 1.0);
        assert_abs_diff_eq!(db_to_gain(6.0), 1.9952623149688795, epsilon = 1e-12);
        assert_abs_diff_eq!(db_to_gain(-6.0), 0.5011872336272722, epsilon = 1e-12);

        assert_eq!(gain_to_db(1.0), 0.0);
        assert_abs_diff_eq!(gain_to_db(db_to_gain(6.0)), 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(gain_to_db(db_to_gain(-6.0)), -6.0, epsilon = 1e-12);
    }

    #[test]
    fn wrap_phase_2pi_matches_expected_range() {
        let two_pi = 2.0 * std::f64::consts::PI;

        assert_abs_diff_eq!(
            wrap_phase_2pi(-std::f64::consts::FRAC_PI_2),
            1.5 * std::f64::consts::PI,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            wrap_phase_2pi(std::f64::consts::FRAC_PI_2),
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(wrap_phase_2pi(two_pi), two_pi, epsilon = 1e-12);
    }
}
