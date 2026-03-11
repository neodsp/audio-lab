#[inline(always)]
pub fn db_to_gain(decibels: f64) -> f64 {
    10.0_f64.powf(decibels * 0.05)
}

#[inline(always)]
pub fn gain_to_db(gain: f64) -> f64 {
    gain.log10() * 20.0
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
}
