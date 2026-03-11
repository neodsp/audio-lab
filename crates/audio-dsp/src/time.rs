use std::time::Duration;

use ndarray::s;

use audio_signal::math::db_to_gain;
use audio_signal::signal::{SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum TimeError {
    #[error("invalid trim range: start={start}, end={end}, signal_len={signal_len}")]
    InvalidTrimRange {
        start: usize,
        end: usize,
        signal_len: usize,
    },
    #[error(transparent)]
    Signal(#[from] SignalError),
}

pub fn trim_samples(
    signal: &TimeSignal,
    start_sample: usize,
    end_sample: i64,
) -> Result<TimeSignal, TimeError> {
    let signal_len = signal.num_time_steps();
    let end_sample = if end_sample < 0 {
        signal_len.saturating_sub((-end_sample) as usize)
    } else {
        end_sample as usize
    };

    if start_sample >= end_sample || end_sample > signal_len {
        return Err(TimeError::InvalidTrimRange {
            start: start_sample,
            end: end_sample,
            signal_len,
        });
    }

    Ok(TimeSignal::new(
        signal
            .time_data()
            .slice(s![.., start_sample..end_sample])
            .to_owned(),
        signal.sample_rate(),
    )?)
}

pub fn trim_duration(
    signal: &TimeSignal,
    start_duration: Duration,
    end_duration: Duration,
) -> Result<TimeSignal, TimeError> {
    let start_sample = (signal.sample_rate() * start_duration.as_secs_f64()).round() as usize;
    let end_sample = (signal.sample_rate() * end_duration.as_secs_f64()).round() as i64;
    trim_samples(signal, start_sample, end_sample)
}

pub fn normalize_peak_in_place_linear(signal: &mut TimeSignal, peak_level: f64) {
    let max = signal.iter().map(|sample| sample.abs()).fold(0.0, f64::max);
    if max == 0.0 {
        return;
    }

    let gain = peak_level / max;
    signal.iter_mut().for_each(|sample| *sample *= gain);
}

pub fn normalize_peak_linear(
    signal: &TimeSignal,
    peak_level: f64,
) -> Result<TimeSignal, SignalError> {
    let mut normalized = TimeSignal::new(signal.time_data().to_owned(), signal.sample_rate())?;
    normalize_peak_in_place_linear(&mut normalized, peak_level);
    Ok(normalized)
}

pub fn normalize_peak_in_place_db(signal: &mut TimeSignal, peak_level_db: f64) {
    normalize_peak_in_place_linear(signal, db_to_gain(peak_level_db));
}

pub fn normalize_peak_db(
    signal: &TimeSignal,
    peak_level_db: f64,
) -> Result<TimeSignal, SignalError> {
    normalize_peak_linear(signal, db_to_gain(peak_level_db))
}

pub fn normalize_in_place(signal: &mut TimeSignal, peak_level: f64) {
    normalize_peak_in_place_linear(signal, peak_level);
}

pub fn normalize(signal: &TimeSignal, peak_level: f64) -> Result<TimeSignal, SignalError> {
    normalize_peak_linear(signal, peak_level)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    use super::*;

    #[test]
    fn trim_samples_accepts_positive_and_negative_end() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0, 3.0, 4.0]]), 10.0).unwrap();

        let trimmed = trim_samples(&signal, 1, 4).unwrap();
        assert_eq!(trimmed.num_time_steps(), 3);
        assert_abs_diff_eq!(trimmed.channel(0)[0], 1.0, epsilon = 1e-12);

        let trimmed = trim_samples(&signal, 1, -1).unwrap();
        assert_eq!(trimmed.num_time_steps(), 3);
        assert_abs_diff_eq!(trimmed.channel(0)[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn trim_duration_maps_to_sample_positions() {
        let signal = TimeSignal::new(arr2(&[[0.0; 20]]), 10.0).unwrap();
        let trimmed = trim_duration(
            &signal,
            Duration::from_secs_f64(0.2),
            Duration::from_secs_f64(1.1),
        )
        .unwrap();

        assert_eq!(trimmed.num_time_steps(), 9);
    }

    #[test]
    fn normalize_scales_to_requested_peak_linear() {
        let signal = TimeSignal::new(arr2(&[[0.25, -0.5, 1.0]]), 48_000.0).unwrap();
        let normalized = normalize_peak_linear(&signal, 0.5).unwrap();

        assert_abs_diff_eq!(normalized.channel(0)[0], 0.125, epsilon = 1e-12);
        assert_abs_diff_eq!(normalized.channel(0)[1], -0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(normalized.channel(0)[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn normalize_scales_to_requested_peak_db() {
        let signal = TimeSignal::new(arr2(&[[0.25, -0.5, 1.0]]), 48_000.0).unwrap();
        let normalized = normalize_peak_db(&signal, -6.0).unwrap();

        let expected_peak = 10.0_f64.powf(-6.0 / 20.0);
        assert_abs_diff_eq!(
            normalized.channel(0)[0],
            0.25 * expected_peak,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            normalized.channel(0)[1],
            -0.5 * expected_peak,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(normalized.channel(0)[2], expected_peak, epsilon = 1e-12);
    }

    #[test]
    fn trim_rejects_invalid_ranges() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0]]), 1.0).unwrap();
        let result = trim_samples(&signal, 2, 2);

        assert!(matches!(result, Err(TimeError::InvalidTrimRange { .. })));
    }
}
