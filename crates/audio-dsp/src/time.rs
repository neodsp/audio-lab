use std::time::Duration;

use ndarray::s;

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
    fn trim_rejects_invalid_ranges() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0]]), 1.0).unwrap();
        let result = trim_samples(&signal, 2, 2);

        assert!(matches!(result, Err(TimeError::InvalidTrimRange { .. })));
    }
}
