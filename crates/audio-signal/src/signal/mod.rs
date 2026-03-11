use ndarray::prelude::*;

pub use freq_signal::FreqSignal;
pub use spectrogram::{Spectrogram, SpectrogramNormalization};
pub use time_signal::TimeSignal;

pub mod freq_signal;
pub mod spectrogram;
pub mod time_signal;

#[derive(Debug, thiserror::Error)]
pub enum SignalError {
    #[error("Sample Rate must be > 0")]
    SampleRateZeroOrNeg,
    #[error("At least one signal is required")]
    EmptySignalList,
    #[error("All signals must have the same sample rate")]
    SampleRateMismatch,
    #[error("All signals must have the same number of time steps")]
    TimeStepMismatch,
}

pub fn join_channels(signals: &[&TimeSignal]) -> Result<TimeSignal, SignalError> {
    let Some(first_signal) = signals.first() else {
        return Err(SignalError::EmptySignalList);
    };

    let sample_rate = first_signal.sample_rate();
    let num_time_steps = first_signal.num_time_steps();

    if signals
        .iter()
        .any(|signal| signal.sample_rate() != sample_rate)
    {
        return Err(SignalError::SampleRateMismatch);
    }
    if signals
        .iter()
        .any(|signal| signal.num_time_steps() != num_time_steps)
    {
        return Err(SignalError::TimeStepMismatch);
    }

    let total_channels = signals.iter().map(|signal| signal.num_channels()).sum();
    let mut data = Array2::zeros((total_channels, num_time_steps));
    let mut next_channel = 0;

    for signal in signals {
        let end_channel = next_channel + signal.num_channels();
        data.slice_mut(s![next_channel..end_channel, ..])
            .assign(&signal.time_data());
        next_channel = end_channel;
    }

    let mut stacked = TimeSignal::new(data, sample_rate)?;
    let comments = signals
        .iter()
        .enumerate()
        .filter_map(|(index, signal)| {
            signal
                .comment()
                .map(|comment| format!("signal{}: {}", index + 1, comment))
        })
        .collect::<Vec<_>>();
    if !comments.is_empty() {
        stacked.set_comment(Some(&comments.join(" | ")));
    }

    Ok(stacked)
}

pub mod utils {
    use super::*;

    pub fn f_from_t(t: usize) -> usize {
        t / 2 + 1
    }

    pub fn t_from_f(f: usize) -> usize {
        f.saturating_sub(1) * 2
    }

    pub fn generate_time_steps(num: usize, sample_rate: f64) -> Array1<f64> {
        if num == 0 {
            return Array1::zeros(0);
        }
        let end_time = (num - 1) as f64 / sample_rate;
        Array1::linspace(0.0, end_time, num)
    }

    pub fn generate_freq_steps(num: usize, sample_rate: f64, num_time_steps: usize) -> Array1<f64> {
        if num == 0 || num_time_steps == 0 {
            return Array1::zeros(num);
        }
        let end_freq = (num - 1) as f64 * sample_rate / num_time_steps as f64;
        Array1::linspace(0.0, end_freq, num)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn join_channels_from_multiple_signals() -> Result<(), SignalError> {
        let mut first = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), 48_000.0)?;
        first.set_comment(Some("first"));
        let second = TimeSignal::new(arr2(&[[6.0, 7.0, 8.0]]), 48_000.0)?;

        let joined = join!(first, second)?;

        assert_eq!(joined.num_channels(), 3);
        assert_eq!(
            joined.time_data(),
            arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        );
        assert_eq!(joined.comment(), Some("signal1: first"));
        Ok(())
    }

    #[test]
    fn join_channels_rejects_empty_input() {
        let result = join_channels(&[]);
        assert!(matches!(result, Err(SignalError::EmptySignalList)));
    }

    #[test]
    fn join_channels_rejects_mismatched_inputs() -> Result<(), SignalError> {
        let base = TimeSignal::new(arr2(&[[0.0, 1.0]]), 48_000.0)?;
        let different_rate = TimeSignal::new(arr2(&[[2.0, 3.0]]), 44_100.0)?;
        let different_length = TimeSignal::new(arr2(&[[2.0, 3.0, 4.0]]), 48_000.0)?;

        let result = join!(base, different_rate);
        assert!(matches!(result, Err(SignalError::SampleRateMismatch)));

        let result = join!(base, different_length);
        assert!(matches!(result, Err(SignalError::TimeStepMismatch)));
        Ok(())
    }
}
