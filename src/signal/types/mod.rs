use ndarray::prelude::*;

use crate::signal::data::XAxis;

pub use freq_signal::FreqSignal;
pub use spectrogram::{Spectrogram, SpectrogramNormalization};
pub use time_signal::TimeSignal;

mod freq_signal;
mod spectrogram;
mod time_signal;

#[derive(Debug, thiserror::Error)]
pub enum SignalError {
    #[error("Sample rate must be finite and > 0")]
    SampleRateZeroOrNeg,
    #[error("At least one signal is required")]
    EmptySignalList,
    #[error("All signals must have the same sample rate")]
    SampleRateMismatch,
    #[error("All signals must have the same number of time steps")]
    TimeStepMismatch,
    #[error("All frequency signals must have the same number of bins")]
    FrequencyBinMismatch,
    #[error("All signals must have the same number of channels")]
    ChannelMismatch,
    #[error("num_time_steps must be > 0 when there are {num_freq_bins} frequency bins")]
    NumTimeStepsZero { num_freq_bins: usize },
    #[error("Window size must be > 0")]
    WindowSizeZero,
    #[error("Hop size must be > 0")]
    HopSizeZero,
    #[error(
        "A window of {window_size} samples gives {expected} frequency bins, but the data has {actual}"
    )]
    SpectrogramBinMismatch {
        window_size: usize,
        expected: usize,
        actual: usize,
    },
}

fn validate_sample_rate(sample_rate: f64) -> Result<(), SignalError> {
    if sample_rate.is_finite() && sample_rate > 0.0 {
        Ok(())
    } else {
        Err(SignalError::SampleRateZeroOrNeg)
    }
}

/// Uniform time axis `i / sample_rate` in seconds. `sample_rate` must be validated.
fn time_axis(num_time_steps: usize, sample_rate: f64) -> XAxis {
    XAxis::uniform(0.0, 1.0 / sample_rate, num_time_steps)
        .expect("validated sample rate gives a finite, positive step")
}

/// Uniform frequency axis `k · sample_rate / num_time_steps` in Hz.
/// `sample_rate` must be validated.
fn freq_axis(
    num_freq_bins: usize,
    sample_rate: f64,
    num_time_steps: usize,
) -> Result<XAxis, SignalError> {
    if num_time_steps == 0 {
        // No spacing is defined. A single bin sits at 0 Hz either way.
        if num_freq_bins > 1 {
            return Err(SignalError::NumTimeStepsZero { num_freq_bins });
        }
        return Ok(XAxis::indices(num_freq_bins));
    }
    Ok(
        XAxis::uniform(0.0, sample_rate / num_time_steps as f64, num_freq_bins)
            .expect("validated sample rate gives a finite, positive step"),
    )
}

pub fn join_time_signals(signals: &[TimeSignal]) -> Result<TimeSignal, SignalError> {
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
    propagate_channel_labels(&mut stacked, signals);
    Ok(stacked)
}

fn propagate_channel_labels(stacked: &mut TimeSignal, signals: &[TimeSignal]) {
    let mut next_channel = 0;
    for signal in signals {
        let multi_channel = signal.num_channels() > 1;
        for source_channel in 0..signal.num_channels() {
            let label = source_label(
                signal.channel_label(source_channel),
                signal.comment(),
                source_channel,
                multi_channel,
            );
            stacked.set_channel_label(next_channel, label.as_deref());
            next_channel += 1;
        }
    }
}

fn source_label(
    explicit_label: Option<&str>,
    comment: Option<&str>,
    source_channel: usize,
    multi_channel: bool,
) -> Option<String> {
    if let Some(label) = explicit_label {
        return Some(label.to_string());
    }
    let base = comment?;
    Some(if multi_channel {
        format!("{base} · Ch {source_channel}")
    } else {
        base.to_string()
    })
}

pub fn mix_time_signals(signals: &[TimeSignal]) -> Result<TimeSignal, SignalError> {
    let Some(first) = signals.first() else {
        return Err(SignalError::EmptySignalList);
    };

    let sample_rate = first.sample_rate();
    let num_time_steps = first.num_time_steps();
    let num_channels = first.num_channels();

    if signals.iter().any(|s| s.sample_rate() != sample_rate) {
        return Err(SignalError::SampleRateMismatch);
    }
    if signals.iter().any(|s| s.num_time_steps() != num_time_steps) {
        return Err(SignalError::TimeStepMismatch);
    }
    if signals.iter().any(|s| s.num_channels() != num_channels) {
        return Err(SignalError::ChannelMismatch);
    }

    let mut data = Array2::zeros((num_channels, num_time_steps));
    for signal in signals {
        data += &signal.time_data();
    }

    TimeSignal::new(data, sample_rate)
}

pub fn join_freq_signals(signals: &[FreqSignal]) -> Result<FreqSignal, SignalError> {
    let Some(first_signal) = signals.first() else {
        return Err(SignalError::EmptySignalList);
    };

    let sample_rate = first_signal.sample_rate();
    let num_time_steps = first_signal.num_time_steps();
    let num_freq_bins = first_signal.num_freq_bins();

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
    if signals
        .iter()
        .any(|signal| signal.num_freq_bins() != num_freq_bins)
    {
        return Err(SignalError::FrequencyBinMismatch);
    }

    let total_channels = signals.iter().map(|signal| signal.num_channels()).sum();
    let mut data = Array2::zeros((total_channels, num_freq_bins));
    let mut next_channel = 0;

    for signal in signals {
        let end_channel = next_channel + signal.num_channels();
        data.slice_mut(s![next_channel..end_channel, ..])
            .assign(&signal.freq_data());
        next_channel = end_channel;
    }

    let mut stacked = FreqSignal::new(data, sample_rate, Some(num_time_steps))?;
    propagate_channel_labels_freq(&mut stacked, signals);
    Ok(stacked)
}

fn propagate_channel_labels_freq(stacked: &mut FreqSignal, signals: &[FreqSignal]) {
    let mut next_channel = 0;
    for signal in signals {
        let multi_channel = signal.num_channels() > 1;
        for source_channel in 0..signal.num_channels() {
            let label = source_label(
                signal.channel_label(source_channel),
                signal.comment(),
                source_channel,
                multi_channel,
            );
            stacked.set_channel_label(next_channel, label.as_deref());
            next_channel += 1;
        }
    }
}

pub fn mix_freq_signals(signals: &[FreqSignal]) -> Result<FreqSignal, SignalError> {
    let Some(first) = signals.first() else {
        return Err(SignalError::EmptySignalList);
    };

    let sample_rate = first.sample_rate();
    let num_time_steps = first.num_time_steps();
    let num_freq_bins = first.num_freq_bins();
    let num_channels = first.num_channels();

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
    if signals
        .iter()
        .any(|signal| signal.num_freq_bins() != num_freq_bins)
    {
        return Err(SignalError::FrequencyBinMismatch);
    }
    if signals
        .iter()
        .any(|signal| signal.num_channels() != num_channels)
    {
        return Err(SignalError::ChannelMismatch);
    }

    let mut data = Array2::zeros((num_channels, num_freq_bins));
    for signal in signals {
        data += &signal.freq_data();
    }

    FreqSignal::new(data, sample_rate, Some(num_time_steps))
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
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;
    use num::complex::Complex64;

    use super::*;

    #[test]
    fn join_channels_from_multiple_signals() -> Result<(), SignalError> {
        let mut first = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), 48_000.0)?;
        first.set_comment(Some("first"));
        let second = TimeSignal::new(arr2(&[[6.0, 7.0, 8.0]]), 48_000.0)?;

        let joined = join_time!(first, second)?;

        assert_eq!(joined.num_channels(), 3);
        assert_eq!(
            joined.time_data(),
            arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        );
        assert_eq!(joined.comment(), None);
        assert_eq!(joined.channel_label(0), Some("first · Ch 0"));
        assert_eq!(joined.channel_label(1), Some("first · Ch 1"));
        assert_eq!(joined.channel_label(2), None);
        Ok(())
    }

    #[test]
    fn join_channels_rejects_empty_input() {
        let result = join_time_signals(&[]);
        assert!(matches!(result, Err(SignalError::EmptySignalList)));
    }

    #[test]
    fn join_channels_rejects_mismatched_inputs() -> Result<(), SignalError> {
        let base = TimeSignal::new(arr2(&[[0.0, 1.0]]), 48_000.0)?;
        let different_rate = TimeSignal::new(arr2(&[[2.0, 3.0]]), 44_100.0)?;
        let different_length = TimeSignal::new(arr2(&[[2.0, 3.0, 4.0]]), 48_000.0)?;

        let result = join_time!(base.clone(), different_rate);
        assert!(matches!(result, Err(SignalError::SampleRateMismatch)));

        let result = join_time!(base, different_length);
        assert!(matches!(result, Err(SignalError::TimeStepMismatch)));
        Ok(())
    }

    #[test]
    fn mix_combines_matching_signals() -> Result<(), SignalError> {
        let first = TimeSignal::new(arr2(&[[0.0, 1.0, 2.0]]), 48_000.0)?;
        let second = TimeSignal::new(arr2(&[[0.5, 1.5, -1.0]]), 48_000.0)?;

        let mixed = mix_time!(first, second)?;

        assert_eq!(mixed.num_channels(), 1);
        assert_eq!(mixed.time_data(), arr2(&[[0.5, 2.5, 1.0]]));
        Ok(())
    }

    #[test]
    fn mix_rejects_channel_mismatch() -> Result<(), SignalError> {
        let mono = TimeSignal::new(arr2(&[[0.0, 1.0]]), 48_000.0)?;
        let stereo = TimeSignal::new(arr2(&[[0.0, 1.0], [2.0, 3.0]]), 48_000.0)?;

        let result = mix_time!(mono, stereo);

        assert!(matches!(result, Err(SignalError::ChannelMismatch)));
        Ok(())
    }

    #[test]
    fn mixing_time_and_frequency_signals_produces_same_time_result() -> Result<(), SignalError> {
        let first = TimeSignal::new(
            arr2(&[[0.25, -0.5, 1.0, 0.75, -0.125, 0.0, 0.5, -0.25]]),
            48_000.0,
        )?;
        let second = TimeSignal::new(
            arr2(&[[0.5, 0.25, -0.75, 0.125, 0.875, -0.5, 0.0, 0.25]]),
            48_000.0,
        )?;

        let mixed_time = mix_time_signals(&[first.clone(), second.clone()])?;
        let mixed_freq = mix_freq_signals(&[first.into_freq(), second.into_freq()])?.into_time();

        assert_abs_diff_eq!(mixed_time, mixed_freq, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn join_frequency_channels_from_multiple_signals() -> Result<(), SignalError> {
        let mut first = FreqSignal::new(
            arr2(&[[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ]]),
            48_000.0,
            Some(4),
        )?;
        first.set_comment(Some("first"));
        let second = FreqSignal::new(
            arr2(&[[
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]]),
            48_000.0,
            Some(4),
        )?;

        let joined = join_freq_signals(&[first, second])?;

        assert_eq!(joined.num_channels(), 2);
        assert_eq!(
            joined.freq_data(),
            arr2(&[
                [
                    Complex64::new(1.0, 0.0),
                    Complex64::new(2.0, 0.0),
                    Complex64::new(3.0, 0.0),
                ],
                [
                    Complex64::new(4.0, 0.0),
                    Complex64::new(5.0, 0.0),
                    Complex64::new(6.0, 0.0),
                ]
            ])
        );
        assert_eq!(joined.comment(), None);
        assert_eq!(joined.channel_label(0), Some("first"));
        assert_eq!(joined.channel_label(1), None);
        Ok(())
    }

    #[test]
    fn join_frequency_channels_rejects_empty_input() {
        let result = join_freq_signals(&[]);
        assert!(matches!(result, Err(SignalError::EmptySignalList)));
    }

    #[test]
    fn join_frequency_channels_rejects_mismatched_inputs() -> Result<(), SignalError> {
        let base = FreqSignal::new(
            arr2(&[[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]]),
            48_000.0,
            Some(2),
        )?;
        let different_rate = FreqSignal::new(
            arr2(&[[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]]),
            44_100.0,
            Some(2),
        )?;
        let different_length = FreqSignal::new(
            arr2(&[[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]]),
            48_000.0,
            Some(4),
        )?;
        let different_bins = FreqSignal::new(
            arr2(&[[
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
            ]]),
            48_000.0,
            Some(2),
        )?;

        let result = join_freq_signals(&[base.clone(), different_rate]);
        assert!(matches!(result, Err(SignalError::SampleRateMismatch)));

        let result = join_freq_signals(&[base.clone(), different_length]);
        assert!(matches!(result, Err(SignalError::TimeStepMismatch)));

        let result = join_freq_signals(&[base, different_bins]);
        assert!(matches!(result, Err(SignalError::FrequencyBinMismatch)));
        Ok(())
    }

    #[test]
    fn mix_frequency_signals_combines_matching_spectra() -> Result<(), SignalError> {
        let first = FreqSignal::new(
            arr2(&[[
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, 6.0),
            ]]),
            48_000.0,
            Some(4),
        )?;
        let second = FreqSignal::new(
            arr2(&[[
                Complex64::new(-1.0, 1.0),
                Complex64::new(2.0, -2.0),
                Complex64::new(0.5, 0.5),
            ]]),
            48_000.0,
            Some(4),
        )?;

        let mixed = mix_freq_signals(&[first, second])?;

        assert_eq!(
            mixed.freq_data(),
            arr2(&[[
                Complex64::new(0.0, 3.0),
                Complex64::new(5.0, 2.0),
                Complex64::new(5.5, 6.5),
            ]])
        );
        Ok(())
    }

    #[test]
    fn mix_frequency_signals_rejects_mismatched_inputs() -> Result<(), SignalError> {
        let base = FreqSignal::new(
            arr2(&[[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]]),
            48_000.0,
            Some(2),
        )?;
        let different_rate = FreqSignal::new(
            arr2(&[[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]]),
            44_100.0,
            Some(2),
        )?;
        let different_length = FreqSignal::new(
            arr2(&[[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]]),
            48_000.0,
            Some(4),
        )?;
        let different_bins = FreqSignal::new(
            arr2(&[[
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
            ]]),
            48_000.0,
            Some(2),
        )?;
        let stereo = FreqSignal::new(
            arr2(&[
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
            ]),
            48_000.0,
            Some(2),
        )?;

        let result = mix_freq_signals(&[base.clone(), different_rate]);
        assert!(matches!(result, Err(SignalError::SampleRateMismatch)));

        let result = mix_freq_signals(&[base.clone(), different_length]);
        assert!(matches!(result, Err(SignalError::TimeStepMismatch)));

        let result = mix_freq_signals(&[base.clone(), different_bins]);
        assert!(matches!(result, Err(SignalError::FrequencyBinMismatch)));

        let result = mix_freq_signals(&[base, stereo]);
        assert!(matches!(result, Err(SignalError::ChannelMismatch)));
        Ok(())
    }
}

/// Reading data back runs the same checks as the constructors.
#[cfg(all(test, feature = "serde"))]
mod serde_validation_tests {
    use ndarray::arr2;
    use num::complex::Complex64;
    use serde_json::{Value, json};

    use super::*;
    use crate::signal::data::{ComplexData, RealData};

    fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(value: &T) -> T {
        serde_json::from_str(&serde_json::to_string(value).unwrap()).unwrap()
    }

    /// Serialize `value`, apply `edit` to the JSON, and try to read it back.
    fn read_edited<T: serde::Serialize + serde::de::DeserializeOwned>(
        value: &T,
        edit: impl FnOnce(&mut Value),
    ) -> Result<T, serde_json::Error> {
        let mut json = serde_json::to_value(value).unwrap();
        edit(&mut json);
        serde_json::from_value(json)
    }

    fn time_signal() -> TimeSignal {
        let mut signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0, 4.0]]), 8.0).unwrap();
        signal.set_comment(Some("comment"));
        signal.set_channel_label(0, Some("left"));
        signal
    }

    #[test]
    fn valid_values_round_trip_with_metadata() {
        let time = time_signal();
        let back = round_trip(&time);
        assert_eq!(back, time);
        assert_eq!(back.comment(), Some("comment"));
        assert_eq!(back.channel_label(0), Some("left"));

        let freq = time.to_freq();
        assert_eq!(round_trip(&freq), freq);

        let real =
            RealData::new(ndarray::arr1(&[0.0, 1.0, 5.0]), arr2(&[[1.0, 2.0, 3.0]])).unwrap();
        assert_eq!(round_trip(&real), real);
    }

    #[test]
    fn data_rejects_y_length_that_does_not_match_x_axis() {
        let real = RealData::from_y_data(arr2(&[[1.0, 2.0, 3.0]]));
        let error = read_edited(&real, |json| {
            json["y_data"] = json!({ "v": 1, "dim": [1, 2], "data": [1.0, 2.0] });
        })
        .unwrap_err();
        assert!(error.to_string().contains("same length"), "{error}");

        let complex = ComplexData::from_y_data(arr2(&[[Complex64::new(1.0, 0.0); 3]]));
        let error = read_edited(&complex, |json| {
            json["x_axis"] = json!({ "Uniform": { "start": 0.0, "step": 1.0, "len": 2 } });
        })
        .unwrap_err();
        assert!(error.to_string().contains("same length"), "{error}");
    }

    #[test]
    fn data_rejects_wrong_number_of_channel_labels() {
        let real = RealData::from_y_data(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let error = read_edited(&real, |json| json["channel_labels"] = json!([null])).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("1 channel labels for 2 channels"),
            "{error}"
        );
    }

    #[test]
    fn signals_store_only_what_cannot_be_derived() {
        let time = serde_json::to_value(time_signal()).unwrap();
        let mut keys: Vec<_> = time.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            ["channel_labels", "comment", "sample_rate", "samples"]
        );

        let freq = serde_json::to_value(time_signal().to_freq()).unwrap();
        let mut keys: Vec<_> = freq.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "channel_labels",
                "comment",
                "num_time_steps",
                "sample_rate",
                "spectrum"
            ]
        );
    }

    #[test]
    fn time_signal_rejects_invalid_sample_rate_and_label_count() {
        let signal = time_signal();

        let error = read_edited(&signal, |json| json["sample_rate"] = json!(-8.0)).unwrap_err();
        assert!(error.to_string().contains("Sample rate"), "{error}");

        let error =
            read_edited(&signal, |json| json["channel_labels"] = json!([null, null])).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("2 channel labels for 1 channels"),
            "{error}"
        );
    }

    #[test]
    fn freq_signal_rejects_invalid_sample_rate_time_steps_and_label_count() {
        let signal = time_signal().to_freq();

        let error = read_edited(&signal, |json| json["sample_rate"] = json!(0.0)).unwrap_err();
        assert!(error.to_string().contains("Sample rate"), "{error}");

        // No frequency spacing is defined; must be an error, not a panic.
        let error = read_edited(&signal, |json| json["num_time_steps"] = json!(0)).unwrap_err();
        assert!(
            error.to_string().contains("num_time_steps must be > 0"),
            "{error}"
        );

        let error = read_edited(&signal, |json| json["channel_labels"] = json!([])).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("0 channel labels for 1 channels"),
            "{error}"
        );
    }

    #[test]
    fn read_signal_rebuilds_axis_from_sample_rate() {
        let signal = time_signal();
        let back = round_trip(&signal);
        assert_eq!(back.time_axis().as_uniform(), Some((0.0, 1.0 / 8.0)));

        // Changing the stored sample rate changes the rebuilt axis with it.
        let resampled: TimeSignal =
            read_edited(&signal, |json| json["sample_rate"] = json!(16.0)).unwrap();
        assert_eq!(resampled.time_axis().as_uniform(), Some((0.0, 1.0 / 16.0)));
    }
}
