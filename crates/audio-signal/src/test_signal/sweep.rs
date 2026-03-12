use std::{f64::consts::PI, ops::Range};

use ndarray::Array1;

use crate::signal::{SignalError, TimeSignal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepType {
    Linear,
    Exponential,
}

#[derive(Debug, thiserror::Error)]
pub enum SweepError {
    #[error("Frequency range exceeds Nyquist frequency")]
    Nyquist,
    #[error("Start of frequency range must be smaller than end")]
    FreqRange,
    #[error("Fade out is longer than sweep")]
    FadeOut,
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone, Copy)]
pub struct SweepConfig {
    pub amplitude: f64,
    pub sample_rate: f64,
    pub num_channels: usize,
    pub fade_out: usize,
    pub sweep_type: SweepType,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            sample_rate: 48_000.0,
            num_channels: 1,
            fade_out: 90,
            sweep_type: SweepType::Exponential,
        }
    }
}

pub fn generate_sweep(
    num_time_steps: usize,
    freq_range: Range<f64>,
    config: &SweepConfig,
) -> Result<TimeSignal, SweepError> {
    let SweepConfig {
        amplitude,
        sample_rate,
        num_channels,
        fade_out,
        sweep_type,
    } = *config;

    if freq_range.end <= freq_range.start {
        return Err(SweepError::FreqRange);
    }
    if freq_range.end >= sample_rate / 2.0 {
        return Err(SweepError::Nyquist);
    }
    if fade_out > num_time_steps {
        return Err(SweepError::FadeOut);
    }

    let mut signal = match sweep_type {
        SweepType::Linear => linear_sweep(
            num_time_steps,
            freq_range.clone(),
            amplitude,
            sample_rate,
            num_channels,
        )?,
        SweepType::Exponential => exponential_sweep(
            num_time_steps,
            freq_range.clone(),
            amplitude,
            sample_rate,
            num_channels,
        )?,
    };

    if fade_out > 0 {
        let fade = Array1::linspace(0.0, PI / 2.0, fade_out).mapv(|v| v.cos().powi(2));
        for mut channel in signal.channel_iter_mut() {
            for (idx, gain) in fade.iter().enumerate() {
                let frame = num_time_steps - fade_out + idx;
                channel[frame] *= *gain;
            }
        }
    }

    let comment = format!(
        "{sweep_type:?} sweep ({:.1} Hz..{:.1} Hz, amplitude = {amplitude})",
        freq_range.start, freq_range.end
    );
    signal.set_comment(Some(&comment));
    Ok(signal)
}

fn linear_sweep(
    num_time_steps: usize,
    freq_range: Range<f64>,
    amplitude: f64,
    sample_rate: f64,
    num_channels: usize,
) -> Result<TimeSignal, SignalError> {
    let duration = num_time_steps as f64 / sample_rate;
    let mut signal = TimeSignal::zeros(num_channels, num_time_steps, sample_rate)?;
    let times = signal.time_steps().to_owned();

    for mut channel in signal.channel_iter_mut() {
        for (sample, time) in channel.iter_mut().zip(times.iter()) {
            *sample = amplitude
                * (2.0 * PI * freq_range.start * *time
                    + PI * (freq_range.end - freq_range.start) * time.powi(2) / duration)
                    .sin();
        }
    }

    Ok(signal)
}

fn exponential_sweep(
    num_time_steps: usize,
    freq_range: Range<f64>,
    amplitude: f64,
    sample_rate: f64,
    num_channels: usize,
) -> Result<TimeSignal, SignalError> {
    let c = (freq_range.end / freq_range.start).ln();
    let l = num_time_steps as f64 / sample_rate / c;
    let mut signal = TimeSignal::zeros(num_channels, num_time_steps, sample_rate)?;
    let times = signal.time_steps().to_owned();

    for mut channel in signal.channel_iter_mut() {
        for (sample, time) in channel.iter_mut().zip(times.iter()) {
            *sample =
                amplitude * (2.0 * PI * freq_range.start * l * ((*time / l).exp() - 1.0)).sin();
        }
    }

    Ok(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_sweep_range() {
        assert!(matches!(
            generate_sweep(128, 1_000.0..500.0, &SweepConfig::default()),
            Err(SweepError::FreqRange)
        ));
    }

    #[test]
    fn generates_linear_sweep() {
        let signal = generate_sweep(
            128,
            20.0..20_000.0,
            &SweepConfig {
                amplitude: 0.5,
                num_channels: 2,
                fade_out: 8,
                sweep_type: SweepType::Linear,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(signal.num_channels(), 2);
        assert_eq!(signal.num_time_steps(), 128);
    }
}
