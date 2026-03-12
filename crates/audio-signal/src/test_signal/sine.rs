use std::f64::consts::TAU;

use ndarray::Array2;

use crate::signal::{SignalError, TimeSignal};

#[derive(Debug, Clone, Copy)]
pub struct SineConfig {
    pub amplitude: f64,
    pub sample_rate: f64,
    pub num_channels: usize,
}

impl Default for SineConfig {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            sample_rate: 48_000.0,
            num_channels: 1,
        }
    }
}

pub fn generate_sine(
    num_time_steps: usize,
    frequency: f64,
    config: &SineConfig,
) -> Result<TimeSignal, SignalError> {
    let SineConfig {
        amplitude,
        sample_rate,
        num_channels,
    } = *config;

    let data = Array2::from_shape_fn((num_channels, num_time_steps), |(_, frame)| {
        let t = frame as f64 / sample_rate;
        amplitude * (TAU * frequency * t).sin()
    });

    let mut signal = TimeSignal::new(data, sample_rate)?;
    let comment = format!("Sine signal (frequency = {frequency} Hz, amplitude = {amplitude})");
    signal.set_comment(Some(&comment));
    Ok(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_sine() {
        let signal = generate_sine(
            8,
            1.0,
            &SineConfig {
                amplitude: 0.5,
                sample_rate: 8.0,
                num_channels: 2,
            },
        )
        .unwrap();
        assert_eq!(signal.num_channels(), 2);
        assert_eq!(signal.num_time_steps(), 8);
        assert_eq!(signal.channel(0)[0], 0.0);
        assert!((signal.channel(0)[2] - 0.5).abs() < 1e-12);
        assert_eq!(signal.channel(0), signal.channel(1));
    }
}
