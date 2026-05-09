use ndarray::Array2;

use crate::signal::{SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum ImpulseError {
    #[error("delay ({delay}) must be less than num_time_steps ({num_time_steps})")]
    DelayOutOfRange { delay: usize, num_time_steps: usize },
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone, Copy)]
pub struct ImpulseConfig {
    pub delay: usize,
    pub amplitude: f64,
    pub sample_rate: f64,
    pub num_channels: usize,
}

impl Default for ImpulseConfig {
    fn default() -> Self {
        Self {
            delay: 0,
            amplitude: 1.0,
            sample_rate: 48_000.0,
            num_channels: 1,
        }
    }
}

pub fn generate_impulse(
    num_time_steps: usize,
    config: ImpulseConfig,
) -> Result<TimeSignal, ImpulseError> {
    let ImpulseConfig {
        delay,
        amplitude,
        sample_rate,
        num_channels,
    } = config;

    if delay >= num_time_steps {
        return Err(ImpulseError::DelayOutOfRange {
            delay,
            num_time_steps,
        });
    }

    let mut data = Array2::zeros((num_channels, num_time_steps));
    for ch in 0..num_channels {
        data[[ch, delay]] = amplitude;
    }

    let mut signal = TimeSignal::new(data, sample_rate)?;
    let comment = format!("Impulse signal (delay = {delay} samples, amplitude = {amplitude})");
    signal.set_comment(Some(&comment));
    Ok(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_impulse_at_zero() {
        let signal = generate_impulse(8, ImpulseConfig::default()).unwrap();
        assert_eq!(signal.channel(0)[0], 1.0);
        assert_eq!(signal.channel(0)[1], 0.0);
    }

    #[test]
    fn generates_impulse_with_delay() {
        let signal = generate_impulse(
            8,
            ImpulseConfig {
                delay: 3,
                amplitude: 0.5,
                num_channels: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(signal.channel(0)[3], 0.5);
        assert_eq!(signal.channel(1)[3], 0.5);
        assert_eq!(signal.channel(0)[0], 0.0);
    }

    #[test]
    fn validates_delay() {
        assert!(matches!(
            generate_impulse(
                8,
                ImpulseConfig {
                    delay: 8,
                    ..Default::default()
                }
            ),
            Err(ImpulseError::DelayOutOfRange { .. })
        ));
    }
}
