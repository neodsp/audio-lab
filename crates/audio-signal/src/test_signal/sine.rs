use std::f64::consts::TAU;

use ndarray::Array2;

use crate::signal::{SignalError, TimeSignal};

pub fn generate_sine(
    num_time_steps: usize,
    frequency: f64,
    amplitude: f64,
    sample_rate: f64,
    num_channels: usize,
) -> Result<TimeSignal, SignalError> {
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
        let signal = generate_sine(8, 1.0, 0.5, 8.0, 2).unwrap();
        assert_eq!(signal.num_channels(), 2);
        assert_eq!(signal.num_time_steps(), 8);
        assert_eq!(signal.channel(0)[0], 0.0);
        assert!((signal.channel(0)[2] - 0.5).abs() < 1e-12);
        assert_eq!(signal.channel(0), signal.channel(1));
    }
}
