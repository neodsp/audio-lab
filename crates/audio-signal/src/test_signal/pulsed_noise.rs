use std::f64::consts::PI;

use ndarray::{Array1, Array2, s};

use crate::signal::TimeSignal;
use crate::test_signal::noise::{NoiseError, Spectrum, generate_noise};

#[derive(Debug, thiserror::Error)]
pub enum PulsedNoiseError {
    #[error("fade is too large; it must be at most half the pulse length")]
    FadeTooLarge,
    #[error(transparent)]
    Noise(#[from] NoiseError),
}

pub fn generate_pulsed_noise(
    pulse_length: usize,
    pause_length: usize,
    fade_length: usize,
    repetitions: usize,
    spectrum: Spectrum,
    amplitude: f64,
    frozen: bool,
    sample_rate: f64,
    num_channels: usize,
    seed: u64,
) -> Result<TimeSignal, PulsedNoiseError> {
    if fade_length * 2 > pulse_length {
        return Err(PulsedNoiseError::FadeTooLarge);
    }

    let noise_samples = if frozen {
        pulse_length
    } else {
        pulse_length * repetitions
    };
    let source = generate_noise(
        noise_samples,
        spectrum,
        amplitude,
        sample_rate,
        num_channels,
        seed,
    )?;

    let mut data = Array2::zeros((
        num_channels,
        repetitions
            .saturating_mul(pulse_length + pause_length)
            .saturating_sub(pause_length),
    ));
    let fade = Array1::linspace(0.0, PI / 2.0, fade_length).mapv(|v| v.sin().powi(2));

    for ch in 0..num_channels {
        for repetition in 0..repetitions {
            let pulse_start = repetition * pulse_length;
            let signal_start = repetition * (pulse_length + pause_length);

            if frozen {
                let source_channel = source.channel(ch);
                data.slice_mut(s![ch, signal_start..signal_start + pulse_length])
                    .assign(&source_channel);
            } else {
                let source_channel = source.channel(ch);
                let source_slice =
                    source_channel.slice(s![pulse_start..pulse_start + pulse_length]);
                data.slice_mut(s![ch, signal_start..signal_start + pulse_length])
                    .assign(&source_slice);
            }

            if fade_length > 0 {
                {
                    let mut attack =
                        data.slice_mut(s![ch, signal_start..signal_start + fade_length]);
                    attack.zip_mut_with(&fade, |sample, gain| *sample *= *gain);
                }
                {
                    let mut release = data.slice_mut(s![
                        ch,
                        signal_start + pulse_length - fade_length..signal_start + pulse_length
                    ]);
                    release.zip_mut_with(&fade.slice(s![..;-1]), |sample, gain| *sample *= *gain);
                }
            }
        }
    }

    let mut signal = TimeSignal::new(data, sample_rate).expect("validated sample rate");
    let frozen_str = if frozen { "frozen " } else { "" };
    let comment = format!(
        "{frozen_str}{spectrum:?} pulsed noise signal (amplitude = {amplitude}, {repetitions} repetitions, {pulse_length} samples pulse duration, {pause_length} samples pauses, and {fade_length} samples fades)"
    );
    signal.set_comment(Some(&comment));
    Ok(signal)
}

#[cfg(test)]
mod tests {
    use crate::test_signal::noise::Spectrum;

    use super::*;

    #[test]
    fn validates_fade_length() {
        assert!(matches!(
            generate_pulsed_noise(8, 2, 5, 2, Spectrum::White, 0.5, true, 48_000.0, 1, 7),
            Err(PulsedNoiseError::FadeTooLarge)
        ));
    }

    #[test]
    fn generates_frozen_pulsed_noise() {
        let signal =
            generate_pulsed_noise(4, 2, 1, 3, Spectrum::White, 0.5, true, 48_000.0, 1, 7).unwrap();
        assert_eq!(signal.num_time_steps(), 16);
        assert_eq!(signal.channel(0)[4], 0.0);
        assert_eq!(signal.channel(0)[5], 0.0);
        assert_eq!(
            signal.channel(0).slice(s![0..4]),
            signal.channel(0).slice(s![6..10])
        );
    }

    #[test]
    fn generates_non_frozen_pulsed_noise() {
        let signal =
            generate_pulsed_noise(4, 1, 0, 2, Spectrum::White, 0.5, false, 48_000.0, 1, 7).unwrap();
        assert_eq!(signal.num_time_steps(), 9);
        assert_ne!(
            signal.channel(0).slice(s![0..4]),
            signal.channel(0).slice(s![5..9])
        );
    }
}
