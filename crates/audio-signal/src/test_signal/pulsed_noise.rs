use std::f64::consts::PI;

use ndarray::{Array1, Array2, s};

use crate::signal::TimeSignal;
use crate::test_signal::noise::{NoiseConfig, NoiseError, Spectrum, generate_noise};

#[derive(Debug, thiserror::Error)]
pub enum PulsedNoiseError {
    #[error("fade is too large; it must be at most half the pulse length")]
    FadeTooLarge,
    #[error(transparent)]
    Noise(#[from] NoiseError),
}

#[derive(Debug, Clone, Copy)]
pub struct PulsedNoiseConfig {
    pub fade_length: usize,
    pub spectrum: Spectrum,
    pub amplitude: f64,
    pub frozen: bool,
    pub sample_rate: f64,
    pub num_channels: usize,
    /// Seed for the random number generator. `None` uses a random seed.
    pub seed: Option<u64>,
}

impl Default for PulsedNoiseConfig {
    fn default() -> Self {
        Self {
            fade_length: 0,
            spectrum: Spectrum::White,
            amplitude: 1.0,
            frozen: false,
            sample_rate: 48_000.0,
            num_channels: 1,
            seed: None,
        }
    }
}

pub fn generate_pulsed_noise(
    pulse_length: usize,
    pause_length: usize,
    repetitions: usize,
    config: &PulsedNoiseConfig,
) -> Result<TimeSignal, PulsedNoiseError> {
    let PulsedNoiseConfig {
        fade_length,
        spectrum,
        amplitude,
        frozen,
        sample_rate,
        num_channels,
        seed,
    } = *config;

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
        &NoiseConfig {
            spectrum,
            amplitude,
            sample_rate,
            num_channels,
            seed,
        },
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
            generate_pulsed_noise(
                8,
                2,
                2,
                &PulsedNoiseConfig {
                    fade_length: 5,
                    spectrum: Spectrum::White,
                    frozen: true,
                    seed: Some(7),
                    ..Default::default()
                }
            ),
            Err(PulsedNoiseError::FadeTooLarge)
        ));
    }

    #[test]
    fn generates_frozen_pulsed_noise() {
        let signal = generate_pulsed_noise(
            4,
            2,
            3,
            &PulsedNoiseConfig {
                fade_length: 1,
                frozen: true,
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
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
        let signal = generate_pulsed_noise(
            4,
            1,
            2,
            &PulsedNoiseConfig {
                seed: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(signal.num_time_steps(), 9);
        assert_ne!(
            signal.channel(0).slice(s![0..4]),
            signal.channel(0).slice(s![5..9])
        );
    }
}
