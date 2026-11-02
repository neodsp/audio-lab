use ndarray::{Array1, Array2};
use ndarray_rand::{
    RandomExt,
    rand::{SeedableRng, rngs::StdRng},
    rand_distr::Uniform,
};

use crate::signal::{SignalError, TimeSignal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spectrum {
    White,
    Pink,
}

#[derive(Debug, thiserror::Error)]
pub enum NoiseError {
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseConfig {
    pub spectrum: Spectrum,
    pub amplitude: f64,
    pub sample_rate: f64,
    pub num_channels: usize,
    /// Seed for the random number generator. `None` uses a random seed.
    pub seed: Option<u64>,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            spectrum: Spectrum::White,
            amplitude: 1.0,
            sample_rate: 48_000.0,
            num_channels: 1,
            seed: None,
        }
    }
}

pub fn generate_noise(
    num_time_steps: usize,
    config: NoiseConfig,
) -> Result<TimeSignal, NoiseError> {
    let NoiseConfig {
        spectrum,
        amplitude,
        sample_rate,
        num_channels,
        seed,
    } = config;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };
    let data = Array2::random_using(
        (num_channels, num_time_steps),
        Uniform::new(-1.0, 1.0).unwrap(),
        &mut rng,
    );
    let mut signal = TimeSignal::new(data, sample_rate)?;

    if spectrum == Spectrum::Pink {
        let mut freq = signal.into_freq();
        let filter = Array1::from_shape_fn(freq.num_freq_bins(), |i| ((i + 1) as f64).sqrt());
        for mut channel in freq.channel_iter_mut() {
            channel.zip_mut_with(&filter, |bin, scale| *bin /= *scale);
        }
        signal = freq.into_time();
    }

    signal.normalize_peak(amplitude);

    let comment = format!("{spectrum:?} noise signal (amplitude = {amplitude})");
    signal.set_comment(Some(&comment));
    Ok(signal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_reproducible_white_noise() {
        let config = NoiseConfig {
            seed: Some(123),
            amplitude: 0.5,
            ..Default::default()
        };
        let left = generate_noise(32, config).unwrap();
        let right = generate_noise(32, config).unwrap();
        assert_eq!(left, right);
        let peak = left
            .iter()
            .fold(0.0_f64, |peak, sample| peak.max(sample.abs()));
        assert!((peak - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pink_noise_is_supported() {
        let noise = generate_noise(
            64,
            NoiseConfig {
                spectrum: Spectrum::Pink,
                seed: Some(123),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(noise.num_time_steps(), 64);
    }
}
