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

pub fn generate_noise(
    num_time_steps: usize,
    spectrum: Spectrum,
    amplitude: f64,
    sample_rate: f64,
    num_channels: usize,
    seed: u64,
) -> Result<TimeSignal, NoiseError> {
    let mut rng = StdRng::seed_from_u64(seed);
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

    normalize_peak(&mut signal, amplitude);

    let comment = format!("{spectrum:?} noise signal (amplitude = {amplitude})");
    signal.set_comment(Some(&comment));
    Ok(signal)
}

pub(crate) fn normalize_peak(signal: &mut TimeSignal, amplitude: f64) {
    let peak = signal
        .iter()
        .fold(0.0_f64, |peak, sample| peak.max(sample.abs()));
    if peak > 0.0 {
        let gain = amplitude / peak;
        signal.iter_mut().for_each(|sample| *sample *= gain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_reproducible_white_noise() {
        let left = generate_noise(32, Spectrum::White, 0.5, 48_000.0, 1, 123).unwrap();
        let right = generate_noise(32, Spectrum::White, 0.5, 48_000.0, 1, 123).unwrap();
        assert_eq!(left, right);
        let peak = left
            .iter()
            .fold(0.0_f64, |peak, sample| peak.max(sample.abs()));
        assert!((peak - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pink_noise_is_supported() {
        let noise = generate_noise(64, Spectrum::Pink, 0.5, 48_000.0, 1, 123).unwrap();
        assert_eq!(noise.num_time_steps(), 64);
    }
}
