use ndarray::Array3;

use crate::signal::Spectrogram;

fn one_sided_scale(spec: &Spectrogram, bin: usize) -> f64 {
    let last_bin = spec.num_freq_bins().saturating_sub(1);
    if bin == 0 || (spec.window_size().is_multiple_of(2) && bin == last_bin) {
        1.0
    } else {
        2.0
    }
}

pub fn raw_magnitude(spec: &Spectrogram) -> Array3<f64> {
    spec.data().map(|c| c.norm())
}

pub fn raw_magnitude_db(spec: &Spectrogram, floor_db: f64) -> Array3<f64> {
    let floor_linear = 10f64.powf(floor_db / 20.0);
    spec.data()
        .map(|c| 20.0 * c.norm().max(floor_linear).log10())
}

pub fn amplitude_spectrum(spec: &Spectrogram) -> Array3<f64> {
    let coherent_sum = spec.normalization().coherent_gain() * spec.window_size() as f64;
    Array3::from_shape_fn(spec.data().raw_dim(), |(ch, frame, bin)| {
        spec.data()[[ch, frame, bin]].norm() * one_sided_scale(spec, bin) / coherent_sum
    })
}

pub fn amplitude_spectrum_db(spec: &Spectrogram, floor_db: f64) -> Array3<f64> {
    let floor_linear = 10f64.powf(floor_db / 20.0);
    amplitude_spectrum(spec).map(|v| 20.0 * v.max(floor_linear).log10())
}

pub fn power_spectrum(spec: &Spectrogram) -> Array3<f64> {
    let denom = spec.window_size() as f64 * spec.normalization().window_energy();
    Array3::from_shape_fn(spec.data().raw_dim(), |(ch, frame, bin)| {
        let mag2 = spec.data()[[ch, frame, bin]].norm_sqr();
        mag2 * one_sided_scale(spec, bin) / denom
    })
}

pub fn power_spectrum_db(spec: &Spectrogram, floor_db: f64) -> Array3<f64> {
    let floor_linear = 10f64.powf(floor_db / 10.0);
    power_spectrum(spec).map(|v| 10.0 * v.max(floor_linear).log10())
}

pub fn power_spectral_density(spec: &Spectrogram) -> Array3<f64> {
    let denom = spec.sample_rate() * spec.normalization().window_energy();
    Array3::from_shape_fn(spec.data().raw_dim(), |(ch, frame, bin)| {
        let mag2 = spec.data()[[ch, frame, bin]].norm_sqr();
        mag2 * one_sided_scale(spec, bin) / denom
    })
}

pub fn power_spectral_density_db(spec: &Spectrogram, floor_db: f64) -> Array3<f64> {
    let floor_linear = 10f64.powf(floor_db / 10.0);
    power_spectral_density(spec).map(|v| 10.0 * v.max(floor_linear).log10())
}
