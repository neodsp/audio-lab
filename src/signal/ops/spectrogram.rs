use ndarray::{Array1, Array3, Zip};
use num::complex::Complex64;

use crate::signal::Spectrogram;

fn one_sided_scale(spec: &Spectrogram, bin: usize) -> f64 {
    let last_bin = spec.num_freq_bins().saturating_sub(1);
    if bin == 0 || (spec.window_size().is_multiple_of(2) && bin == last_bin) {
        1.0
    } else {
        2.0
    }
}

/// Apply `value` to every bin and multiply by the per-bin one-sided scale
/// divided by `denom`. The scales are computed once per bin, not per element.
fn scaled_per_bin(
    spec: &Spectrogram,
    denom: f64,
    value: impl Fn(&Complex64) -> f64,
) -> Array3<f64> {
    let scales = Array1::from_shape_fn(spec.num_freq_bins(), |bin| {
        one_sided_scale(spec, bin) / denom
    });
    Zip::from(spec.data())
        .and_broadcast(&scales)
        .map_collect(|c, &scale| value(c) * scale)
}

fn to_db_in_place(mut values: Array3<f64>, factor: f64, floor_linear: f64) -> Array3<f64> {
    values.mapv_inplace(|v| factor * v.max(floor_linear).log10());
    values
}

impl Spectrogram {
    pub fn raw_magnitude(&self) -> Array3<f64> {
        self.data().map(|c| c.norm())
    }

    pub fn raw_magnitude_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 20.0);
        self.data()
            .map(|c| 20.0 * c.norm().max(floor_linear).log10())
    }

    pub fn amplitude_spectrum(&self) -> Array3<f64> {
        let coherent_sum = self.normalization().coherent_gain() * self.window_size() as f64;
        scaled_per_bin(self, coherent_sum, |c| c.norm())
    }

    pub fn amplitude_spectrum_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 20.0);
        to_db_in_place(self.amplitude_spectrum(), 20.0, floor_linear)
    }

    pub fn power_spectrum(&self) -> Array3<f64> {
        let denom = self.window_size() as f64 * self.normalization().window_energy();
        scaled_per_bin(self, denom, |c| c.norm_sqr())
    }

    pub fn power_spectrum_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 10.0);
        to_db_in_place(self.power_spectrum(), 10.0, floor_linear)
    }

    pub fn power_spectral_density(&self) -> Array3<f64> {
        let denom = self.sample_rate() * self.normalization().window_energy();
        scaled_per_bin(self, denom, |c| c.norm_sqr())
    }

    pub fn power_spectral_density_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 10.0);
        to_db_in_place(self.power_spectral_density(), 10.0, floor_linear)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;
    use num::complex::Complex64;

    use crate::dsp::{WindowFn, generate_window};
    use crate::signal::Spectrogram;

    fn make_spectrogram(window_size: usize) -> Spectrogram {
        let num_freq_bins = window_size / 2 + 1;
        let data = Array3::from_shape_fn((2, 3, num_freq_bins), |(ch, frame, bin)| {
            Complex64::new(
                (ch + 1) as f64 * (bin as f64 + 0.5),
                frame as f64 - bin as f64 * 0.25,
            )
        });
        Spectrogram::new(data, 8_000.0, window_size, window_size / 2, WindowFn::Hann).unwrap()
    }

    #[test]
    fn scaled_spectra_match_per_element_formulas() {
        // Even window: DC and Nyquist bins are not doubled. Odd window: only DC.
        for window_size in [8, 9] {
            let spec = make_spectrogram(window_size);
            let last_bin = spec.num_freq_bins() - 1;
            let scale = |bin: usize| {
                if bin == 0 || (window_size % 2 == 0 && bin == last_bin) {
                    1.0
                } else {
                    2.0
                }
            };
            // Gains of the Hann window the spectrogram was built with.
            let window = generate_window(WindowFn::Hann, window_size);
            let window_energy: f64 = window.iter().map(|w| w * w).sum();
            let coherent_sum = window.sum();
            let power_denom = window_size as f64 * window_energy;
            let psd_denom = 8_000.0 * window_energy;

            let amplitude = spec.amplitude_spectrum();
            let amplitude_db = spec.amplitude_spectrum_db(-200.0);
            let power = spec.power_spectrum();
            let power_db = spec.power_spectrum_db(-200.0);
            let psd = spec.power_spectral_density();
            let psd_db = spec.power_spectral_density_db(-200.0);

            for ((ch, frame, bin), c) in spec.data().indexed_iter() {
                let idx = [ch, frame, bin];
                let expected_amplitude = c.norm() * scale(bin) / coherent_sum;
                let expected_power = c.norm_sqr() * scale(bin) / power_denom;
                let expected_psd = c.norm_sqr() * scale(bin) / psd_denom;

                assert_abs_diff_eq!(amplitude[idx], expected_amplitude, epsilon = 1e-12);
                assert_abs_diff_eq!(power[idx], expected_power, epsilon = 1e-12);
                assert_abs_diff_eq!(psd[idx], expected_psd, epsilon = 1e-12);
                assert_abs_diff_eq!(
                    amplitude_db[idx],
                    20.0 * expected_amplitude.log10(),
                    epsilon = 1e-9
                );
                assert_abs_diff_eq!(power_db[idx], 10.0 * expected_power.log10(), epsilon = 1e-9);
                assert_abs_diff_eq!(psd_db[idx], 10.0 * expected_psd.log10(), epsilon = 1e-9);
            }
        }
    }
}
