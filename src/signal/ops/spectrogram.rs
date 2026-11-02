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
        Array3::from_shape_fn(self.data().raw_dim(), |(ch, frame, bin)| {
            self.data()[[ch, frame, bin]].norm() * one_sided_scale(self, bin) / coherent_sum
        })
    }

    pub fn amplitude_spectrum_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 20.0);
        self.amplitude_spectrum()
            .map(|v| 20.0 * v.max(floor_linear).log10())
    }

    pub fn power_spectrum(&self) -> Array3<f64> {
        let denom = self.window_size() as f64 * self.normalization().window_energy();
        Array3::from_shape_fn(self.data().raw_dim(), |(ch, frame, bin)| {
            let mag2 = self.data()[[ch, frame, bin]].norm_sqr();
            mag2 * one_sided_scale(self, bin) / denom
        })
    }

    pub fn power_spectrum_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 10.0);
        self.power_spectrum()
            .map(|v| 10.0 * v.max(floor_linear).log10())
    }

    pub fn power_spectral_density(&self) -> Array3<f64> {
        let denom = self.sample_rate() * self.normalization().window_energy();
        Array3::from_shape_fn(self.data().raw_dim(), |(ch, frame, bin)| {
            let mag2 = self.data()[[ch, frame, bin]].norm_sqr();
            mag2 * one_sided_scale(self, bin) / denom
        })
    }

    pub fn power_spectral_density_db(&self, floor_db: f64) -> Array3<f64> {
        let floor_linear = 10f64.powf(floor_db / 10.0);
        self.power_spectral_density()
            .map(|v| 10.0 * v.max(floor_linear).log10())
    }
}
