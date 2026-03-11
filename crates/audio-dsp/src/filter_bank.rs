use ndarray::prelude::*;
use ndrustfft::{R2cFftHandler, ndfft_r2c, ndifft_r2c};
use num::complex::Complex64;

use audio_signal::signal::{SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    #[error("overlap must be between 0 and 1, got {0}")]
    InvalidOverlap(f64),
    #[error("n_samples must be at least 2")]
    InvalidNSamples,
    #[error("no valid frequency bands found in range {0}..{1} Hz")]
    NoBandsInRange(f64, f64),
    #[error(transparent)]
    Signal(#[from] SignalError),
}

/// Compute fractional octave center and cut-off frequencies (IEC 61260).
///
/// Returns `(centers, lower_cutoffs, upper_cutoffs)` for all bands whose
/// center frequency falls within `frequency_range`.
pub fn fractional_octave_frequencies(
    num_fractions: usize,
    frequency_range: (f64, f64),
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (f_min, f_max) = frequency_range;
    let nf = num_fractions as f64;

    // Generous k range – filtered to frequency_range below
    let k_min = (nf * (f_min / 1000.0).log2()).floor() as i32 - 1;
    let k_max = (nf * (f_max / 1000.0).log2()).ceil() as i32 + 1;

    let mut centers = Vec::new();
    let mut lowers = Vec::new();
    let mut uppers = Vec::new();

    for k in k_min..=k_max {
        let f_m = 1000.0 * 2.0_f64.powf(k as f64 / nf);
        if f_m >= f_min * (1.0 - 1e-6) && f_m <= f_max * (1.0 + 1e-6) {
            centers.push(f_m);
            lowers.push(f_m * 2.0_f64.powf(-0.5 / nf));
            uppers.push(f_m * 2.0_f64.powf(0.5 / nf));
        }
    }

    (centers, lowers, uppers)
}

/// A reconstructing linear-phase fractional octave filter bank.
///
/// # Quick start
///
/// ```rust,ignore
/// // Design a third-octave bank and apply it to a signal
/// let fb = OctaveBands::design(3, (63.0, 16000.0), 1.0, 0, 4096, signal.sample_rate())?;
/// let bands: Vec<TimeSignal> = fb.apply(&signal)?;
/// let freqs: &[f64] = &fb.center_frequencies;
/// ```
///
/// The sum of all `bands` approximately equals the input delayed by
/// `n_samples / 2` samples. The final Hann window improves FIR sidelobes but
/// makes reconstruction approximate rather than exact.
pub struct OctaveBands {
    /// One FIR impulse response per band: shape `[num_bands, n_samples]`.
    pub impulse_responses: TimeSignal,
    /// Center frequency of each band in Hz.
    pub center_frequencies: Vec<f64>,
}

impl OctaveBands {
    /// Design a reconstructing fractional octave filter bank.
    ///
    /// Filters are windowed FIR filters designed in the frequency domain
    /// following Antoni (2010), doi: 10.1121/1.3273888. The first band acts as
    /// a low-pass below the lowest centre frequency; the last as a high-pass
    /// above the highest.
    ///
    /// # Parameters
    /// - `num_fractions` – octave fraction (1 = octave, 3 = third-octave, …).
    /// - `frequency_range` – Hz range `(low, high)` for the band centres.
    /// - `overlap` – transition overlap in `[0, 1]`; smaller values give wider
    ///   pass-bands and steeper slopes (typical default: `1.0`).
    /// - `slope` – steepness iterations ≥ 0; each step applies one sine
    ///   recursion for a steeper roll-off (typical default: `0`).
    /// - `n_samples` – FIR filter length in samples; longer → more accurate
    ///   (typical default: `4096`).
    /// - `sample_rate` – sampling rate in Hz.
    pub fn design(
        num_fractions: usize,
        frequency_range: (f64, f64),
        overlap: f64,
        slope: usize,
        n_samples: usize,
        sample_rate: f64,
    ) -> Result<Self, FilterError> {
        if !(0.0..=1.0).contains(&overlap) {
            return Err(FilterError::InvalidOverlap(overlap));
        }
        if n_samples < 2 {
            return Err(FilterError::InvalidNSamples);
        }

        let n_bins = n_samples / 2 + 1;
        let (f_center, f_lower, f_upper) =
            fractional_octave_frequencies(num_fractions, frequency_range);

        // Drop bands above Nyquist
        let nyquist = sample_rate / 2.0;
        let mut f_m = Vec::new();
        let mut f_l = Vec::new();
        let mut f_u = Vec::new();
        for ((fc, fl), fu) in f_center.iter().zip(&f_lower).zip(&f_upper) {
            if *fc < nyquist {
                f_m.push(*fc);
                f_l.push(*fl);
                f_u.push(*fu);
            }
        }

        let num_bands = f_m.len();
        if num_bands == 0 {
            return Err(FilterError::NoBandsInRange(
                frequency_range.0,
                frequency_range.1,
            ));
        }

        // Frequency → nearest FFT bin
        let freq_to_bin = |f: f64| -> usize {
            ((n_samples as f64 * f / sample_rate).round() as usize).min(n_bins - 1)
        };

        let k_1: Vec<usize> = f_l.iter().map(|&f| freq_to_bin(f)).collect(); // lower cut-off
        let k_m: Vec<usize> = f_m.iter().map(|&f| freq_to_bin(f)).collect(); // centre
        let k_2: Vec<usize> = f_u.iter().map(|&f| freq_to_bin(f)).collect(); // upper cut-off

        // Half-width of each transition region (bins).
        // Based on the upper half-bandwidth of band b (centre → upper cut-off).
        let p: Vec<usize> = (0..num_bands)
            .map(|b| ((overlap / 2.0) * (k_2[b] as f64 - k_m[b] as f64)).round() as usize)
            .collect();

        // Magnitude responses: start all-pass (ones)
        let mut g = Array2::<f64>::ones((num_bands, n_bins));

        // Build transitions between adjacent bands (start at b=1: first band is low-pass)
        for b in 1..num_bands {
            if p[b] > 0 {
                let pb = p[b] as isize;
                let k1b = k_1[b] as isize;

                for pp in -pb..=pb {
                    let bin = k1b + pp;
                    if bin < 0 || bin as usize >= n_bins {
                        continue;
                    }
                    let idx = bin as usize;

                    // φ ∈ [-1, 1], optionally refined by sine recursion
                    let mut phi = pp as f64 / pb as f64;
                    for _ in 0..slope {
                        phi = (std::f64::consts::PI / 2.0 * phi).sin();
                    }
                    phi = 0.5 * (phi + 1.0); // shift to [0, 1]

                    // Fade out band b-1, fade in band b
                    g[[b - 1, idx]] = (std::f64::consts::PI / 2.0 * phi).cos();
                    g[[b, idx]] = (std::f64::consts::PI / 2.0 * phi).sin();
                }
            }

            // Zero outside each band's support
            let high = (k_1[b] + p[b]).min(n_bins - 1) + 1;
            g.slice_mut(s![b - 1, high..]).fill(0.0);

            let low = k_1[b].saturating_sub(p[b]);
            g.slice_mut(s![b, ..low]).fill(0.0);
        }

        // Square magnitudes → -6 dB at cut-off frequencies
        g.mapv_inplace(|v| v * v);

        // Multiply by linear phase term: exp(-j·2π·f·T_d), T_d = n_samples/2 / sr
        let group_delay = n_samples as f64 / 2.0 / sample_rate;
        let freq_step = sample_rate / n_samples as f64;

        let mut g_complex = Array2::<Complex64>::zeros((num_bands, n_bins));
        for b in 0..num_bands {
            for k in 0..n_bins {
                let freq = k as f64 * freq_step;
                let phase = -2.0 * std::f64::consts::PI * freq * group_delay;
                g_complex[[b, k]] =
                    Complex64::new(g[[b, k]] * phase.cos(), g[[b, k]] * phase.sin());
            }
        }

        // IFFT → impulse responses
        let fft_handler = R2cFftHandler::<f64>::new(n_samples);
        let mut ir = Array2::<f64>::zeros((num_bands, n_samples));
        ndifft_r2c(&g_complex.view(), &mut ir.view_mut(), &fft_handler, 1);

        // Hanning window (suppresses side lobes of the finite-length FIR)
        let window = hanning_window(n_samples);
        for b in 0..num_bands {
            ir.slice_mut(s![b, ..])
                .zip_mut_with(&window, |x, &w| *x *= w);
        }

        let impulse_responses = TimeSignal::new(ir, sample_rate)?;
        Ok(Self {
            impulse_responses,
            center_frequencies: f_m,
        })
    }

    /// Apply the filter bank to `signal` via FFT-based linear convolution.
    ///
    /// Returns one [`TimeSignal`] per band (same channel count as the input).
    /// Each output is `signal.num_time_steps() + n_samples - 1` samples long
    /// and is delayed by `n_samples / 2` samples relative to the input.
    pub fn apply(&self, signal: &TimeSignal) -> Result<Vec<TimeSignal>, FilterError> {
        let n_sig = signal.num_time_steps();
        let n_filt = self.impulse_responses.num_time_steps();
        let n_out = n_sig + n_filt - 1;
        let fft_size = n_out.next_power_of_two();
        let n_freq = fft_size / 2 + 1;
        let num_bands = self.impulse_responses.num_channels();
        let num_ch = signal.num_channels();

        let fft_handler = R2cFftHandler::<f64>::new(fft_size);

        // Pre-compute spectra of all filter impulse responses in one batch FFT
        let mut filter_spectra = Array2::<Complex64>::zeros((num_bands, n_freq));
        {
            let mut padded = Array2::<f64>::zeros((num_bands, fft_size));
            padded
                .slice_mut(s![.., ..n_filt])
                .assign(&self.impulse_responses.time_data());
            ndfft_r2c(
                &padded.view(),
                &mut filter_spectra.view_mut(),
                &fft_handler,
                1,
            );
        }

        // Pre-compute spectra of all input channels in one batch FFT
        let mut sig_spectra = Array2::<Complex64>::zeros((num_ch, n_freq));
        {
            let mut padded = Array2::<f64>::zeros((num_ch, fft_size));
            padded
                .slice_mut(s![.., ..n_sig])
                .assign(&signal.time_data());
            ndfft_r2c(&padded.view(), &mut sig_spectra.view_mut(), &fft_handler, 1);
        }

        // Convolve each band with every channel
        let mut product = Array2::<Complex64>::zeros((num_ch, n_freq));
        let mut ifft_buf = Array2::<f64>::zeros((num_ch, fft_size));

        let mut bands = Vec::with_capacity(num_bands);
        for b in 0..num_bands {
            let h = filter_spectra.slice(s![b, ..]);
            for ch in 0..num_ch {
                for k in 0..n_freq {
                    product[[ch, k]] = sig_spectra[[ch, k]] * h[k];
                }
            }

            ndifft_r2c(&product.view(), &mut ifft_buf.view_mut(), &fft_handler, 1);

            let band_data = ifft_buf.slice(s![.., ..n_out]).to_owned();
            bands.push(TimeSignal::new(band_data, signal.sample_rate())?);
        }

        Ok(bands)
    }
}

// ─── Convenience function ────────────────────────────────────────────────────

/// Design and apply a reconstructing linear-phase fractional octave filter bank.
///
/// Returns one filtered [`TimeSignal`] per band together with the centre
/// frequencies in Hz. The sum of all bands approximately equals the input
/// delayed by `n_samples / 2` samples; the final Hann window trades exact
/// reconstruction for improved FIR sidelobes.
///
/// Internally this calls [`OctaveBands::design`] followed by
/// [`OctaveBands::apply`]. Use those directly if you need to apply the same
/// filter bank to multiple signals.
///
/// # Parameters
/// - `signal` – input signal (any number of channels).
/// - `num_fractions` – octave fraction (1 = octave, 3 = third-octave, …).
/// - `frequency_range` – Hz range `(low, high)` for the band centres.
/// - `overlap` – transition overlap in `[0, 1]` (default `1.0`).
/// - `slope` – steepness iterations ≥ 0 (default `0`).
/// - `n_samples` – FIR filter length in samples (default `4096`).
pub fn reconstructing_fractional_octave_bands(
    signal: &TimeSignal,
    num_fractions: usize,
    frequency_range: (f64, f64),
    overlap: f64,
    slope: usize,
    n_samples: usize,
) -> Result<(Vec<TimeSignal>, Vec<f64>), FilterError> {
    let fb = OctaveBands::design(
        num_fractions,
        frequency_range,
        overlap,
        slope,
        n_samples,
        signal.sample_rate(),
    )?;
    let center_frequencies = fb.center_frequencies.clone();
    let bands = fb.apply(signal)?;
    Ok((bands, center_frequencies))
}

fn hanning_window(n: usize) -> Array1<f64> {
    if n <= 1 {
        return Array1::ones(n);
    }
    let n1 = (n - 1) as f64;
    Array1::from_iter(
        (0..n).map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n1).cos())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn impulse(n: usize, sample_rate: f64) -> TimeSignal {
        let mut data = Array2::<f64>::zeros((1, n));
        data[[0, 0]] = 1.0;
        TimeSignal::new(data, sample_rate).unwrap()
    }

    #[test]
    fn fractional_octave_frequencies_octave() {
        let (c, l, u) = fractional_octave_frequencies(1, (63.0, 16000.0));
        assert!(!c.is_empty());
        for &f in &c {
            assert!(f >= 63.0 * (1.0 - 1e-5) && f <= 16000.0 * (1.0 + 1e-5));
        }
        for ((&fc, &fl), &fu) in c.iter().zip(&l).zip(&u) {
            assert!(fl < fc && fc < fu);
        }
    }

    #[test]
    fn design_output_shape() {
        let n_samples = 256;
        let sample_rate = 44100.0;
        let fb = OctaveBands::design(1, (125.0, 8000.0), 1.0, 0, n_samples, sample_rate).unwrap();

        let (c, _, _) = fractional_octave_frequencies(1, (125.0, 8000.0));
        let expected_bands = c.iter().filter(|&&f| f < sample_rate / 2.0).count();

        assert_eq!(fb.impulse_responses.num_channels(), expected_bands);
        assert_eq!(fb.impulse_responses.num_time_steps(), n_samples);
        assert_eq!(fb.center_frequencies.len(), expected_bands);
    }

    #[test]
    fn reconstruction_impulse() {
        // Sum of all band outputs should equal the delayed impulse.
        let n = 512;
        let sr = 44100.0;
        let n_filt = 256_usize;

        let x = impulse(n, sr);
        let (bands, _) =
            reconstructing_fractional_octave_bands(&x, 1, (125.0, 8000.0), 1.0, 0, n_filt).unwrap();

        let n_out = n + n_filt - 1;
        let mut sum = Array1::<f64>::zeros(n_out);
        for band in &bands {
            sum += &band.channel(0);
        }

        // Peak should sit at n_filt/2 with amplitude ≈ 1
        let peak_idx = sum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(peak_idx, n_filt / 2);
        assert_abs_diff_eq!(sum[peak_idx], 1.0, epsilon = 0.05);
    }
}
