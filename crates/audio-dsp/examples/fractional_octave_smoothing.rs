/// Plot a synthetic transfer function before and after third-octave smoothing.
///
/// The original response contains narrow peaks, dips, and ripple so the
/// smoothing effect is easy to see in the frequency plot.
use audio_dsp::{FractionalOctaveSmoothingConfig, smooth_fractional_octave};
use audio_plot::freq::FreqPlotOptions;
use audio_signal::{join_freq, signal::FreqSignal};
use ndarray::Array2;
use num::complex::Complex64;

fn main() -> Result<(), audio_plot::Error> {
    let sample_rate = 48_000.0;
    let num_time_steps = 16_384;
    let num_bins = num_time_steps / 2 + 1;

    let original = synthetic_transfer_function(sample_rate, num_time_steps, num_bins);
    let config = FractionalOctaveSmoothingConfig::default()
        .with_mode(audio_dsp::FractionalOctaveSmoothingMode::MagnitudeZeroPhase);
    let (smoothed, stats) = smooth_fractional_octave(&original, &config).unwrap();

    println!("Requested smoothing: 1/3 octave");
    println!("Window length on warped axis: {}", stats.window_len);
    println!(
        "Actual smoothing width: 1/{:.3} octave",
        stats.actual_num_fractions
    );

    let comparison = join_freq!(original, smoothed).unwrap();

    audio_plot::show_freq_signal(
        "Fractional octave smoothing - original vs smoothed",
        &comparison,
        FreqPlotOptions::default(),
    )
}

fn synthetic_transfer_function(
    sample_rate: f64,
    num_time_steps: usize,
    num_bins: usize,
) -> FreqSignal {
    let mut data = Array2::zeros((1, num_bins));
    let nyquist = sample_rate / 2.0;

    for (bin, value) in data.row_mut(0).iter_mut().enumerate() {
        let frequency = bin as f64 * sample_rate / num_time_steps as f64;
        let normalized_frequency = if nyquist > 0.0 {
            (frequency / nyquist).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let base_db = -2.5 * normalized_frequency;
        let ripple_db = 2.5 * (70.0 * normalized_frequency.powf(0.7)).sin();
        let low_resonance_db = 10.0 * gaussian_log2(frequency, 180.0, 0.06);
        let broad_dip_db = -7.0 * gaussian_log2(frequency, 1_200.0, 0.12);
        let sharp_peak_db = 8.0 * gaussian_log2(frequency, 3_000.0, 0.035);
        let sharp_notch_db = -16.0 * gaussian_log2(frequency, 7_200.0, 0.02);
        let high_resonance_db = 6.0 * gaussian_log2(frequency, 12_500.0, 0.05);

        let magnitude_db = base_db
            + ripple_db
            + low_resonance_db
            + broad_dip_db
            + sharp_peak_db
            + sharp_notch_db
            + high_resonance_db;
        let magnitude = 10.0_f64.powf(magnitude_db / 20.0);

        let phase = -std::f64::consts::TAU * frequency * 0.0015;
        *value = Complex64::new(magnitude * phase.cos(), magnitude * phase.sin());
    }

    FreqSignal::new(data, sample_rate, Some(num_time_steps)).unwrap()
}

fn gaussian_log2(frequency: f64, center_hz: f64, width_octaves: f64) -> f64 {
    if frequency <= 0.0 {
        return 0.0;
    }

    let distance_octaves = (frequency / center_hz).log2();
    (-0.5 * (distance_octaves / width_octaves).powi(2)).exp()
}
