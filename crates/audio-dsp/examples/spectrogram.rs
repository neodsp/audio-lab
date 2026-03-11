/// Generate a more realistic synthetic audio clip and display its spectrogram.
///
/// The signal combines:
/// - a harmonic tone with slow vibrato,
/// - a rising chirp,
/// - short percussive noise bursts,
/// - gentle amplitude modulation.
///
/// This produces clearer time-varying structure than an impulse response plot.
use std::f64::consts::PI;

use audio_dsp::stft::{StftConfig, WindowFn, stft};
use audio_plot::show_spectrogram;
use audio_signal::signal::TimeSignal;
use ndarray::Array2;

fn smooth_pulse(t: f64, start: f64, end: f64, fade: f64) -> f64 {
    if t <= start - fade || t >= end + fade {
        return 0.0;
    }

    let fade_in = if t < start + fade {
        ((t - (start - fade)) / (2.0 * fade)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let fade_out = if t > end - fade {
        (((end + fade) - t) / (2.0 * fade)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let shape = fade_in.min(fade_out);
    0.5 - 0.5 * (PI * (1.0 - shape)).cos()
}

fn main() -> Result<(), audio_plot::Error> {
    let sample_rate = 48_000.0;
    let duration_s = 6.0;
    let num_samples = (sample_rate * duration_s) as usize;

    let data = Array2::from_shape_fn((1, num_samples), |(_, i)| {
        let t = i as f64 / sample_rate;

        let vibrato = (2.0 * PI * 5.0 * t).sin();
        let base_freq = 110.0 * 2.0_f64.powf(0.03 * vibrato);
        let harmonic_tone = (0.65 * (2.0 * PI * base_freq * t).sin())
            + (0.22 * (2.0 * PI * 2.0 * base_freq * t).sin())
            + (0.10 * (2.0 * PI * 3.0 * base_freq * t).sin());

        let chirp_progress = ((t - 1.0) / 3.5).clamp(0.0, 1.0);
        let chirp_freq = 350.0 * 2.0_f64.powf(3.2 * chirp_progress);
        let chirp_gate = smooth_pulse(t, 1.0, 4.6, 0.08);
        let chirp = chirp_gate * 0.30 * (2.0 * PI * chirp_freq * t).sin();

        let burst_times = [0.55, 1.35, 2.15, 3.05, 4.10, 5.00];
        let bursts = burst_times.iter().fold(0.0, |acc, &center| {
            let dt = t - center;
            let env = (-0.5 * (dt / 0.03).powi(2)).exp();
            let noisy = (2.0 * PI * 1900.0 * dt).sin()
                + 0.7 * (2.0 * PI * 3100.0 * dt).sin()
                + 0.4 * (2.0 * PI * 4700.0 * dt).sin();
            acc + 0.12 * env * noisy
        });

        let am = 0.72 + 0.28 * (2.0 * PI * 0.45 * t).sin().powi(2);
        let fade_out = (1.0 - ((t - 5.4) / 0.6).clamp(0.0, 1.0)).powi(2);

        am * fade_out * (harmonic_tone + chirp + bursts)
    });

    let signal = TimeSignal::new(data, sample_rate).unwrap();
    let spectrogram = stft(
        &signal,
        &StftConfig {
            window_size: 2048,
            hop_size: 256,
            window_fn: WindowFn::Hann,
        },
    )
    .unwrap();

    show_spectrogram("Synthetic audio spectrogram", &spectrogram, -90.0)
}
