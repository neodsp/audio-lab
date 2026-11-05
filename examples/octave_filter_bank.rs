//! Octave filter bank applied to an impulse.
//!
//! Splits an impulse into octave bands (63 Hz to 16 kHz) with
//! `reconstructing_fractional_octave_bands` and plots the magnitude response of each band as
//! its own channel. The bands are linear-phase FIR filters that add up to roughly the
//! input, delayed by half the filter length. It also prints the band centre frequencies.
//!
//! ```sh
//! cargo run --example octave_filter_bank
//! ```

use audio_lab::dsp::filter_bank::{OctaveBandsConfig, reconstructing_fractional_octave_bands};

use audio_lab::signal::join_time_signals;
use audio_lab::signal::{ImpulseConfig, generate_impulse};

fn main() -> Result<(), audio_lab::plot::Error> {
    let sample_rate = 44100.0;
    let n_filt = 4096_usize;

    let impulse = generate_impulse(
        n_filt * 2,
        ImpulseConfig {
            sample_rate,
            ..Default::default()
        },
    )
    .unwrap();

    let (bands, center_freqs) = reconstructing_fractional_octave_bands(
        &impulse,
        1,
        &OctaveBandsConfig {
            n_samples: n_filt,
            ..Default::default()
        },
    )
    .unwrap();

    println!("Octave band centre frequencies (Hz):");
    for (i, f) in center_freqs.iter().enumerate() {
        println!("  Band {i}: {f:.1} Hz");
    }

    let freq_signal = join_time_signals(&bands).unwrap().into_freq();

    audio_lab::plot::show_freq("Octave filter bank – magnitude responses", &[&freq_signal])
}
