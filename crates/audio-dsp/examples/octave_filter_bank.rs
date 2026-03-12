/// Plot the magnitude responses of a reconstructing octave filter bank applied
/// to an impulse. Each band appears as a separate channel in the frequency plot.
use audio_dsp::filter_bank::{OctaveBandsConfig, reconstructing_fractional_octave_bands};
use audio_plot::freq::FreqPlotOptions;
use audio_signal::signal::join_signals;
use audio_signal::{ImpulseConfig, generate_impulse};

fn main() -> Result<(), audio_plot::Error> {
    let sample_rate = 44100.0;
    let n_filt = 4096_usize;

    let impulse = generate_impulse(
        n_filt * 2,
        &ImpulseConfig {
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

    let freq_signal = join_signals(&bands).unwrap().into_freq();

    audio_plot::show_freq_signal(
        "Octave filter bank – magnitude responses",
        &freq_signal,
        FreqPlotOptions::default(),
    )
}
