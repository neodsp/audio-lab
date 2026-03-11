/// Plot the magnitude responses of a reconstructing octave filter bank applied
/// to an impulse. Each band appears as a separate channel in the frequency plot.
use audio_dsp::filter_bank::reconstructing_fractional_octave_bands;
use audio_plot::freq::FreqDisplay;
use audio_signal::signal::{FreqSignal, TimeSignal};
use ndarray::{Array2, s};
use num::complex::Complex64;

fn main() -> Result<(), audio_plot::Error> {
    let sample_rate = 44100.0;
    let n_filt = 4096_usize;

    // Unit impulse – its spectrum is flat, so each band's output directly
    // shows the filter's magnitude response.
    let mut impulse_data = Array2::<f64>::zeros((1, n_filt * 2));
    impulse_data[[0, 0]] = 1.0;
    let impulse = TimeSignal::new(impulse_data, sample_rate).unwrap();

    let (bands, center_freqs) =
        reconstructing_fractional_octave_bands(&impulse, 1, (63.0, 16000.0), 1.0, 0, n_filt)
            .unwrap();

    println!("Octave band centre frequencies (Hz):");
    for (i, f) in center_freqs.iter().enumerate() {
        println!("  Band {i}: {f:.1} Hz");
    }

    // Convert each band to the frequency domain and stack into a single
    // multi-channel FreqSignal (one channel per band) for plotting.
    let num_bands = bands.len();
    let n_time = bands[0].num_time_steps();
    let n_freq = bands[0].num_freq_bins();

    let mut freq_data = Array2::<Complex64>::zeros((num_bands, n_freq));
    for (b, band) in bands.iter().enumerate() {
        let band_freq = band.clone().into_freq();
        freq_data
            .slice_mut(s![b, ..])
            .assign(&band_freq.freq_data().slice(s![0, ..]));
    }

    let freq_signal = FreqSignal::new(freq_data, sample_rate, Some(n_time)).unwrap();

    audio_plot::show_freq_signal(
        "Octave filter bank – magnitude responses",
        &freq_signal,
        true, // logarithmic frequency axis
        FreqDisplay::Magnitude { db: true },
    )
}
