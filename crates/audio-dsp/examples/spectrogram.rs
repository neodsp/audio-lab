use audio_dsp::{StftConfig, WindowFn, stft};
use audio_plot::show_spectrogram;
use audio_signal::{
    PulsedNoiseConfig, SineConfig, Spectrum, SweepConfig, SweepType, generate_pulsed_noise,
    generate_sine, generate_sweep, mix,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 48_000.0;
    let duration_s = 6.0;
    let num_samples = (sample_rate * duration_s) as usize;

    let sine_config = SineConfig {
        sample_rate,
        amplitude: 0.0,
        num_channels: 1,
    };
    let sweep_config = SweepConfig {
        amplitude: 0.25,
        sample_rate,
        num_channels: 1,
        fade_out: (0.35 * sample_rate) as usize,
        sweep_type: SweepType::Exponential,
    };
    let noise_config = PulsedNoiseConfig {
        fade_length: 800,
        spectrum: Spectrum::Pink,
        amplitude: 0.18,
        frozen: false,
        sample_rate,
        num_channels: 1,
        seed: Some(7),
    };

    let fundamental = generate_sine(
        num_samples,
        880.0,
        &SineConfig {
            amplitude: 0.45,
            ..sine_config
        },
    )?;
    let second_harmonic = generate_sine(
        num_samples,
        1_760.0,
        &SineConfig {
            amplitude: 0.18,
            ..sine_config
        },
    )?;
    let third_harmonic = generate_sine(
        num_samples,
        2_640.0,
        &SineConfig {
            amplitude: 0.08,
            ..sine_config
        },
    )?;
    let chirp = generate_sweep(num_samples, 320.0..16_000.0, &sweep_config)?;
    let bursts = generate_pulsed_noise(2_400, 54_720, 6, &noise_config)?;

    let mut signal = mix!(fundamental, second_harmonic, third_harmonic, chirp, bursts)?;
    signal.normalize_peak(0.9);

    let spectrogram = stft(
        &signal,
        &StftConfig {
            window_size: 2_048,
            hop_size: 256,
            window_fn: WindowFn::Hann,
        },
    )?;

    show_spectrogram("Combined test-signal spectrogram", &spectrogram, -90.0)?;
    Ok(())
}
