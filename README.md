<!-- cargo-rdme start -->

# audio-lab

`audio-lab` is a Rust library for offline audio research and algorithm testing. It provides a compact set of tools for generating signals, analyzing and manipulating them, plotting results, playing them back, and saving them to audio files or NumPy arrays. It is designed for experiments, validation, and measurement workflows, not for real-time processing.

## Features

- Generate test signals such as sweeps, sine waves, impulses, noise, and pulsed noise
- Work with time-domain signals, frequency-domain signals, and spectrograms
- Manipulate signals with trimming, gain, shifting, resampling, convolution, and deconvolution
- Plot waveforms, spectra, and spectrograms in native windows
- Play signals back for quick listening checks
- Save and load audio files and NumPy arrays

## Installation

Add `audio-lab` to your `Cargo.toml`:

```toml
[dependencies]
audio-lab = "0.1.0"
```

## Usage

### Generating a test signal

```rust
use audio_lab::signal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sweep = signal::generate_sweep(
        48_000,
        20.0..20_000.0,
        &signal::SweepConfig::default(),
    )?;

    let noise = signal::generate_noise(
        48_000,
        &signal::NoiseConfig {
            spectrum: signal::Spectrum::Pink,
            amplitude: 0.2,
            ..Default::default()
        },
    )?;

    let test_signal = audio_lab::mix_time!(sweep, noise)?;
    Ok(())
}
```

### Plotting

```rust
use audio_lab::{plot, signal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = signal::generate_sine(48_000, 440.0, &signal::SineConfig::default())?;

    plot::show_time("time", &signal)?;
    plot::show_freq("spectrum", &signal.clone().into_freq(), Default::default())?;
    Ok(())
}
```

### Playback

```rust
use audio_lab::signal::{self, playback};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = signal::generate_sine(48_000, 440.0, &signal::SineConfig::default())?;
    playback::play(&signal)?;
    Ok(())
}
```

### Saving to audio or NumPy

```rust
use audio_lab::signal::{
    self,
    io::{audio, npy},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = signal::generate_sweep(
        48_000,
        20.0..20_000.0,
        &signal::SweepConfig::default(),
    )?;

    audio::write_audio(&signal, "test.wav")?;
    npy::write_npy_time(&signal, "test.npy")?;
    Ok(())
}
```

### Computing a spectrogram

```rust
use audio_lab::{dsp, plot, signal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = signal::generate_sweep(
        48_000,
        20.0..20_000.0,
        &signal::SweepConfig::default(),
    )?;

    let spec = dsp::stft(&signal, &dsp::StftConfig::default())?;
    plot::show_spectrogram("spectrogram", &spec, Default::default())?;
    Ok(())
}
```

### Estimating an impulse response

```rust
use audio_lab::{dsp, signal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let excitation = signal::generate_sweep(
        48_000,
        20.0..20_000.0,
        &signal::SweepConfig::default(),
    )?;

    let recorded_output = excitation.clone();

    let impulse_response = dsp::deconvolve(
        &recorded_output,
        &excitation,
        &dsp::DeconvolutionConfig::default(),
    )?;

    println!("{}", impulse_response);
    Ok(())
}
```

## Workspace crates

- `audio-signal`: signal types, generators, I/O, playback
- `audio-dsp`: DSP and measurement primitives
- `audio-plot`: plotting and visualization

<!-- cargo-rdme end -->
