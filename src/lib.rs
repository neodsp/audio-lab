//! # audio-lab
//!
//! `audio-lab` is a Rust library for offline audio research and algorithm testing. It provides a compact set of tools for generating signals, analyzing and manipulating them, plotting results, playing them back, and saving them to audio files or NumPy arrays. It is designed for experiments, validation, and measurement workflows, not for real-time processing.
//!
//! ## Features
//!
//! - Generate test signals such as sweeps, sine waves, impulses, noise, and pulsed noise
//! - Work with time-domain signals, frequency-domain signals, and spectrograms
//! - Manipulate signals with trimming, gain, shifting, resampling, convolution, and deconvolution
//! - Plot waveforms, spectra, and spectrograms in native windows
//! - Play signals back for quick listening checks
//! - Save and load audio files and NumPy arrays
//!
//! ## Installation
//!
//! Add `audio-lab` to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! audio-lab = "0.1.0"
//! ```
//!
//! ## Usage
//!
//! ### Generating a test signal
//!
//! ```rust
//! use audio_lab::signal;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let sweep = signal::generate_sweep(
//!         48_000,
//!         20.0..20_000.0,
//!         signal::SweepConfig::default(),
//!     )?;
//!
//!     let noise = signal::generate_noise(
//!         48_000,
//!         signal::NoiseConfig {
//!             spectrum: signal::Spectrum::Pink,
//!             amplitude: 0.2,
//!             ..Default::default()
//!         },
//!     )?;
//!
//!     let test_signal = audio_lab::signal::mix_time!(sweep, noise)?;
//!     Ok(())
//! }
//! ```
//!
//! ### Plotting
//!
//! ```rust,no_run
//! use audio_lab::{plot, signal};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let signal = signal::generate_sine(48_000, 440.0, signal::SineConfig::default())?;
//!
//!     plot::show_time("time", &[&signal])?;
//!     plot::show_freq("spectrum", &[&signal.into_freq()])?;
//!     Ok(())
//! }
//! ```
//!
//! ### Playback
//!
//! ```rust,no_run
//! use audio_lab::signal::{self, playback};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let signal = signal::generate_sine(48_000, 440.0, signal::SineConfig::default())?;
//!     playback::play(&signal)?;
//!     Ok(())
//! }
//! ```
//!
//! ### Saving to audio or NumPy
//!
//! ```rust,no_run
//! use audio_lab::signal::{
//!     self,
//!     fs::{audio, npy},
//! };
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let signal = signal::generate_sweep(
//!         48_000,
//!         20.0..20_000.0,
//!         signal::SweepConfig::default(),
//!     )?;
//!
//!     audio::write_audio(&signal, "test.wav", audio::SampleFormat::Float32)?;
//!     npy::write_npy_time(&signal, "test.npy")?;
//!     Ok(())
//! }
//! ```
//!
//! ### Computing a spectrogram
//!
//! ```rust,no_run
//! use audio_lab::{dsp, plot, signal};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let signal = signal::generate_sweep(
//!         48_000,
//!         20.0..20_000.0,
//!         signal::SweepConfig::default(),
//!     )?;
//!
//!     let spec = dsp::stft(&signal, &dsp::StftConfig::default())?;
//!     plot::show_spectrogram("spectrogram", &spec, Default::default())?;
//!     Ok(())
//! }
//! ```
//!
//! ### Estimating an impulse response
//!
//! Play a sweep on a speaker, record it with a microphone and deconvolve. Several speakers and
//! mics work the same way, see `examples/measure_impulse_response.rs`. For live measurements with
//! noise, see `examples/live_noise.rs`.
//!
//! ```rust,no_run
//! use audio_lab::signal::measure::{SequenceConfig, Step};
//! use audio_lab::signal::playback::{DeviceConfig, play_and_record};
//! use audio_lab::{dsp, signal};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let excitation = signal::generate_sweep(
//!         1 << 18,
//!         20.0..20_000.0,
//!         signal::SweepConfig::default(),
//!     )?;
//!
//!     // the speaker on output 0 plays the sweep while the mic on input 0 records
//!     let steps = vec![Step::new(vec![0], excitation.clone())];
//!     let recordings = play_and_record(steps, &SequenceConfig::default(), &DeviceConfig::default())?;
//!
//!     let impulse_response = dsp::deconvolve(
//!         &recordings[0],
//!         &excitation,
//!         &dsp::DeconvolutionConfig::default(),
//!     )?;
//!
//!     println!("{}", impulse_response);
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - `signal`: signal types, generators, I/O, playback and recording measurements
//! - `dsp`: DSP and measurement primitives
//! - `plot`: plotting and visualization
//!
//! ## Cargo features
//!
//! All of these are enabled by default, except `serde`:
//!
//! - `plot`: the `plot` module. Pulls in egui, so turn it off if you don't need windows.
//! - `playback`: `signal::playback`, plays signals on an audio device and records measurements.
//! - `audio-file`: `signal::fs::audio`, reads and writes audio files.
//! - `numpy`: `signal::fs::npy`, reads and writes NumPy `.npy` files.
//! - `serde`: `Serialize` and `Deserialize` for the signal types.

pub mod dsp;
#[cfg(feature = "plot")]
pub mod plot;
pub mod signal;
