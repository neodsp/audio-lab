//! The built-in test signals.
//!
//! Generates pink noise, a 440 Hz sine and an exponential sweep from 20 Hz to 20 kHz, 4096
//! samples each, and plots them over time.
//!
//! ```sh
//! cargo run --example show_test_signals
//! ```

use audio_lab::plot::show_time;
use audio_lab::signal::{
    NoiseConfig, SineConfig, Spectrum, SweepConfig, SweepType, generate_noise, generate_sine,
    generate_sweep,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let noise = generate_noise(
        4_096,
        NoiseConfig {
            spectrum: Spectrum::Pink,
            amplitude: 0.8,
            ..Default::default()
        },
    )?;
    let sine = generate_sine(
        4_096,
        440.0,
        SineConfig {
            amplitude: 0.8,
            ..Default::default()
        },
    )?;
    let sweep = generate_sweep(
        4_096,
        20.0..20_000.0,
        SweepConfig {
            amplitude: 0.8,
            sweep_type: SweepType::Exponential,
            ..Default::default()
        },
    )?;

    show_time("Test Signals", &[&noise, &sine, &sweep])?;
    Ok(())
}
