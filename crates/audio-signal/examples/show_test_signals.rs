use audio_plot::show_time;
use audio_signal::{
    NoiseConfig, SineConfig, Spectrum, SweepConfig, SweepType, generate_noise, generate_sine,
    generate_sweep, join_time,
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
