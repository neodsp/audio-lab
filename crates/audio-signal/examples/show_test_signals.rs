use audio_plot::show_time_signal;
use audio_signal::{
    join,
    test_signal::{
        noise::{Spectrum, generate_noise},
        sine::generate_sine,
        sweep::{SweepType, generate_sweep},
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let noise = generate_noise(4_096, Spectrum::Pink, 0.8, 48_000.0, 1, 123)?;
    let sine = generate_sine(4_096, 440.0, 0.8, 48_000.0, 1)?;
    let sweep = generate_sweep(
        4_096,
        20.0..20_000.0,
        0.8,
        48_000.0,
        1,
        240,
        SweepType::Exponential,
    )?;

    let signal = join!(noise, sine, sweep)?;
    show_time_signal("Noise Test Signal", &signal)?;
    Ok(())
}
