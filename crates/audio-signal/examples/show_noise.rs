use audio_plot::show_time_signal;
use audio_signal::test_signal::noise::{Spectrum, generate_noise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = generate_noise(4_096, Spectrum::Pink, 0.8, 48_000.0, 1, 123)?;
    show_time_signal("Noise Test Signal", &signal)?;
    Ok(())
}
