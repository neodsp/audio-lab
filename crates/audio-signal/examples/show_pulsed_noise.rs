use audio_plot::show_time_signal;
use audio_signal::test_signal::{noise::Spectrum, pulsed_noise::generate_pulsed_noise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = generate_pulsed_noise(
        4_800,
        2_400,
        240,
        4,
        Spectrum::White,
        0.8,
        false,
        48_000.0,
        1,
        123,
    )?;
    show_time_signal("Pulsed Noise Test Signal", &signal)?;
    Ok(())
}
