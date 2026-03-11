use audio_plot::show_time_signal;
use audio_signal::test_signal::sine::generate_sine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = generate_sine(2_400, 440.0, 0.8, 48_000.0, 2)?;
    show_time_signal("Sine Test Signal", &signal)?;
    Ok(())
}
