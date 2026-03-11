use audio_plot::show_time_signal;
use audio_signal::test_signal::sweep::{SweepType, generate_sweep};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let signal = generate_sweep(
        48_000,
        20.0..20_000.0,
        0.8,
        48_000.0,
        1,
        2_400,
        SweepType::Exponential,
    )?;
    show_time_signal("Sweep Test Signal", &signal)?;
    Ok(())
}
