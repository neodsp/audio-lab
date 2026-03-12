use std::time::Duration;

use audio_io::{AudioBackend, AudioBlock, AudioBlockMut, AudioHost, Config};

use crate::signal::TimeSignal;

pub fn play(signal: &TimeSignal) -> Result<(), audio_io::Error> {
    let num_output_channels = signal.num_channels() as u16;
    let sample_rate = signal.sample_rate().round() as u32;
    let num_frames = 512;
    let interleaved: Vec<f32> = signal.time_data().t().iter().map(|&s| s as f32).collect();
    let total_frames = signal.num_time_steps();
    let mut next_frame = 0usize;

    let mut host = AudioHost::new()?;
    host.start(
        Config {
            num_input_channels: 0,
            num_output_channels,
            sample_rate,
            num_frames,
        },
        move |_input, mut output| {
            for frame in 0..output.num_frames() {
                let src_frame = next_frame;
                next_frame += 1;

                for ch in 0..output.num_channels() as usize {
                    let sample = if src_frame < total_frames {
                        interleaved[src_frame * num_output_channels as usize + ch]
                    } else {
                        0.0
                    };
                    *output.sample_mut(ch as u16, frame) = sample;
                }
            }
        },
    )?;

    std::thread::sleep(Duration::from_secs_f64(
        signal.length_in_seconds() + num_frames as f64 / sample_rate as f64,
    ));
    host.stop()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn play_audio() {
        use std::f64::consts::TAU;

        let sample_rate = 48_000.0;
        let seconds = 2.0;
        let num_time_steps = (sample_rate * seconds) as usize;
        let frequency = 440.0;

        let data = ndarray::Array2::from_shape_fn((2, num_time_steps), |(_, frame)| {
            let t = frame as f64 / sample_rate;
            0.2 * (TAU * frequency * t).sin()
        });

        let signal = TimeSignal::new(data, sample_rate).unwrap();
        play(&signal).unwrap();
    }
}
