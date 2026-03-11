use ndarray::Array2;

use crate::signal::TimeSignal;

pub fn read_audio(path: impl AsRef<std::path::Path>) -> Result<TimeSignal, audio_file::ReadError> {
    let audio = audio_file::read::<f64>(path, audio_file::ReadConfig::default())?;
    let num_channels = audio.num_channels as usize;
    let num_samples = audio.samples_interleaved.len() / num_channels;
    let data = Array2::from_shape_fn((num_channels, num_samples), |(ch, t)| {
        audio.samples_interleaved[t * num_channels + ch]
    });
    Ok(TimeSignal::new(data, audio.sample_rate as f64)
        .expect("audio-file returned a non-positive sample rate"))
}

pub fn write_audio(
    signal: &TimeSignal,
    path: impl AsRef<std::path::Path>,
) -> Result<(), audio_file::WriteError> {
    audio_file::write(
        path,
        &signal.interleaved_f64(),
        signal.num_channels() as u16,
        signal.sample_rate().round() as u32,
        audio_file::WriteConfig::default(),
    )
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use ndarray::arr2;

    use super::*;

    #[test]
    fn writes_and_reads_audio_file() {
        let signal = TimeSignal::new(arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), 48_000.0).unwrap();
        let path = std::env::temp_dir().join(format!(
            "audio-signal-{}-{}.wav",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        write_audio(&signal, &path).unwrap();
        let loaded = read_audio(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.sample_rate(), 48_000.0);
        assert_eq!(loaded.num_channels(), 2);
        assert_eq!(loaded.num_time_steps(), 3);
    }
}
