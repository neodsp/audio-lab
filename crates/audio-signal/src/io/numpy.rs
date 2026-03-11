use crate::signal::{SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum NpyOrSignalError {
    #[error(transparent)]
    Npy(#[from] ndarray_npy::ReadNpyError),
    #[error(transparent)]
    Signal(#[from] SignalError),
}

pub fn read_time_signal(
    path: impl AsRef<std::path::Path>,
    sample_rate: f64,
) -> Result<TimeSignal, NpyOrSignalError> {
    let data = crate::data::real_data::RealData::from_npy(path).map_err(NpyOrSignalError::Npy)?;
    TimeSignal::from_real_data(data, sample_rate).map_err(NpyOrSignalError::Signal)
}

pub fn write_time_signal(
    signal: &TimeSignal,
    path: impl AsRef<std::path::Path>,
) -> Result<(), ndarray_npy::WriteNpyError> {
    ndarray_npy::write_npy(path, &signal.time_data().to_owned())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use ndarray::arr2;

    use super::*;

    #[test]
    fn writes_and_reads_npy() {
        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 44_100.0).unwrap();
        let path = std::env::temp_dir().join(format!(
            "audio-signal-{}-{}.npy",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        write_time_signal(&signal, &path).unwrap();
        let loaded = read_time_signal(&path, 44_100.0).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded, signal);
    }
}
