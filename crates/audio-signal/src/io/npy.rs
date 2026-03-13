use crate::signal::{FreqSignal, SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum NpyOrSignalError {
    #[error(transparent)]
    Npy(#[from] ndarray_npy::ReadNpyError),
    #[error(transparent)]
    Signal(#[from] SignalError),
}

pub fn read_npy_time(
    path: impl AsRef<std::path::Path>,
    sample_rate: f64,
) -> Result<TimeSignal, NpyOrSignalError> {
    let y_data = ndarray_npy::read_npy(path).map_err(NpyOrSignalError::Npy)?;
    TimeSignal::new(y_data, sample_rate).map_err(NpyOrSignalError::Signal)
}

pub fn write_npy_time(
    signal: &TimeSignal,
    path: impl AsRef<std::path::Path>,
) -> Result<(), ndarray_npy::WriteNpyError> {
    ndarray_npy::write_npy(path, &signal.time_data().to_owned())
}

pub fn read_npy_freq(
    path: impl AsRef<std::path::Path>,
    sample_rate: f64,
    num_time_steps: Option<usize>,
) -> Result<FreqSignal, NpyOrSignalError> {
    let y_data = ndarray_npy::read_npy(path).map_err(NpyOrSignalError::Npy)?;
    FreqSignal::new(y_data, sample_rate, num_time_steps).map_err(NpyOrSignalError::Signal)
}

pub fn write_npy_freq(
    signal: &FreqSignal,
    path: impl AsRef<std::path::Path>,
) -> Result<(), ndarray_npy::WriteNpyError> {
    ndarray_npy::write_npy(path, &signal.freq_data().to_owned())
}

#[deprecated(note = "use read_npy_time")]
pub fn read_numpy_time_signal(
    path: impl AsRef<std::path::Path>,
    sample_rate: f64,
) -> Result<TimeSignal, NpyOrSignalError> {
    read_npy_time(path, sample_rate)
}

#[deprecated(note = "use write_npy_time")]
pub fn write_numpy_time_signal(
    signal: &TimeSignal,
    path: impl AsRef<std::path::Path>,
) -> Result<(), ndarray_npy::WriteNpyError> {
    write_npy_time(signal, path)
}

#[deprecated(note = "use read_npy_freq")]
pub fn read_numpy_freq_signal(
    path: impl AsRef<std::path::Path>,
    sample_rate: f64,
    num_time_steps: Option<usize>,
) -> Result<FreqSignal, NpyOrSignalError> {
    read_npy_freq(path, sample_rate, num_time_steps)
}

#[deprecated(note = "use write_npy_freq")]
pub fn write_numpy_freq_signal(
    signal: &FreqSignal,
    path: impl AsRef<std::path::Path>,
) -> Result<(), ndarray_npy::WriteNpyError> {
    write_npy_freq(signal, path)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use ndarray::arr2;
    use num::complex::Complex64;

    use super::*;

    #[test]
    fn writes_and_reads_time_signal_npy() {
        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 44_100.0).unwrap();
        let path = std::env::temp_dir().join(format!(
            "audio-signal-{}-{}.npy",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        write_npy_time(&signal, &path).unwrap();
        let loaded = read_npy_time(&path, 44_100.0).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded, signal);
    }

    #[test]
    fn writes_and_reads_freq_signal_npy() {
        let signal = FreqSignal::new(
            arr2(&[
                [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
                [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)],
            ]),
            44_100.0,
            Some(3),
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "audio-signal-freq-{}-{}.npy",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        write_npy_freq(&signal, &path).unwrap();
        let loaded = read_npy_freq(&path, 44_100.0, Some(3)).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded, signal);
    }
}
