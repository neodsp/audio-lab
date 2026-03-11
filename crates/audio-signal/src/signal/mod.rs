use ndarray::prelude::*;

pub use freq_signal::FreqSignal;
pub use spectrogram::Spectrogram;
pub use time_signal::TimeSignal;

pub mod freq_signal;
pub mod spectrogram;
pub mod time_signal;

#[derive(Debug, thiserror::Error)]
pub enum SignalError {
    #[error("Sample Rate must be > 0")]
    SampleRateZeroOrNeg,
}

pub mod utils {
    use super::*;

    pub fn f_from_t(t: usize) -> usize {
        t / 2 + 1
    }

    pub fn t_from_f(f: usize) -> usize {
        f.saturating_sub(1) * 2
    }

    pub fn generate_time_steps(num: usize, sample_rate: f64) -> Array1<f64> {
        if num == 0 {
            return Array1::zeros(0);
        }
        let end_time = (num - 1) as f64 / sample_rate;
        Array1::linspace(0.0, end_time, num)
    }

    pub fn generate_freq_steps(num: usize, sample_rate: f64, num_time_steps: usize) -> Array1<f64> {
        if num == 0 || num_time_steps == 0 {
            return Array1::zeros(num);
        }
        let end_freq = (num - 1) as f64 * sample_rate / num_time_steps as f64;
        Array1::linspace(0.0, end_freq, num)
    }
}
