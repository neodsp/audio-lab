use crate::signal::FreqSignal;

pub fn freq_spacing(signal: &FreqSignal) -> f64 {
    signal.sample_rate() / signal.num_time_steps() as f64
}

pub fn nearest_frequency_index(signal: &FreqSignal, freq: f64) -> usize {
    let index = (freq / freq_spacing(signal)).round() as usize;
    index.min(signal.freq_bins().len().saturating_sub(1))
}

pub fn nyquist_frequency(signal: &FreqSignal) -> f64 {
    signal.sample_rate() / 2.0
}

#[cfg(test)]
mod tests {
    use crate::signal::{FreqSignal, SignalError};

    use super::*;

    #[test]
    fn frequency_helpers() -> Result<(), SignalError> {
        let signal = FreqSignal::zeros(1, 5, 16.0, Some(8))?;

        assert_eq!(freq_spacing(&signal), 2.0);
        assert_eq!(nyquist_frequency(&signal), 8.0);
        assert_eq!(nearest_frequency_index(&signal, 0.1), 0);
        assert_eq!(nearest_frequency_index(&signal, 3.1), 2);
        assert_eq!(nearest_frequency_index(&signal, 99.0), 4);

        Ok(())
    }
}
