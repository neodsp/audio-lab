use crate::signal::FreqSignal;

impl FreqSignal {
    pub fn freq_spacing(&self) -> f64 {
        self.sample_rate() / self.num_time_steps() as f64
    }

    pub fn nyquist(&self) -> f64 {
        self.sample_rate() / 2.0
    }

    pub fn nearest_freq_index(&self, freq: f64) -> usize {
        let index = (freq / self.freq_spacing()).round() as usize;
        index.min(self.freq_bins().len().saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use crate::signal::{FreqSignal, SignalError};

    #[test]
    fn frequency_helpers() -> Result<(), SignalError> {
        let signal = FreqSignal::zeros(1, 5, 16.0, Some(8))?;

        assert_eq!(signal.freq_spacing(), 2.0);
        assert_eq!(signal.nyquist(), 8.0);
        assert_eq!(signal.nearest_freq_index(0.1), 0);
        assert_eq!(signal.nearest_freq_index(3.1), 2);
        assert_eq!(signal.nearest_freq_index(99.0), 4);

        Ok(())
    }
}
