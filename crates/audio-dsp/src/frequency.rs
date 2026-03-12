use audio_signal::data::RealData;
use audio_signal::signal::FreqSignal;
use ndarray::Array2;

#[derive(Debug, thiserror::Error)]
pub enum FrequencyError {
    #[error("group delay requires at least two frequency bins")]
    TooFewFrequencyBins,
    #[error("frequency bins must be strictly increasing")]
    NonIncreasingFrequencyBins,
}

pub fn group_delay(signal: &FreqSignal) -> Result<RealData, FrequencyError> {
    let freq_bins = signal.freq_bins();
    if freq_bins.len() < 2 {
        return Err(FrequencyError::TooFewFrequencyBins);
    }

    if freq_bins
        .windows(2)
        .into_iter()
        .any(|window| window[1] <= window[0])
    {
        return Err(FrequencyError::NonIncreasingFrequencyBins);
    }

    let time = signal.clone().into_time();
    let weighted_time = Array2::from_shape_fn(time.time_data().raw_dim(), |(channel, sample)| {
        time.time_data()[[channel, sample]] * sample as f64
    });
    let weighted_freq = audio_signal::signal::TimeSignal::new(weighted_time, time.sample_rate())
        .expect("weighted time data preserves valid signal dimensions")
        .into_freq();

    let mut group_delay = Array2::zeros((signal.num_channels(), signal.num_freq_bins()));
    for ((mut out_channel, numerator), denominator) in group_delay
        .outer_iter_mut()
        .zip(weighted_freq.freq_data().outer_iter())
        .zip(signal.freq_data().outer_iter())
    {
        for bin in 0..freq_bins.len() {
            if denominator[bin].norm() < 1e-15 {
                out_channel[bin] = 0.0;
            } else {
                out_channel[bin] = (numerator[bin] / denominator[bin]).re;
            }
        }
    }

    Ok(RealData::new(freq_bins.to_owned(), group_delay)
        .expect("freq bins originate from FreqSignal"))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use audio_signal::signal::TimeSignal;
    use ndarray::arr2;

    use super::*;

    #[test]
    fn group_delay_matches_integer_sample_delay() {
        let delayed = TimeSignal::new(arr2(&[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 8.0)
            .unwrap()
            .into_freq();

        let delay = group_delay(&delayed).unwrap();

        for value in delay
            .channel(0)
            .iter()
            .skip(1)
            .take(delay.num_data_points() - 2)
        {
            assert_abs_diff_eq!(*value, 2.0, epsilon = 1e-10);
        }
    }
}
