use crate::signal::FreqSignal;
use crate::signal::data::RealData;
use ndarray::{Array1, Zip};

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

    // Weight h[n] by n in place: FFT(n·h[n]) / FFT(h[n]) gives the group delay.
    let mut weighted_time = signal.to_time();
    let ramp = Array1::from_iter((0..weighted_time.num_time_steps()).map(|n| n as f64));
    for mut channel in weighted_time.channel_iter_mut() {
        channel *= &ramp;
    }
    let weighted_freq = weighted_time.into_freq();

    // Re(N / D) = Re(N · conj(D)) / |D|², which avoids a sqrt and a full
    // complex division per bin. |D|² < 1e-30 is the same cutoff as |D| < 1e-15.
    let group_delay = Zip::from(weighted_freq.freq_data())
        .and(signal.freq_data())
        .map_collect(|numerator, denominator| {
            let denominator_power = denominator.norm_sqr();
            if denominator_power < 1e-30 {
                0.0
            } else {
                (numerator * denominator.conj()).re / denominator_power
            }
        });

    Ok(RealData::new(freq_bins.to_owned(), group_delay)
        .expect("freq bins originate from FreqSignal"))
}

#[cfg(test)]
mod tests {
    use crate::signal::TimeSignal;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    use super::*;

    #[test]
    fn group_delay_matches_direct_complex_division() {
        let time = TimeSignal::new(
            arr2(&[
                [0.1, 0.7, -0.4, 0.2, 0.05, -0.1, 0.0, 0.02, 0.3],
                [0.0, 0.0, 1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0],
            ]),
            9.0,
        )
        .unwrap();
        let freq = time.to_freq();

        let mut weighted = time.time_data().to_owned();
        for mut channel in weighted.outer_iter_mut() {
            for (n, sample) in channel.iter_mut().enumerate() {
                *sample *= n as f64;
            }
        }
        let weighted_freq = TimeSignal::new(weighted, 9.0).unwrap().into_freq();

        let delay = group_delay(&freq).unwrap();
        for ch in 0..2 {
            for (bin, actual) in delay.channel(ch).iter().enumerate() {
                let expected = (weighted_freq.channel(ch)[bin] / freq.channel(ch)[bin]).re;
                assert_abs_diff_eq!(*actual, expected, epsilon = 1e-10);
            }
        }
    }

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
