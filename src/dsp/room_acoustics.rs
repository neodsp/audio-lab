//! Room acoustics from a measured impulse response (ISO 3382-1).
//!
//! [`energy_decay_curve`] integrates the squared impulse response backwards (Schroeder), and
//! stops where the decay meets the noise floor, so the noise doesn't flatten the end of the curve.

use std::time::Duration;

use crate::signal::{SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum RoomAcousticsError {
    #[error("the impulse response must not be empty")]
    EmptySignal,
    #[error("channel {channel} of the impulse response is silent")]
    SilentChannel { channel: usize },
    #[error("the noise floor fraction must be > 0 and < 1, got {0}")]
    InvalidNoiseFloorFraction(f64),
    #[error("the noise margin must be finite and >= 0 dB, got {0} dB")]
    InvalidNoiseMargin(f64),
    #[error("the block duration must map to at least one sample, got {0:?}")]
    InvalidBlockDuration(Duration),
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyDecayConfig {
    /// The last part of the impulse response, as a fraction of its length, that holds only
    /// noise. Its mean power is the noise floor. ISO 3382-1 and Lundeby use the last 10 %.
    pub noise_floor_fraction: f64,
    /// The integration stops at the first block after the peak whose mean power is at most
    /// this much above the noise floor.
    pub noise_margin_db: f64,
    /// Length of the blocks the squared impulse response is averaged over to find where the
    /// decay meets the noise floor. 10 to 50 ms is common.
    pub block_duration: Duration,
}

impl Default for EnergyDecayConfig {
    fn default() -> Self {
        Self {
            noise_floor_fraction: 0.1,
            noise_margin_db: 5.0,
            block_duration: Duration::from_millis(10),
        }
    }
}

impl EnergyDecayConfig {
    pub fn with_noise_floor_fraction(mut self, noise_floor_fraction: f64) -> Self {
        self.noise_floor_fraction = noise_floor_fraction;
        self
    }

    pub fn with_noise_margin_db(mut self, noise_margin_db: f64) -> Self {
        self.noise_margin_db = noise_margin_db;
        self
    }

    pub fn with_block_duration(mut self, block_duration: Duration) -> Self {
        self.block_duration = block_duration;
        self
    }

    fn validate(&self, sample_rate: f64) -> Result<usize, RoomAcousticsError> {
        if !(self.noise_floor_fraction > 0.0 && self.noise_floor_fraction < 1.0) {
            return Err(RoomAcousticsError::InvalidNoiseFloorFraction(
                self.noise_floor_fraction,
            ));
        }
        if !self.noise_margin_db.is_finite() || self.noise_margin_db < 0.0 {
            return Err(RoomAcousticsError::InvalidNoiseMargin(self.noise_margin_db));
        }
        let block_len = (self.block_duration.as_secs_f64() * sample_rate).round() as usize;
        if block_len == 0 {
            return Err(RoomAcousticsError::InvalidBlockDuration(
                self.block_duration,
            ));
        }
        Ok(block_len)
    }
}

/// The result of [`energy_decay_curve`].
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyDecay {
    /// The energy decay curve, one channel per channel of the impulse response, linear and
    /// normalized to 1 at the first sample. It is 0 from the truncation point on.
    pub curve: TimeSignal,
    /// Per channel, the sample where the integration stops: the decay met the noise floor
    /// there. The impulse response length if it never did.
    pub truncation: Vec<usize>,
    /// Per channel, the noise floor in dB relative to the peak power of the impulse response
    /// (its peak-to-noise ratio, negated). `-inf` for a noise-free impulse response.
    pub noise_floor_db: Vec<f64>,
}

impl EnergyDecay {
    /// The curve in dB, 0 dB at the first sample and `-inf` from the truncation point on.
    pub fn curve_db(&self) -> TimeSignal {
        let mut curve = self.curve.clone();
        curve.iter_mut().for_each(|e| *e = power_to_db(*e));
        curve
    }
}

/// The energy decay curve of every channel of `impulse_response`, by Schroeder backward
/// integration: `E(t) = ∫ₜ h²(τ) dτ`, normalized to 1 at the first sample.
///
/// The curve starts at the first sample, so trim the impulse response to its start first,
/// e.g. with [`find_impulse_response_start`](crate::dsp::find_impulse_response_start).
///
/// Integrating the noise after the decay would flatten the end of the curve and lengthen
/// reverberation times. So the noise floor is the mean power of the last
/// [`noise_floor_fraction`](EnergyDecayConfig::noise_floor_fraction) of the impulse response,
/// and the integration stops at the first block after the peak that is within
/// [`noise_margin_db`](EnergyDecayConfig::noise_margin_db) of it. The energy that the decay
/// would have had after that point is left out, which bends the very end of the curve down
/// slightly. The end is taken as noise even if it is still part of the decay (e.g. a
/// synthetic impulse response), so make it long enough for the decay to reach the floor.
pub fn energy_decay_curve(
    impulse_response: &TimeSignal,
    config: &EnergyDecayConfig,
) -> Result<EnergyDecay, RoomAcousticsError> {
    let sample_rate = impulse_response.sample_rate();
    let block_len = config.validate(sample_rate)?;
    let len = impulse_response.num_time_steps();
    if len == 0 || impulse_response.num_channels() == 0 {
        return Err(RoomAcousticsError::EmptySignal);
    }

    let noise_len = ((len as f64 * config.noise_floor_fraction).ceil() as usize).clamp(1, len);
    let noise_start = len - noise_len;
    let margin = 10.0_f64.powf(config.noise_margin_db / 10.0);

    let mut curve = TimeSignal::zeros(impulse_response.num_channels(), len, sample_rate)?;
    let mut truncation = Vec::with_capacity(impulse_response.num_channels());
    let mut noise_floor_db = Vec::with_capacity(impulse_response.num_channels());

    for (channel, (ir, mut edc)) in impulse_response
        .channel_iter()
        .zip(curve.channel_iter_mut())
        .enumerate()
    {
        let power: Vec<f64> = ir.iter().map(|x| x * x).collect();
        let (peak_index, peak_power) = power
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0.0), |max, (i, p)| if p > max.1 { (i, p) } else { max });
        if peak_power == 0.0 {
            return Err(RoomAcousticsError::SilentChannel { channel });
        }

        let noise_power = power[noise_start..].iter().sum::<f64>() / noise_len as f64;
        let threshold = noise_power * margin;
        let end = power
            .chunks(block_len)
            .enumerate()
            .skip(peak_index / block_len + 1)
            .find(|(_, block)| block.iter().sum::<f64>() / block.len() as f64 <= threshold)
            .map_or(len, |(block, _)| block * block_len);

        let mut energy = 0.0;
        for (e, p) in edc.iter_mut().zip(&power).take(end).rev() {
            energy += p;
            *e = energy;
        }
        edc.iter_mut().take(end).for_each(|e| *e /= energy);

        truncation.push(end);
        noise_floor_db.push(power_to_db(noise_power / peak_power));
    }

    Ok(EnergyDecay {
        curve,
        truncation,
        noise_floor_db,
    })
}

fn power_to_db(power: f64) -> f64 {
    10.0 * power.log10()
}

#[cfg(test)]
pub(crate) mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    use super::*;
    use crate::signal::{NoiseConfig, generate_noise};

    pub(crate) const SAMPLE_RATE: f64 = 48_000.0;

    /// White noise that decays by 60 dB in `t60` seconds, plus a stationary noise floor
    /// `noise_floor_db` below the decay's initial mean power (`None` for no noise floor).
    pub(crate) fn decaying_noise(
        t60: f64,
        duration: f64,
        noise_floor_db: Option<f64>,
        seed: u64,
    ) -> TimeSignal {
        let len = (duration * SAMPLE_RATE) as usize;
        let noise = |seed| {
            generate_noise(
                len,
                NoiseConfig {
                    sample_rate: SAMPLE_RATE,
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let decay = noise(seed);
        let floor = noise(seed + 1);
        // Uniform noise in ±1 has a mean power of 1/3, so this keeps the initial power at 1.
        let scale = 3.0_f64.sqrt();
        let floor_gain = noise_floor_db.map_or(0.0, |db| 10.0_f64.powf(db / 20.0));
        let data = Array2::from_shape_fn((1, len), |(_, n)| {
            let t = n as f64 / SAMPLE_RATE;
            let envelope = 10.0_f64.powf(-3.0 * t / t60);
            scale * (decay.channel(0)[n] * envelope + floor.channel(0)[n] * floor_gain)
        });
        TimeSignal::new(data, SAMPLE_RATE).unwrap()
    }

    /// The slope of a least-squares line through the curve (in dB) between `upper` and
    /// `lower` dB, as a T60 in seconds.
    fn t60_from_curve(decay: &EnergyDecay, upper: f64, lower: f64) -> f64 {
        let db = decay.curve_db();
        let points: Vec<(f64, f64)> = db
            .channel(0)
            .iter()
            .enumerate()
            .filter(|(_, l)| (lower..=upper).contains(*l))
            .map(|(n, l)| (n as f64 / SAMPLE_RATE, *l))
            .collect();
        let count = points.len() as f64;
        let mean_t = points.iter().map(|p| p.0).sum::<f64>() / count;
        let mean_l = points.iter().map(|p| p.1).sum::<f64>() / count;
        let cov: f64 = points.iter().map(|p| (p.0 - mean_t) * (p.1 - mean_l)).sum();
        let var: f64 = points.iter().map(|p| (p.0 - mean_t).powi(2)).sum();
        -60.0 / (cov / var)
    }

    #[test]
    fn curve_starts_at_one_and_decreases() {
        let ir = decaying_noise(0.5, 1.0, None, 1);
        let decay = energy_decay_curve(&ir, &EnergyDecayConfig::default()).unwrap();
        let curve = decay.curve.channel(0);
        assert_abs_diff_eq!(curve[0], 1.0, epsilon = 1e-12);
        assert!(curve.windows(2).into_iter().all(|w| w[1] <= w[0]));
        // The last 10 % is taken as noise even though there is none, so the decay there is
        // well within the margin of the "floor" and the curve stops where that part starts.
        assert_eq!(decay.truncation, vec![ir.num_time_steps() * 9 / 10]);
    }

    #[test]
    fn slope_matches_t60_without_noise() {
        for t60 in [0.3, 0.8] {
            let ir = decaying_noise(t60, 3.0 * t60, None, 2);
            let decay = energy_decay_curve(&ir, &EnergyDecayConfig::default()).unwrap();
            let measured = t60_from_curve(&decay, -5.0, -35.0);
            assert_abs_diff_eq!(measured, t60, epsilon = 0.03 * t60);
        }
    }

    #[test]
    fn truncates_at_the_noise_floor() {
        let t60 = 0.5;
        let ir = decaying_noise(t60, 1.5, Some(-60.0), 3);
        let decay = energy_decay_curve(&ir, &EnergyDecayConfig::default()).unwrap();

        // The decay is 5 dB above the floor at -55 dB, i.e. after 55/60 of the T60.
        let expected = 55.0 / 60.0 * t60;
        let truncation = decay.truncation[0] as f64 / SAMPLE_RATE;
        assert_abs_diff_eq!(truncation, expected, epsilon = 0.05);
        let curve = decay.curve.channel(0);
        assert!(curve.iter().skip(decay.truncation[0]).all(|e| *e == 0.0));

        // The peak is √3 (uniform noise scaled to unit power), 4.8 dB above the initial power.
        assert_abs_diff_eq!(decay.noise_floor_db[0], -64.8, epsilon = 0.5);

        let measured = t60_from_curve(&decay, -5.0, -35.0);
        assert_abs_diff_eq!(measured, t60, epsilon = 0.03 * t60);
    }

    #[test]
    fn noise_floor_lengthens_the_decay_without_truncation() {
        // Without truncation (the margin can't be reached), the noise flattens the curve.
        let t60 = 0.5;
        let ir = decaying_noise(t60, 1.5, Some(-40.0), 4);
        let config = EnergyDecayConfig::default();
        let truncated = t60_from_curve(&energy_decay_curve(&ir, &config).unwrap(), -5.0, -25.0);
        assert_abs_diff_eq!(truncated, t60, epsilon = 0.05 * t60);

        let untruncated = EnergyDecayConfig {
            noise_floor_fraction: 0.01,
            noise_margin_db: 0.0,
            block_duration: Duration::from_secs(10),
        };
        let flattened =
            t60_from_curve(&energy_decay_curve(&ir, &untruncated).unwrap(), -5.0, -25.0);
        assert!(flattened > 1.1 * t60, "{flattened}");
    }

    #[test]
    fn several_channels_are_independent() {
        let short = decaying_noise(0.3, 1.0, Some(-50.0), 5);
        let long = decaying_noise(0.6, 1.0, Some(-50.0), 7);
        let mut data = Array2::zeros((2, short.num_time_steps()));
        data.row_mut(0).assign(&short.channel(0));
        data.row_mut(1).assign(&long.channel(0));
        let ir = TimeSignal::new(data, SAMPLE_RATE).unwrap();

        let decay = energy_decay_curve(&ir, &EnergyDecayConfig::default()).unwrap();
        let single = energy_decay_curve(&short, &EnergyDecayConfig::default()).unwrap();
        assert_eq!(decay.truncation[0], single.truncation[0]);
        assert!(decay.truncation[1] > decay.truncation[0]);
        assert_abs_diff_eq!(
            decay.curve.channel(0),
            single.curve.channel(0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn rejects_invalid_input() {
        let ir = decaying_noise(0.5, 0.5, None, 8);
        let config = EnergyDecayConfig::default();
        let check = |config: EnergyDecayConfig| energy_decay_curve(&ir, &config).unwrap_err();

        assert!(matches!(
            check(config.clone().with_noise_floor_fraction(1.0)),
            RoomAcousticsError::InvalidNoiseFloorFraction(_)
        ));
        assert!(matches!(
            check(config.clone().with_noise_margin_db(f64::NAN)),
            RoomAcousticsError::InvalidNoiseMargin(_)
        ));
        assert!(matches!(
            check(config.clone().with_block_duration(Duration::ZERO)),
            RoomAcousticsError::InvalidBlockDuration(_)
        ));

        let silent = TimeSignal::zeros(2, 100, SAMPLE_RATE).unwrap();
        assert!(matches!(
            energy_decay_curve(&silent, &config).unwrap_err(),
            RoomAcousticsError::SilentChannel { channel: 0 }
        ));
        let empty = TimeSignal::zeros(1, 0, SAMPLE_RATE).unwrap();
        assert!(matches!(
            energy_decay_curve(&empty, &config).unwrap_err(),
            RoomAcousticsError::EmptySignal
        ));
    }
}
