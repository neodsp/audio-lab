//! Room acoustics from a measured impulse response (ISO 3382-1).
//!
//! - [`energy_decay_curve`] integrates the squared impulse response backwards (Schroeder), and
//!   stops where the decay meets the noise floor, so the noise doesn't flatten the end of the
//!   curve.
//! - [`room_parameters`] gives the reverberation times T20 and T30, the early decay time, the
//!   clarity C50 and C80 and the definition D50, and [`room_parameters_in_bands`] gives them per
//!   octave or third-octave band.

use std::time::Duration;

use ndarray::s;

use crate::dsp::filter_bank::{FilterError, OctaveBands};
use crate::dsp::time::{TimeError, find_impulse_response_start};
use crate::signal::{SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum RoomAcousticsError {
    #[error("the impulse response must not be empty")]
    EmptySignal,
    #[error(
        "sample rates do not match: the impulse response has {signal} Hz, the bands {bands} Hz"
    )]
    SampleRateMismatch { signal: f64, bands: f64 },
    #[error("in the band at {center_frequency:.0} Hz: {source}")]
    Band {
        center_frequency: f64,
        source: Box<RoomAcousticsError>,
    },
    #[error(transparent)]
    Time(#[from] TimeError),
    #[error(transparent)]
    Filter(#[from] FilterError),
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
/// e.g. with [`find_impulse_response_start`].
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

#[derive(Debug, Clone, PartialEq)]
pub struct RoomParametersConfig {
    /// How the energy decay curve is computed and where it stops.
    pub energy_decay: EnergyDecayConfig,
    /// The impulse response starts where its power first rises to this much below its peak.
    /// ISO 3382-1 says at most 20 dB.
    pub start_threshold_db: f64,
}

impl Default for RoomParametersConfig {
    fn default() -> Self {
        Self {
            energy_decay: EnergyDecayConfig::default(),
            start_threshold_db: 20.0,
        }
    }
}

impl RoomParametersConfig {
    pub fn with_energy_decay(mut self, energy_decay: EnergyDecayConfig) -> Self {
        self.energy_decay = energy_decay;
        self
    }

    pub fn with_start_threshold_db(mut self, start_threshold_db: f64) -> Self {
        self.start_threshold_db = start_threshold_db;
        self
    }
}

/// A decay time from a straight line fitted to part of the energy decay curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayTime {
    /// The time the fitted line takes to fall by 60 dB.
    pub seconds: f64,
    /// The correlation coefficient of the fit, close to -1 if the decay is a straight line in
    /// dB. Curved decays (e.g. coupled rooms) give values further from -1.
    pub correlation: f64,
}

/// The ISO 3382-1 parameters of one impulse response.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomParameters {
    /// Early decay time: fitted from 0 to -10 dB.
    pub edt: Option<DecayTime>,
    /// Reverberation time fitted from -5 to -25 dB. `None` if the dynamic range is below
    /// 35 dB.
    pub t20: Option<DecayTime>,
    /// Reverberation time fitted from -5 to -35 dB. `None` if the dynamic range is below
    /// 45 dB.
    pub t30: Option<DecayTime>,
    /// Clarity: the energy in the first 50 ms over the rest, in dB. Used for speech.
    pub c50: f64,
    /// Clarity: the energy in the first 80 ms over the rest, in dB. Used for music.
    pub c80: f64,
    /// Definition: the energy in the first 50 ms over the total, from 0 to 1.
    pub d50: f64,
    /// The sample where the impulse response starts; times count from here.
    pub start: usize,
    /// The mean power of the first [`block_duration`](EnergyDecayConfig::block_duration)
    /// after the start over the noise floor, in dB. ISO 3382-1 wants it at least 10 dB above
    /// the lowest level of a fit: 35 dB for T20, 45 dB for T30.
    pub dynamic_range_db: f64,
}

/// The ISO 3382-1 parameters of every channel of `impulse_response`, one entry per channel.
///
/// Each channel starts where its power first rises to
/// [`start_threshold_db`](RoomParametersConfig::start_threshold_db) below its peak, and its
/// energy decay curve comes from [`energy_decay_curve`]. The decay times are least-squares fits
/// of that curve in dB. Clarity and definition compare the energy before and after 50 or 80 ms,
/// up to where the curve stops at the noise floor.
///
/// Unlike the T20 and T30, the EDT, C50, C80 and D50 are always given, but are only as good as
/// the dynamic range lets them be.
pub fn room_parameters(
    impulse_response: &TimeSignal,
    config: &RoomParametersConfig,
) -> Result<Vec<RoomParameters>, RoomAcousticsError> {
    if impulse_response.num_time_steps() == 0 || impulse_response.num_channels() == 0 {
        return Err(RoomAcousticsError::EmptySignal);
    }
    let sample_rate = impulse_response.sample_rate();
    let block_len = config.energy_decay.validate(sample_rate)?;
    let starts = find_impulse_response_start(impulse_response, Some(config.start_threshold_db))?;

    impulse_response
        .channel_iter()
        .zip(starts)
        .enumerate()
        .map(|(channel, (ir, start))| {
            let ir = TimeSignal::new(
                ir.slice(s![start..])
                    .insert_axis(ndarray::Axis(0))
                    .to_owned(),
                sample_rate,
            )?;
            let decay = energy_decay_curve(&ir, &config.energy_decay).map_err(|e| match e {
                RoomAcousticsError::SilentChannel { .. } => {
                    RoomAcousticsError::SilentChannel { channel }
                }
                e => e,
            })?;
            Ok(parameters_from_decay(&ir, &decay, start, block_len))
        })
        .collect()
}

/// The ISO 3382-1 parameters of `impulse_response` in every band of `bands`, from the lowest
/// band to the highest, one entry per channel in each.
///
/// Each band is filtered with `bands` (time-aligned with the input, so the impulse response
/// keeps its length) and then measured like [`room_parameters`] does. The first band of an
/// [`OctaveBands`] also takes everything below it and the last band everything above it.
///
/// Short decays in low bands are lengthened by the band filter's own decay: keep the filter
/// shorter than the decay, or treat those bands with care.
pub fn room_parameters_in_bands(
    impulse_response: &TimeSignal,
    bands: &OctaveBands,
    config: &RoomParametersConfig,
) -> Result<Vec<BandRoomParameters>, RoomAcousticsError> {
    let bands_rate = bands.impulse_responses.sample_rate();
    if bands_rate != impulse_response.sample_rate() {
        return Err(RoomAcousticsError::SampleRateMismatch {
            signal: impulse_response.sample_rate(),
            bands: bands_rate,
        });
    }

    let len = impulse_response.num_time_steps();
    let delay = bands.impulse_responses.num_time_steps() / 2;
    bands
        .apply(impulse_response)?
        .into_iter()
        .zip(&bands.center_frequencies)
        .map(|(band, &center_frequency)| {
            let band = TimeSignal::new(
                band.time_data()
                    .slice(s![.., delay..delay + len])
                    .to_owned(),
                band.sample_rate(),
            )?;
            let channels =
                room_parameters(&band, config).map_err(|e| RoomAcousticsError::Band {
                    center_frequency,
                    source: Box::new(e),
                })?;
            Ok(BandRoomParameters {
                center_frequency,
                channels,
            })
        })
        .collect()
}

/// The parameters of one band, from [`room_parameters_in_bands`].
#[derive(Debug, Clone, PartialEq)]
pub struct BandRoomParameters {
    /// The band's center frequency in Hz.
    pub center_frequency: f64,
    /// One entry per channel of the impulse response.
    pub channels: Vec<RoomParameters>,
}

fn parameters_from_decay(
    ir: &TimeSignal,
    decay: &EnergyDecay,
    start: usize,
    block_len: usize,
) -> RoomParameters {
    let sample_rate = ir.sample_rate();
    let curve = decay.curve.channel(0);
    let curve_db: Vec<f64> = curve.iter().map(|e| power_to_db(*e)).collect();

    let first_block = ir.channel(0);
    let first_block = first_block.slice(s![..block_len.min(first_block.len())]);
    let initial_power = first_block.iter().map(|x| x * x).sum::<f64>() / first_block.len() as f64;
    // `noise_floor_db` is relative to the peak power.
    let peak_power = ir.iter().fold(0.0_f64, |max, x| max.max(x * x));
    let noise_power = peak_power * 10.0_f64.powf(decay.noise_floor_db[0] / 10.0);
    let dynamic_range_db = power_to_db(initial_power / noise_power);

    let fit = |upper: f64, lower: f64| fit_decay(&curve_db, upper, lower, sample_rate);
    let needs = |lower: f64| dynamic_range_db >= -lower + 10.0;

    // The fraction of the energy after `time`.
    let late = |time: f64| {
        let index = (time * sample_rate).round() as usize;
        curve.get(index).copied().unwrap_or(0.0)
    };
    let clarity = |time: f64| {
        let late = late(time);
        power_to_db((1.0 - late) / late)
    };

    RoomParameters {
        edt: fit(0.0, -10.0),
        t20: if needs(-25.0) { fit(-5.0, -25.0) } else { None },
        t30: if needs(-35.0) { fit(-5.0, -35.0) } else { None },
        c50: clarity(0.05),
        c80: clarity(0.08),
        d50: 1.0 - late(0.05),
        start,
        dynamic_range_db,
    }
}

/// A least-squares line through the curve (in dB, falling) from where it first reaches
/// `upper` to where it last is at or above `lower`, as the time for 60 dB.
fn fit_decay(curve_db: &[f64], upper: f64, lower: f64, sample_rate: f64) -> Option<DecayTime> {
    let first = curve_db.iter().position(|l| *l <= upper)?;
    let last = curve_db.iter().rposition(|l| *l >= lower)?;
    // The curve must go below `lower`, or the range isn't covered.
    if last + 1 >= curve_db.len() || curve_db[last + 1] == f64::NEG_INFINITY || last <= first {
        return None;
    }

    let points = &curve_db[first..=last];
    let count = points.len() as f64;
    let mean_n = (first + last) as f64 / 2.0;
    let mean_l = points.iter().sum::<f64>() / count;
    let (mut cov, mut var_n, mut var_l) = (0.0, 0.0, 0.0);
    for (n, l) in (first..=last).zip(points) {
        let dn = n as f64 - mean_n;
        let dl = l - mean_l;
        cov += dn * dl;
        var_n += dn * dn;
        var_l += dl * dl;
    }
    let slope = cov / var_n * sample_rate; // dB per second
    if slope >= 0.0 {
        return None;
    }
    Some(DecayTime {
        seconds: -60.0 / slope,
        correlation: cov / (var_n * var_l).sqrt(),
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
        delayed_decaying_noise(t60, duration, noise_floor_db, 0.0, seed)
    }

    /// Like [`decaying_noise`], but the decay only starts after `delay` seconds of the noise
    /// floor (or silence).
    pub(crate) fn delayed_decaying_noise(
        t60: f64,
        duration: f64,
        noise_floor_db: Option<f64>,
        delay: f64,
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
            let t = n as f64 / SAMPLE_RATE - delay;
            let envelope = if t < 0.0 {
                0.0
            } else {
                10.0_f64.powf(-3.0 * t / t60)
            };
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

    /// The energy decay rate in 1/s of an exponential decay with this T60.
    fn decay_rate(t60: f64) -> f64 {
        6.0 * std::f64::consts::LN_10 / t60
    }

    fn expected_clarity(t60: f64, time: f64) -> f64 {
        power_to_db((decay_rate(t60) * time).exp() - 1.0)
    }

    fn expected_definition(t60: f64, time: f64) -> f64 {
        1.0 - (-decay_rate(t60) * time).exp()
    }

    fn assert_decay_time(decay: Option<DecayTime>, t60: f64, tolerance: f64) {
        let decay = decay.expect("decay time");
        assert_abs_diff_eq!(decay.seconds, t60, epsilon = tolerance * t60);
        assert!(decay.correlation < -0.99, "{}", decay.correlation);
    }

    #[test]
    fn parameters_of_an_exponential_decay() {
        let t60 = 0.5;
        let delay = 0.01;
        let ir = delayed_decaying_noise(t60, 1.5, Some(-60.0), delay, 9);
        let [parameters] = room_parameters(&ir, &RoomParametersConfig::default())
            .unwrap()
            .try_into()
            .unwrap();

        let delay_samples = (delay * SAMPLE_RATE) as usize;
        assert!(
            (delay_samples - 5..=delay_samples + 5).contains(&parameters.start),
            "{}",
            parameters.start
        );
        assert_abs_diff_eq!(parameters.dynamic_range_db, 60.0, epsilon = 1.5);

        assert_decay_time(parameters.edt, t60, 0.05);
        assert_decay_time(parameters.t20, t60, 0.03);
        assert_decay_time(parameters.t30, t60, 0.03);
        assert_abs_diff_eq!(parameters.c50, expected_clarity(t60, 0.05), epsilon = 0.3);
        assert_abs_diff_eq!(parameters.c80, expected_clarity(t60, 0.08), epsilon = 0.3);
        assert_abs_diff_eq!(
            parameters.d50,
            expected_definition(t60, 0.05),
            epsilon = 0.02
        );
    }

    #[test]
    fn decay_times_need_enough_dynamic_range() {
        let t60 = 0.4;
        let ir = decaying_noise(t60, 1.0, Some(-40.0), 10);
        let [parameters] = room_parameters(&ir, &RoomParametersConfig::default())
            .unwrap()
            .try_into()
            .unwrap();
        assert_abs_diff_eq!(parameters.dynamic_range_db, 40.0, epsilon = 1.5);
        assert_decay_time(parameters.t20, t60, 0.05);
        assert_eq!(parameters.t30, None);
    }

    #[test]
    fn one_entry_per_channel() {
        let short = decaying_noise(0.3, 1.0, Some(-60.0), 11);
        let long = delayed_decaying_noise(0.6, 1.0, Some(-60.0), 0.02, 13);
        let mut data = Array2::zeros((2, short.num_time_steps()));
        data.row_mut(0).assign(&short.channel(0));
        data.row_mut(1).assign(&long.channel(0));
        let ir = TimeSignal::new(data, SAMPLE_RATE).unwrap();

        let parameters = room_parameters(&ir, &RoomParametersConfig::default()).unwrap();
        assert_eq!(parameters.len(), 2);
        assert_decay_time(parameters[0].t20, 0.3, 0.05);
        assert_decay_time(parameters[1].t20, 0.6, 0.05);
        assert!(parameters[1].start > parameters[0].start);
        assert!(parameters[0].c80 > parameters[1].c80);
    }

    #[test]
    fn parameters_per_band() {
        use crate::dsp::filter_bank::OctaveBandsConfig;

        let bands = OctaveBands::design(
            1,
            &OctaveBandsConfig {
                frequency_range: (125.0, 4000.0),
                ..Default::default()
            },
            SAMPLE_RATE,
        )
        .unwrap();
        // The T60 falls from 1 s in the lowest band to 0.5 s in the highest.
        let num_bands = bands.center_frequencies.len();
        let t60 = |band: usize| 1.0 - 0.5 * band as f64 / (num_bands - 1) as f64;

        // Band-limited noise, each band decaying with its own T60.
        let len = (2.0 * SAMPLE_RATE) as usize;
        let noise = generate_noise(
            len,
            NoiseConfig {
                sample_rate: SAMPLE_RATE,
                seed: Some(14),
                ..Default::default()
            },
        )
        .unwrap();
        let delay = bands.impulse_responses.num_time_steps() / 2;
        let mut data = Array2::zeros((1, len));
        for (band, filtered) in bands.apply(&noise).unwrap().iter().enumerate() {
            for n in 0..len {
                let envelope = 10.0_f64.powf(-3.0 * n as f64 / SAMPLE_RATE / t60(band));
                data[[0, n]] += filtered.channel(0)[n + delay] * envelope;
            }
        }
        let ir = TimeSignal::new(data, SAMPLE_RATE).unwrap();

        let result =
            room_parameters_in_bands(&ir, &bands, &RoomParametersConfig::default()).unwrap();
        assert_eq!(result.len(), num_bands);
        for (band, parameters) in result.iter().enumerate() {
            assert_abs_diff_eq!(parameters.center_frequency, bands.center_frequencies[band]);
            // The first and last band also hold everything below and above.
            assert_decay_time(parameters.channels[0].t30, t60(band), 0.05);
        }
    }

    #[test]
    fn bands_must_match_the_sample_rate() {
        let bands = OctaveBands::design(1, &Default::default(), 44_100.0).unwrap();
        let ir = decaying_noise(0.5, 1.0, None, 15);
        assert!(matches!(
            room_parameters_in_bands(&ir, &bands, &RoomParametersConfig::default()).unwrap_err(),
            RoomAcousticsError::SampleRateMismatch { .. }
        ));
    }
}
