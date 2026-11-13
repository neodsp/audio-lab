//! Averaging measurements: by power, in dB, or as vectors.

use std::f64::consts::TAU;

use audio_lab::dsp::{DecayTime, RoomParameters};
use audio_lab::signal::ndarray::Array2;
use audio_lab::signal::{FreqSignal, TimeSignal};
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::measure::{Origin, Response, peak_index, response_without_delay};
use crate::room::{BANDS, Parameter, RoomAcoustics};

/// How far below the loudest bin the averaged magnitude may fall, in dB: deeper notches would
/// make the minimum phase numerically meaningless.
const FLOOR_DB: f64 = 200.0;

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// The mean of |H|², in dB.
    Power,
    /// The mean of the levels in dB.
    Decibel,
    /// The mean of the complex responses, aligned at their peaks.
    Vector,
}

impl Method {
    pub const ALL: [Method; 3] = [Method::Power, Method::Decibel, Method::Vector];

    pub fn label(self) -> &'static str {
        match self {
            Method::Power => "Power average (RMS)",
            Method::Decibel => "dB average",
            Method::Vector => "Vector average",
        }
    }

    /// What it does.
    pub fn description(self) -> &'static str {
        match self {
            Method::Power => {
                "Averages the energy, |H|², at each frequency and turns the result into dB. Loud \
                 positions weigh more: a dip at one position barely lowers the average, as long \
                 as the others have the energy there. The result is a magnitude only: an \
                 average over positions has no phase or impulse response of its own."
            }
            Method::Decibel => {
                "Averages the levels in dB at each frequency. Peaks and dips count alike, so a \
                 deep dip at one position pulls the average down. Never above the power \
                 average. The result is a magnitude only, like the power average's."
            }
            Method::Vector => {
                "Averages the complex responses, magnitude and phase, after aligning their \
                 impulse response peaks. What the measurements share adds up; what differs \
                 between them, like noise, averages out. Keeps the real phase and impulse \
                 response."
            }
        }
    }

    /// When to choose it.
    pub fn when_to_use(self) -> &'static str {
        match self {
            Method::Power => {
                "Use for several mic positions around a listening area: it is the sound energy \
                 over that area, what rooms are compared by and the usual basis for room EQ, \
                 since it doesn't show cancellations at single positions as deep dips to boost."
            }
            Method::Decibel => {
                "Use to see the typical position rather than the total energy. Compare it with \
                 the power average: where it lies clearly below, the positions disagree, \
                 usually because of a cancellation that moves with the position."
            }
            Method::Vector => {
                "Use for repeated measurements at the same position, to lower noise and \
                 distortion. Not for different positions: their reflections arrive at \
                 different times and cancel, so the average comes out too low, most of all at \
                 high frequencies."
            }
        }
    }

    /// What the result has besides its magnitude, for its info.
    pub fn phase_note(self) -> &'static str {
        match self {
            Method::Power | Method::Decibel => "Magnitude only: no phase or impulse response",
            Method::Vector => "Aligned at their impulse response peaks",
        }
    }
}

/// The `method` average of `responses`, at least two, measured at the same sample rate. Named
/// after the method, with the room acoustics averaged too.
pub fn average(responses: &[Response], method: Method) -> Result<Response, String> {
    let first = responses
        .first()
        .filter(|_| responses.len() >= 2)
        .ok_or("Choose at least two measurements to average.")?;
    let sample_rate = first.impulse_response.sample_rate();
    if responses
        .iter()
        .any(|r| (r.impulse_response.sample_rate() - sample_rate).abs() > 1e-6)
    {
        return Err("The measurements were taken at different sample rates.".into());
    }
    let longest = responses
        .iter()
        .map(|r| r.impulse_response.num_time_steps())
        .max()
        .unwrap_or(0);
    if longest == 0 {
        return Err("The measurements have no impulse response.".into());
    }
    if method == Method::Vector && responses.iter().any(|r| !r.origin.has_phase()) {
        return Err(
            "A vector average needs real phases, which power and dB averages don't have.".into(),
        );
    }
    // the delay to take out of the phase, and where the peak is
    let (impulse_response, delay, peak) = match method {
        Method::Power | Method::Decibel => {
            // a minimum phase response, only to store the magnitude in the impulse response
            // like every response does; it isn't shown. It has no delay to take out: its phase
            // counts from its start, which keeps the stored magnitude where it belongs.
            let ir = magnitude_average(responses, method, longest);
            let peak = peak_index(&ir) as f64;
            (ir, 0.0, peak)
        }
        Method::Vector => {
            let (ir, delay) = vector_average(responses, longest);
            (ir, delay, delay)
        }
    };
    // the peak where the measurements have theirs on average, latency included
    let mean_peak = responses
        .iter()
        .map(|r| r.start as f64 + r.delay)
        .sum::<f64>()
        / responses.len() as f64;
    let start = (mean_peak - peak).round().max(0.0) as usize;
    Ok(Response {
        speaker: method.label().to_string(),
        mic: String::new(),
        origin: Origin::Average(method),
        frequency_response: response_without_delay(&impulse_response, delay),
        delay,
        impulse_response,
        start,
        coherence: None,
        peak_db: None,
        room: average_room(responses.iter().filter_map(|r| r.room.as_ref())),
    })
}

/// The power or dB average of the magnitudes, as a minimum phase impulse response.
fn magnitude_average(responses: &[Response], method: Method, longest: usize) -> TimeSignal {
    // long enough that the minimum phase response, computed on this grid, doesn't wrap around
    let len = 4 * longest;
    let sample_rate = responses[0].impulse_response.sample_rate();
    let mut sum = vec![0.0; len / 2 + 1];
    for response in responses {
        let spectrum = padded(&response.impulse_response, len).to_freq();
        for (total, h) in sum.iter_mut().zip(spectrum.channel(0)) {
            *total += match method {
                Method::Power => h.norm_sqr(),
                _ => 20.0 * h.norm().max(1e-300).log10(),
            };
        }
    }
    let n = responses.len() as f64;
    let mut level_db: Vec<f64> = sum
        .iter()
        .map(|total| match method {
            Method::Power => 10.0 * (total / n).max(1e-300).log10(),
            _ => total / n,
        })
        .collect();
    let loudest = level_db.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for db in &mut level_db {
        *db = db.max(loudest - FLOOR_DB);
    }
    let ir = minimum_phase(&level_db, len, sample_rate);
    // the minimum phase response has its energy at the start: twice the longest is plenty
    truncated(&ir, 2 * longest)
}

/// The minimum phase impulse response, `len` samples, with the magnitude `level_db` (one value
/// per bin of a real FFT of `len`), by folding the real cepstrum.
fn minimum_phase(level_db: &[f64], len: usize, sample_rate: f64) -> TimeSignal {
    let log_magnitude: Vec<Complex64> = level_db
        .iter()
        .map(|db| Complex64::new(db / 20.0 * std::f64::consts::LN_10, 0.0))
        .collect();
    let cepstrum = freq(log_magnitude, sample_rate, len).to_time();
    // the causal part, doubled: the cepstrum of the minimum phase response
    let c = cepstrum.channel(0);
    let folded: Vec<f64> = (0..len)
        .map(|i| match i {
            0 => c[0],
            i if i < len / 2 => 2.0 * c[i],
            i if i == len / 2 => c[i],
            _ => 0.0,
        })
        .collect();
    let spectrum = time(folded, sample_rate).to_freq();
    let response: Vec<Complex64> = spectrum.channel(0).iter().map(|c| c.exp()).collect();
    freq(response, sample_rate, len).to_time()
}

/// The mean of the complex responses, each delayed so its peak lands on the latest one: the
/// impulse response and the delay to the peak.
fn vector_average(responses: &[Response], longest: usize) -> (TimeSignal, f64) {
    let sample_rate = responses[0].impulse_response.sample_rate();
    let latest = responses.iter().map(|r| r.delay).fold(0.0, f64::max);
    let earliest = responses
        .iter()
        .map(|r| r.delay)
        .fold(f64::INFINITY, f64::min);
    let spread = (latest - earliest).ceil() as usize;
    // room for the shifts, and as much again, so nothing wraps around
    let len = 2 * (longest + spread + 1);
    let mut sum = vec![Complex64::new(0.0, 0.0); len / 2 + 1];
    for response in responses {
        let spectrum = padded(&response.impulse_response, len).to_freq();
        let shift = latest - response.delay;
        for ((total, h), f) in sum
            .iter_mut()
            .zip(spectrum.channel(0))
            .zip(spectrum.freq_bins())
        {
            *total += h * Complex64::from_polar(1.0, -TAU * f * shift / sample_rate);
        }
    }
    let n = responses.len() as f64;
    let mean = sum.into_iter().map(|h| h / n).collect();
    let ir = freq(mean, sample_rate, len).to_time();
    (truncated(&ir, longest + spread), latest)
}

/// Peak times further apart than this, in ms, aren't repeats at one position.
const REPEAT_SPREAD_MS: f64 = 1.0;

/// How far apart the peaks of `responses` are, in ms, latency included: within a fraction of
/// a millisecond for repeats at one position.
pub fn peak_spread_ms(responses: &[Response]) -> f64 {
    let times = responses
        .iter()
        .map(|r| (r.start as f64 + r.delay) / r.impulse_response.sample_rate() * 1000.0);
    let (low, high) = times.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), t| {
        (low.min(t), high.max(t))
    });
    if low <= high { high - low } else { 0.0 }
}

/// Why `responses` don't look like repeats at one position, which a vector average is for:
/// measured from different speakers or to different mics, or with their peaks far apart. Empty
/// if they do.
pub fn vector_concerns(responses: &[Response]) -> Vec<String> {
    let mut concerns = Vec::new();
    // the speaker and mic of measured ones; imported files and averages don't tell
    let mut pairs: Vec<String> = Vec::new();
    for response in responses.iter().filter(|r| r.origin == Origin::Measured) {
        let pair = format!("{} → {}", response.speaker, response.mic);
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    if pairs.len() > 1 {
        const LISTED: usize = 3;
        let mut listed = pairs[..pairs.len().min(LISTED)].join(", ");
        if pairs.len() > LISTED {
            listed.push_str(&format!(" and {} more", pairs.len() - LISTED));
        }
        concerns.push(format!(
            "They come from different speakers or mics: {listed}."
        ));
    }
    let spread = peak_spread_ms(responses);
    if spread > REPEAT_SPREAD_MS {
        concerns.push(format!("Their peaks are {spread:.1} ms apart."));
    }
    concerns
}

/// What the info of an average of `responses` tells about its room acoustics: that they are the
/// mean of the measurements' values, and where fewer measurements had a value. Nothing if none
/// has room acoustics.
pub fn room_notes(responses: &[Response]) -> Vec<String> {
    let rooms: Vec<&RoomAcoustics> = responses.iter().filter_map(|r| r.room.as_ref()).collect();
    let total = rooms.len();
    if total == 0 {
        return Vec::new();
    }
    let mut notes = vec![if total == 1 {
        "Room acoustics: those of the one measurement that has them".to_string()
    } else {
        format!("Room acoustics: mean of the {total} measurements' values per band")
    }];
    let without = responses.len() - total;
    if without > 0 {
        notes.push(format!(
            "{without} of the {} {} no room acoustics",
            responses.len(),
            if without == 1 { "has" } else { "have" }
        ));
    }
    // the broadband values (no band), then the bands
    let bands: Vec<(String, Option<usize>)> = std::iter::once(("broadband".to_string(), None))
        .chain(
            BANDS
                .iter()
                .enumerate()
                .map(|(i, &hz)| (format_band(hz), Some(i))),
        )
        .collect();
    for parameter in Parameter::ALL {
        // the bands some but not all measurements have a value in, by how many
        let mut partial: Vec<(usize, Vec<&str>)> = Vec::new();
        for (label, band) in &bands {
            let count = rooms
                .iter()
                .filter(|room| {
                    parameters_in(room, *band)
                        .and_then(|p| parameter.value(p))
                        .is_some()
                })
                .count();
            if count == 0 || count == total {
                continue;
            }
            match partial.iter_mut().find(|(c, _)| *c == count) {
                Some((_, labels)) => labels.push(label),
                None => partial.push((count, vec![label])),
            }
        }
        for (count, labels) in partial {
            notes.push(format!(
                "{} at {} from {count} of {total}",
                parameter.label(),
                labels.join(", ")
            ));
        }
    }
    notes
}

/// The parameters of `room` in the octave `band` of [`BANDS`], or broadband for `None`.
fn parameters_in(room: &RoomAcoustics, band: Option<usize>) -> Option<&RoomParameters> {
    match band {
        None => room.broadband.as_ref(),
        Some(index) => room.bands.get(index).and_then(Option::as_ref),
    }
}

/// An octave band's center as the room plot labels it: 63 Hz, 1 kHz.
fn format_band(hz: f64) -> String {
    if hz >= 1000.0 {
        format!("{} kHz", hz / 1000.0)
    } else {
        format!("{} Hz", hz.round())
    }
}

/// The mean of each room acoustics parameter per band, over the `rooms` that have it. `None`
/// without any.
pub fn average_room<'a>(rooms: impl Iterator<Item = &'a RoomAcoustics>) -> Option<RoomAcoustics> {
    let rooms: Vec<&RoomAcoustics> = rooms.collect();
    if rooms.is_empty() {
        return None;
    }
    let bands = rooms.iter().map(|room| room.bands.len()).max().unwrap_or(0);
    Some(RoomAcoustics {
        broadband: average_parameters(rooms.iter().map(|room| room.broadband.as_ref())),
        bands: (0..bands)
            .map(|band| {
                average_parameters(
                    rooms
                        .iter()
                        .map(|room| room.bands.get(band).and_then(Option::as_ref)),
                )
            })
            .collect(),
    })
}

fn average_parameters<'a>(
    parameters: impl Iterator<Item = Option<&'a RoomParameters>>,
) -> Option<RoomParameters> {
    let parameters: Vec<&RoomParameters> = parameters.flatten().collect();
    if parameters.is_empty() {
        return None;
    }
    let mean = |values: Vec<f64>| -> Option<f64> {
        let values: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    let time = |get: fn(&RoomParameters) -> Option<DecayTime>| {
        let times: Vec<DecayTime> = parameters.iter().filter_map(|p| get(p)).collect();
        Some(DecayTime {
            seconds: mean(times.iter().map(|t| t.seconds).collect())?,
            correlation: mean(times.iter().map(|t| t.correlation).collect())?,
        })
    };
    let value = |get: fn(&RoomParameters) -> f64| mean(parameters.iter().map(|p| get(p)).collect());
    Some(RoomParameters {
        edt: time(|p| p.edt),
        t20: time(|p| p.t20),
        t30: time(|p| p.t30),
        c50: value(|p| p.c50).unwrap_or(f64::NAN),
        c80: value(|p| p.c80).unwrap_or(f64::NAN),
        d50: value(|p| p.d50).unwrap_or(f64::NAN),
        start: 0,
        dynamic_range_db: value(|p| p.dynamic_range_db).unwrap_or(f64::NAN),
    })
}

/// `signal` (one channel) with zeros after it up to `len` samples.
fn padded(signal: &TimeSignal, len: usize) -> TimeSignal {
    let mut samples = signal.channel(0).to_vec();
    samples.resize(len, 0.0);
    time(samples, signal.sample_rate())
}

/// The first `len` samples of `signal` (one channel).
fn truncated(signal: &TimeSignal, len: usize) -> TimeSignal {
    let samples = signal.channel(0).iter().take(len).copied().collect();
    time(samples, signal.sample_rate())
}

fn time(samples: Vec<f64>, sample_rate: f64) -> TimeSignal {
    let len = samples.len();
    TimeSignal::new(
        Array2::from_shape_vec((1, len), samples).expect("one row"),
        sample_rate,
    )
    .expect("valid sample rate")
}

fn freq(bins: Vec<Complex64>, sample_rate: f64, num_time_steps: usize) -> FreqSignal {
    let len = bins.len();
    FreqSignal::new(
        Array2::from_shape_vec((1, len), bins).expect("one row"),
        sample_rate,
        Some(num_time_steps),
    )
    .expect("valid sample rate")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::{Lcg, simulated_room};

    const RATE: f64 = 48_000.0;

    /// A response of `ir` scaled by `gain`, delayed by `shift` samples.
    fn response(ir: &TimeSignal, gain: f64, shift: usize) -> Response {
        let mut samples = vec![0.0; shift];
        samples.extend(ir.channel(0).iter().map(|s| s * gain));
        let impulse_response = time(samples, RATE);
        let delay = peak_index(&impulse_response) as f64;
        Response {
            speaker: "Speaker".into(),
            mic: "Mic".into(),
            origin: Origin::Measured,
            frequency_response: response_without_delay(&impulse_response, delay),
            delay,
            impulse_response,
            start: 1000,
            coherence: None,
            peak_db: None,
            room: None,
        }
    }

    fn room_ir() -> TimeSignal {
        let ir = simulated_room(0, 0, RATE, &mut Lcg(1));
        // from shortly before the direct sound, like a measured one
        let start = peak_index(&ir) - 20;
        let samples = ir
            .channel(0)
            .iter()
            .skip(start)
            .take(8192)
            .copied()
            .collect();
        time(samples, RATE)
    }

    /// The mean power of `ir` in the octave around `hz`, in dB, on the grid of a real FFT of
    /// `len` samples: single bins of a reverberant response swing by many dB.
    fn level_at(ir: &TimeSignal, len: usize, hz: f64) -> f64 {
        let spectrum = padded(ir, len).to_freq();
        let band = hz / 2f64.sqrt()..=hz * 2f64.sqrt();
        let powers: Vec<f64> = spectrum
            .freq_bins()
            .iter()
            .zip(spectrum.channel(0))
            .filter(|(f, _)| band.contains(f))
            .map(|(_, h)| h.norm_sqr())
            .collect();
        10.0 * (powers.iter().sum::<f64>() / powers.len() as f64).log10()
    }

    /// How much louder `average` is than `reference` in the octave around `hz`, compared on
    /// one grid.
    fn gain_db(average: &Response, reference: &TimeSignal, hz: f64) -> f64 {
        let len = 4 * average.impulse_response.num_time_steps();
        level_at(&average.impulse_response, len, hz) - level_at(reference, len, hz)
    }

    #[test]
    fn the_methods_weigh_levels_differently() {
        let ir = room_ir();
        let pair = [response(&ir, 2.0, 0), response(&ir, 0.5, 0)];
        for (method, expected_db) in [
            // (4 + 1/4) / 2 in power
            (Method::Power, 10.0 * (4.25f64 / 2.0).log10()),
            // +6 and -6 dB
            (Method::Decibel, 0.0),
            // (2 + 1/2) / 2 in amplitude
            (Method::Vector, 20.0 * 1.25f64.log10()),
        ] {
            let average = average(&pair, method).unwrap();
            for hz in [100.0, 1000.0, 5000.0] {
                let difference = gain_db(&average, &ir, hz);
                assert!(
                    (difference - expected_db).abs() < 0.1,
                    "{method:?} at {hz} Hz: {difference} dB, expected {expected_db}"
                );
            }
        }
    }

    #[test]
    fn the_minimum_phase_response_peaks_at_its_start() {
        let ir = room_ir();
        let pair = [response(&ir, 1.0, 0), response(&ir, 1.0, 37)];
        for method in [Method::Power, Method::Decibel] {
            let average = average(&pair, method).unwrap();
            let peak = peak_index(&average.impulse_response);
            assert!(peak < 5, "{method:?}: {peak}");
            // its phase counts from its start, not from the peak a sample or two in
            assert_eq!(average.delay, 0.0);
            // and the peak stays where the measurements have theirs on average
            let mean_peak = 1000.0 + 20.0 + 37.0 / 2.0;
            let at = (average.start + peak) as f64;
            assert!(
                (at - mean_peak).abs() <= 1.0,
                "{method:?}: {at} {mean_peak}"
            );
        }
    }

    #[test]
    fn power_and_db_averages_keep_their_phase_alike() {
        // two speakers to one mic, like the simulated device measures them: the averages'
        // peaks land on different samples, which mustn't tilt one phase against the other
        let mut settings = crate::measure::Settings {
            simulated: true,
            ..Default::default()
        };
        settings.sync_channels(4, 4);
        settings.outputs[1].enabled = true;
        let responses = crate::measure::run(
            &settings.request(),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap()
        .responses;
        let power = average(&responses, Method::Power).unwrap();
        let db = average(&responses, Method::Decibel).unwrap();
        // the mean difference from 2 to 10 kHz: one sample would be 15 to 75°
        let phase = |response: &Response, hz: f64| {
            let signal = &response.frequency_response;
            let bins = signal.freq_bins();
            let index = bins.iter().position(|&f| f >= hz).unwrap();
            signal.channel(0)[index]
        };
        let differences: Vec<f64> = (0..=40)
            .map(|i| 2000.0 + 200.0 * f64::from(i))
            .map(|hz| (phase(&power, hz) / phase(&db, hz)).arg().to_degrees())
            .collect();
        let mean = differences.iter().sum::<f64>() / differences.len() as f64;
        assert!(mean.abs() < 10.0, "{mean}°: {differences:?}");
    }

    #[test]
    fn the_vector_average_aligns_the_peaks() {
        let ir = room_ir();
        // the same response, once 37 samples later: aligned, they add up fully
        let average = average(
            &[response(&ir, 1.0, 0), response(&ir, 1.0, 37)],
            Method::Vector,
        )
        .unwrap();
        for hz in [100.0, 1000.0, 10_000.0] {
            let difference = gain_db(&average, &ir, hz);
            assert!(difference.abs() < 0.1, "{hz} Hz: {difference} dB");
        }
    }

    #[test]
    fn simulated_measurements_average_with_their_room_acoustics() {
        let mut settings = crate::measure::Settings {
            simulated: true,
            ..Default::default()
        };
        settings.sync_channels(4, 4);
        settings.inputs[1].enabled = true;
        settings.inputs[2].enabled = true;
        let responses = crate::measure::run(
            &settings.request(),
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap()
        .responses;
        assert_eq!(responses.len(), 3);
        for method in Method::ALL {
            let average = average(&responses, method).unwrap();
            assert!(
                average
                    .frequency_response
                    .channel(0)
                    .iter()
                    .all(|h| h.norm().is_finite()),
                "{method:?}"
            );
            // the simulated room's reverb, about 0.35 s, averaged
            let t30 = average.room.as_ref().unwrap().broadband.as_ref().unwrap();
            let t30 = t30.t30.unwrap().seconds;
            assert!((0.3..0.4).contains(&t30), "{method:?}: {t30}");
            // the delays of the mics differ by about a millisecond: the average lies between
            let latest = responses.iter().map(|r| r.start as f64 + r.delay);
            let (low, high) =
                latest.fold((f64::INFINITY, 0.0f64), |(l, h), t| (l.min(t), h.max(t)));
            let peak = average.start as f64 + average.delay;
            assert!(
                low - 2.0 <= peak && peak <= high + 2.0,
                "{method:?}: {peak}"
            );
        }
    }

    #[test]
    fn only_the_vector_average_has_a_phase() {
        let ir = room_ir();
        let pair = [response(&ir, 1.0, 0), response(&ir, 0.5, 0)];
        for method in Method::ALL {
            let average = average(&pair, method).unwrap();
            assert_eq!(average.origin, Origin::Average(method));
            assert_eq!(average.origin.has_phase(), method == Method::Vector);
        }
        // and a vector average can't take the made-up phase of a power average
        let power = average(&pair, Method::Power).unwrap();
        assert!(average(&[power.clone(), pair[0].clone()], Method::Vector).is_err());
        assert!(average(&[power, pair[0].clone()], Method::Power).is_ok());
    }

    #[test]
    fn room_notes_tell_where_fewer_measurements_had_a_value() {
        let parameters = |t30: Option<f64>| RoomParameters {
            edt: None,
            t20: None,
            t30: t30.map(|seconds| DecayTime {
                seconds,
                correlation: -1.0,
            }),
            c50: 0.0,
            c80: 0.0,
            d50: 0.5,
            start: 0,
            dynamic_range_db: 60.0,
        };
        let room = |low: Option<f64>| RoomAcoustics {
            broadband: Some(parameters(Some(0.4))),
            bands: BANDS
                .iter()
                .enumerate()
                .map(|(i, _)| Some(parameters(if i < 2 { low } else { Some(0.4) })))
                .collect(),
        };
        let ir = room_ir();
        let with = |acoustics: Option<RoomAcoustics>| {
            let mut r = response(&ir, 1.0, 0);
            r.room = acoustics;
            r
        };
        // four with room acoustics, two of them without a T30 at 63 and 125 Hz; one live
        let responses = [
            with(Some(room(Some(0.5)))),
            with(Some(room(Some(0.5)))),
            with(Some(room(None))),
            with(Some(room(None))),
            with(None),
        ];
        let notes = room_notes(&responses);
        assert_eq!(
            notes,
            [
                "Room acoustics: mean of the 4 measurements' values per band",
                "1 of the 5 has no room acoustics",
                "T30 at 63 Hz, 125 Hz from 2 of 4",
            ]
        );
        assert!(room_notes(&responses[4..]).is_empty());
    }

    #[test]
    fn a_vector_average_of_other_than_repeats_is_questioned() {
        let ir = room_ir();
        // repeats: same speaker and mic, peaks a few samples apart
        let repeats = [response(&ir, 1.0, 0), response(&ir, 0.9, 5)];
        assert!(vector_concerns(&repeats).is_empty());
        // another speaker, 2.5 ms later
        let mut other = response(&ir, 1.0, 120);
        other.speaker = "Speaker 2".into();
        let concerns = vector_concerns(&[repeats[0].clone(), other]);
        assert_eq!(
            concerns,
            [
                "They come from different speakers or mics: Speaker → Mic, Speaker 2 → Mic.",
                "Their peaks are 2.5 ms apart.",
            ]
        );
        // imported files don't tell their speaker and mic, only their peaks
        let mut imported = response(&ir, 1.0, 0);
        imported.origin = Origin::Imported;
        imported.speaker = "file".into();
        assert!(vector_concerns(&[repeats[0].clone(), imported]).is_empty());
    }

    #[test]
    fn averaging_needs_two_measurements_at_one_rate() {
        let ir = room_ir();
        assert!(average(&[response(&ir, 1.0, 0)], Method::Power).is_err());
        let mut other = response(&ir, 1.0, 0);
        other.impulse_response =
            TimeSignal::new(other.impulse_response.time_data().to_owned(), 44_100.0).unwrap();
        assert!(average(&[response(&ir, 1.0, 0), other], Method::Power).is_err());
    }

    #[test]
    fn room_acoustics_are_averaged_per_band() {
        let parameters = |t30: Option<f64>, c50: f64| RoomParameters {
            edt: None,
            t20: None,
            t30: t30.map(|seconds| DecayTime {
                seconds,
                correlation: -1.0,
            }),
            c50,
            c80: 0.0,
            d50: 0.5,
            start: 0,
            dynamic_range_db: 60.0,
        };
        let rooms = [
            RoomAcoustics {
                broadband: Some(parameters(Some(0.4), 2.0)),
                bands: vec![Some(parameters(Some(0.6), 0.0)), None],
            },
            RoomAcoustics {
                broadband: Some(parameters(None, 4.0)),
                bands: vec![Some(parameters(Some(0.8), 0.0)), None],
            },
        ];
        let average = average_room(rooms.iter()).unwrap();
        let broadband = average.broadband.unwrap();
        // only one has a T30
        assert_eq!(broadband.t30.unwrap().seconds, 0.4);
        assert_eq!(broadband.c50, 3.0);
        assert!((average.bands[0].as_ref().unwrap().t30.unwrap().seconds - 0.7).abs() < 1e-12);
        assert!(average.bands[1].is_none());
        assert!(average_room([].iter()).is_none());
    }
}
