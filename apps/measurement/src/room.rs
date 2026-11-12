//! The room acoustics of a sweep measurement: the ISO 3382-1 parameters, broadband and per
//! octave band.

use audio_lab::dsp::filter_bank::{OctaveBands, OctaveBandsConfig};
use audio_lab::dsp::{
    RoomParameters, RoomParametersConfig, room_parameters, room_parameters_in_bands,
};
use audio_lab::signal::TimeSignal;
use audio_lab::signal::ndarray::{Axis, s};

/// The center frequencies of the octave bands in Hz, 63 Hz to 8 kHz. The lowest band also
/// takes everything below it, the highest everything above.
pub const BANDS: [f64; 8] = [62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

/// The parameters of one response.
#[derive(Clone, Debug)]
pub struct RoomAcoustics {
    /// `None` if they couldn't be measured, e.g. no decay above the noise.
    pub broadband: Option<RoomParameters>,
    /// One per band of [`BANDS`].
    pub bands: Vec<Option<RoomParameters>>,
}

/// The octave band filters for [`analyze`], designed once per measurement.
pub struct Filters(Vec<OctaveBands>);

impl Filters {
    pub fn new(sample_rate: f64) -> Option<Self> {
        let config = OctaveBandsConfig {
            frequency_range: (60.0, 8500.0),
            ..Default::default()
        };
        let all = OctaveBands::design(1, &config, sample_rate).ok()?;
        // one filter bank per band, so a band that can't be measured doesn't take the
        // others with it
        let bands = all
            .center_frequencies
            .iter()
            .enumerate()
            .map(|(index, &center)| OctaveBands {
                impulse_responses: TimeSignal::new(
                    all.impulse_responses
                        .time_data()
                        .slice(s![index..=index, ..])
                        .to_owned(),
                    sample_rate,
                )
                .expect("valid sample rate"),
                center_frequencies: vec![center],
            })
            .collect::<Vec<_>>();
        (bands.len() == BANDS.len()).then_some(Self(bands))
    }
}

/// The parameters of `impulse_response`: one channel, from its start to where it has decayed
/// into the noise, without a window, so the noise floor can be found.
pub fn analyze(impulse_response: &TimeSignal, filters: &Filters) -> RoomAcoustics {
    let config = RoomParametersConfig::default();
    let first = |parameters: Vec<RoomParameters>| parameters.into_iter().next();
    RoomAcoustics {
        broadband: room_parameters(impulse_response, &config)
            .ok()
            .and_then(first),
        bands: filters
            .0
            .iter()
            .map(|band| {
                room_parameters_in_bands(impulse_response, band, &config)
                    .ok()
                    .and_then(|bands| bands.into_iter().next())
                    .and_then(|band| first(band.channels))
            })
            .collect(),
    }
}

/// The part of the deconvolved `impulse_response` (one channel) that holds the room's decay:
/// `len` samples from `start`. With a sweep of length `T` and a recording of `T + tail`, taking
/// `tail` samples keeps out the harmonic distortion, which the deconvolution puts just before
/// the impulse response and so, wrapped around, at the end.
pub fn decay_part(impulse_response: &TimeSignal, start: usize, len: usize) -> TimeSignal {
    let channel = impulse_response.channel(0);
    let end = (start + len).min(channel.len());
    TimeSignal::new(
        channel
            .slice(s![start..end])
            .to_owned()
            .insert_axis(Axis(0)),
        impulse_response.sample_rate(),
    )
    .expect("valid sample rate")
}

/// What the room acoustics plot shows.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Parameter {
    T20,
    T30,
    Edt,
    C50,
    C80,
    D50,
}

impl Parameter {
    pub const ALL: [Parameter; 6] = [
        Parameter::T20,
        Parameter::T30,
        Parameter::Edt,
        Parameter::C50,
        Parameter::C80,
        Parameter::D50,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Parameter::T20 => "T20",
            Parameter::T30 => "T30",
            Parameter::Edt => "EDT",
            Parameter::C50 => "C50",
            Parameter::C80 => "C80",
            Parameter::D50 => "D50",
        }
    }

    /// The name written out.
    pub fn name(self) -> &'static str {
        match self {
            Parameter::T20 => "Reverberation time, from 20 dB of decay",
            Parameter::T30 => "Reverberation time, from 30 dB of decay",
            Parameter::Edt => "Early decay time",
            Parameter::C50 => "Clarity for speech",
            Parameter::C80 => "Clarity for music",
            Parameter::D50 => "Definition",
        }
    }

    /// What the plot shows.
    pub fn description(self) -> &'static str {
        match self {
            Parameter::T20 => {
                "How long the sound takes to fall by 60 dB after the source stops, per octave \
                 band. Fitted to the decay from -5 to -25 dB and extrapolated to 60 dB."
            }
            Parameter::T30 => {
                "How long the sound takes to fall by 60 dB after the source stops, per octave \
                 band. Fitted to the decay from -5 to -35 dB and extrapolated to 60 dB."
            }
            Parameter::Edt => {
                "The reverberation time from the first 10 dB of the decay, extrapolated to \
                 60 dB. The early decay is what the ear mostly hears, so EDT follows how \
                 reverberant the room sounds at this position."
            }
            Parameter::C50 => {
                "The energy arriving in the first 50 ms over the energy arriving later, in dB. \
                 Early sound helps understanding speech, late sound blurs it. Higher is clearer."
            }
            Parameter::C80 => {
                "The energy arriving in the first 80 ms over the energy arriving later, in dB. \
                 Music tolerates more reverb than speech, hence the longer window. Higher is \
                 clearer, lower fuller and more reverberant."
            }
            Parameter::D50 => {
                "The share of all the energy that arrives in the first 50 ms, in percent. The \
                 same information as C50 on a 0 to 100 % scale."
            }
        }
    }

    /// When to look at this parameter rather than the others.
    pub fn when_to_use(self) -> &'static str {
        match self {
            Parameter::T20 => {
                "Use when T30 is missing or noisy: T20 needs only 35 dB of decay above the noise \
                 floor (T30 needs 45 dB), so it works in noisy rooms and at low and high \
                 frequencies. If the decay is straight, T20 and T30 agree."
            }
            Parameter::T30 => {
                "The standard reverberation time: use it to compare a room against targets or \
                 other rooms, e.g. about 0.2 to 0.4 s for control rooms, 0.4 to 0.8 s for \
                 classrooms and offices, 1.5 to 2.2 s for concert halls. Needs 45 dB of decay \
                 above the noise; switch to T20 where bands are missing."
            }
            Parameter::Edt => {
                "Use to judge the perceived reverb, or to compare listening positions: unlike \
                 T20 and T30 it changes across the room. If EDT is much shorter than T30, strong \
                 early reflections dominate; if longer, the decay is curved, e.g. coupled rooms."
            }
            Parameter::C50 => {
                "Use for rooms for speech: classrooms, meeting rooms, lecture halls, \
                 conference calls. Above 0 dB speech is clear, +3 dB and more is good; below \
                 -2 dB it gets hard to follow."
            }
            Parameter::C80 => {
                "Use for rooms for music: concert halls, rehearsal and listening rooms. About \
                 -2 to +2 dB suits orchestral music, higher values suit fast, detailed music \
                 and amplified sound."
            }
            Parameter::D50 => {
                "Use for speech like C50, when a percentage is easier to read or report. Above \
                 50 % is good for speech, i.e. C50 above 0 dB."
            }
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            Parameter::T20 | Parameter::T30 | Parameter::Edt => "s",
            Parameter::C50 | Parameter::C80 => "dB",
            Parameter::D50 => "%",
        }
    }

    /// The value of `parameters` in [`unit`](Self::unit), or `None` where it couldn't be
    /// measured.
    pub fn value(self, parameters: &RoomParameters) -> Option<f64> {
        let seconds = |time: Option<audio_lab::dsp::DecayTime>| time.map(|time| time.seconds);
        let value = match self {
            Parameter::T20 => seconds(parameters.t20)?,
            Parameter::T30 => seconds(parameters.t30)?,
            Parameter::Edt => seconds(parameters.edt)?,
            Parameter::C50 => parameters.c50,
            Parameter::C80 => parameters.c80,
            Parameter::D50 => parameters.d50 * 100.0,
        };
        value.is_finite().then_some(value)
    }

    /// A value in [`unit`](Self::unit), for reading out.
    pub fn format(self, value: f64) -> String {
        match self {
            Parameter::T20 | Parameter::T30 | Parameter::Edt => format!("{value:.2} s"),
            Parameter::C50 | Parameter::C80 => format!("{value:.1} dB"),
            Parameter::D50 => format!("{value:.0} %"),
        }
    }

    /// The values a plot shows without data.
    pub fn default_range(self) -> (f64, f64) {
        match self {
            Parameter::T20 | Parameter::T30 | Parameter::Edt => (0.0, 1.0),
            Parameter::C50 | Parameter::C80 => (-10.0, 20.0),
            Parameter::D50 => (0.0, 100.0),
        }
    }

    /// The values a plot shows to fit data from `low` to `high`: times and shares from 0,
    /// clarity around its values, with room to spare and on round numbers.
    pub fn fitted_range(self, low: f64, high: f64) -> (f64, f64) {
        match self {
            Parameter::T20 | Parameter::T30 | Parameter::Edt => {
                (0.0, (high * 1.2 / 0.1).ceil().max(1.0) * 0.1)
            }
            Parameter::C50 | Parameter::C80 => (
                (low / 5.0).floor() * 5.0 - 5.0,
                (high / 5.0).ceil() * 5.0 + 5.0,
            ),
            Parameter::D50 => (0.0, 100.0),
        }
    }

    /// The points to plot for `acoustics`, at the band centers in log10 Hz. A band without a
    /// value is NaN, which leaves a gap.
    pub fn points(self, acoustics: &RoomAcoustics) -> Vec<(f64, f64)> {
        BANDS
            .iter()
            .zip(&acoustics.bands)
            .map(|(hz, band)| {
                let value = band.as_ref().and_then(|p| self.value(p));
                (hz.log10(), value.unwrap_or(f64::NAN))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::{Lcg, simulated_room};

    #[test]
    fn the_simulated_room_has_its_reverb() {
        let sample_rate = 48_000.0;
        let filters = Filters::new(sample_rate).unwrap();
        // the simulated room, with a little noise after it like a measurement has
        let mut noise = Lcg(1);
        let room = simulated_room(0, 0, sample_rate, &mut noise);
        let len = room.num_time_steps() * 3;
        let mut data = audio_lab::signal::ndarray::Array2::zeros((1, len));
        for (n, sample) in data.row_mut(0).iter_mut().enumerate() {
            *sample = room.channel(0).get(n).copied().unwrap_or(0.0) + 1e-7 * noise.next();
        }
        let ir = TimeSignal::new(data, sample_rate).unwrap();

        let acoustics = analyze(&ir, &filters);
        let broadband = acoustics.broadband.clone().unwrap();
        // the reverb falls by 60 dB in about 0.35 s
        let t30 = broadband.t30.unwrap().seconds;
        assert!((0.3..0.4).contains(&t30), "{t30}");
        assert_eq!(acoustics.bands.len(), BANDS.len());
        let t20 = acoustics.bands[4].as_ref().and_then(|p| p.t20).unwrap();
        assert!((0.3..0.4).contains(&t20.seconds), "{}", t20.seconds);
        assert_eq!(Parameter::T30.value(&broadband), Some(t30));
        let points = Parameter::T30.points(&acoustics);
        assert_eq!(points.len(), BANDS.len());
        assert!((points[4].0 - 3.0).abs() < 1e-12);
        assert!(
            (points[4].1 - acoustics.bands[4].as_ref().unwrap().t30.unwrap().seconds).abs() < 1e-12
        );
    }
}
