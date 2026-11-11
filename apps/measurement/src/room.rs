//! The room acoustics of a sweep measurement: the ISO 3382-1 parameters, broadband and per
//! octave band.

use audio_lab::dsp::filter_bank::{OctaveBands, OctaveBandsConfig};
use audio_lab::dsp::{
    RoomParameters, RoomParametersConfig, room_parameters, room_parameters_in_bands,
};
use audio_lab::signal::TimeSignal;
use audio_lab::signal::ndarray::{Axis, s};

/// The octave bands, by their nominal center frequencies. The lowest one also takes
/// everything below it, the highest everything above.
pub const BANDS: [&str; 8] = ["63", "125", "250", "500", "1k", "2k", "4k", "8k"];

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

/// What the room acoustics tab shows.
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

    pub fn description(self) -> &'static str {
        match self {
            Parameter::T20 => {
                "Reverberation time: how long the sound takes to fall by 60 dB, from the decay \
                 between -5 and -25 dB. Needs 35 dB above the noise."
            }
            Parameter::T30 => {
                "Reverberation time: how long the sound takes to fall by 60 dB, from the decay \
                 between -5 and -35 dB. Needs 45 dB above the noise."
            }
            Parameter::Edt => {
                "Early decay time: the reverberation time from the first 10 dB of the decay, \
                 closer to how reverberant the room sounds."
            }
            Parameter::C50 => {
                "Clarity for speech: the energy in the first 50 ms over the rest. Higher is \
                 clearer."
            }
            Parameter::C80 => {
                "Clarity for music: the energy in the first 80 ms over the rest. Higher is \
                 clearer."
            }
            Parameter::D50 => "Definition: the share of the energy in the first 50 ms.",
        }
    }

    /// The value of `parameters`, or `None` where it couldn't be measured.
    pub fn format(self, parameters: &RoomParameters) -> Option<String> {
        let seconds = |time: Option<audio_lab::dsp::DecayTime>| {
            time.map(|time| format!("{:.2} s", time.seconds))
        };
        let db = |value: f64| value.is_finite().then(|| format!("{value:.1} dB"));
        match self {
            Parameter::T20 => seconds(parameters.t20),
            Parameter::T30 => seconds(parameters.t30),
            Parameter::Edt => seconds(parameters.edt),
            Parameter::C50 => db(parameters.c50),
            Parameter::C80 => db(parameters.c80),
            Parameter::D50 => Some(format!("{:.0} %", parameters.d50 * 100.0)),
        }
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
        let broadband = acoustics.broadband.unwrap();
        // the reverb falls by 60 dB in about 0.35 s
        let t30 = broadband.t30.unwrap().seconds;
        assert!((0.3..0.4).contains(&t30), "{t30}");
        assert_eq!(acoustics.bands.len(), BANDS.len());
        let t20 = acoustics.bands[4].as_ref().and_then(|p| p.t20).unwrap();
        assert!((0.3..0.4).contains(&t20.seconds), "{}", t20.seconds);
        assert_eq!(
            Parameter::T30.format(&broadband),
            Some(format!("{t30:.2} s"))
        );
    }
}
