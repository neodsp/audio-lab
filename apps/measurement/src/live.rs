//! Live measurements: looped pink noise on the chosen speakers while the mics record, analyzed
//! several times a second into one response per mic.
//!
//! The same noise plays on every speaker that is on, so each mic's response is that of those
//! speakers together. Speakers can be switched on and off while it runs: alone, a speaker's
//! own response shows; together, how they add up, e.g. a sub and a main speaker around the
//! crossover. Recordings from before a switch are left out of the analysis.

use std::f64::consts::TAU;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use audio_lab::dsp::{
    ConvolveMode, TransferFunctionConfig, coherence, convolve, transfer_function,
};
use audio_lab::signal::audio_blocks::AudioBlockMut;
use audio_lab::signal::measure::{LiveSnapshot, NoiseControls, NoiseSource, NoiseSourceConfig};
use audio_lab::signal::ndarray::{Array2, Axis};
use audio_lab::signal::playback::{DeviceConfig, LiveSession, LiveSessionConfig};
use audio_lab::signal::{FreqSignal, TimeSignal};
use num::complex::Complex64;

use crate::curves::Curves;
use crate::measure::{Lcg, Response, peak_index, simulated_room};

/// How long after switching speakers the recording still holds the old setting: the
/// device latency plus the sound's travel time, generously.
const SETTLE: Duration = Duration::from_millis(250);
/// How long a speaker takes to fade in or out, so switching doesn't click.
const FADE: Duration = Duration::from_millis(20);

/// What to measure live.
#[derive(Clone)]
pub struct LiveRequest {
    /// `None` for the simulated device.
    pub device: Option<DeviceConfig>,
    /// Channel and name of each speaker that can play.
    pub speakers: Vec<(u16, String)>,
    pub mics: Vec<(u16, String)>,
    pub sample_rate: f64,
    pub level_db: f64,
    /// Noise periods averaged per analysis. More are steadier and tell room noise apart
    /// better; fewer follow changes faster.
    pub periods: usize,
}

impl LiveRequest {
    /// The frequency range shown: what the noise covers, up to 20 kHz.
    pub fn range(&self) -> (f64, f64) {
        (20.0, 20_000f64.min(self.sample_rate / 2.0))
    }

    /// How it measures at the level `level_db`, a line each, for the info of a snapshot.
    pub fn describe(&self, level_db: f64) -> Vec<String> {
        let period = period_len(self.sample_rate) as f64 / self.sample_rate;
        vec![
            format!(
                "Live, pink noise at {level_db:.0} dBFS, {} period{} averaged ({:.1} s)",
                self.periods,
                if self.periods == 1 { "" } else { "s" },
                self.periods as f64 * period
            ),
            crate::measure::describe_device(self.device.as_ref(), self.sample_rate),
        ]
    }
}

/// Length of the looped noise and of each analysis segment, as a power of two: about a third
/// of a second, longer than the latency plus a room's reverb.
pub fn period_len(sample_rate: f64) -> usize {
    ((sample_rate / 3.0) as usize).next_power_of_two()
}

pub fn db_to_gain(db: f64) -> f32 {
    10f64.powf(db / 20.0) as f32
}

/// A running live measurement.
pub struct Live {
    pub request: LiveRequest,
    source: Source,
    noise: NoiseControls,
    /// Whether each speaker of the request plays.
    playing: Arc<[AtomicBool]>,
    period: usize,
    /// When the speakers last changed: what was recorded before is of the old setting.
    changed: Instant,
    /// Delay taken out of the phase per mic: the device latency plus the sound's travel time.
    /// Found from the first analysis and on request, then kept, so changes such as switching
    /// speakers show in the phase. Without it the phase would wrap every few Hz.
    pub delays_ms: Option<Vec<f64>>,
}

enum Source {
    Device(Box<LiveSession>),
    Simulated(Box<Simulation>),
}

impl Live {
    /// Starts playing noise on all speakers of `request` and recording.
    pub fn start(request: LiveRequest) -> Result<Self, String> {
        if request.speakers.is_empty() || request.mics.is_empty() {
            return Err("Choose at least one speaker and one microphone.".into());
        }
        let period = period_len(request.sample_rate);
        let outputs: Vec<u16> = request
            .speakers
            .iter()
            .map(|(channel, _)| *channel)
            .collect();
        let mut noise = NoiseSource::new(&NoiseSourceConfig {
            outputs: outputs.clone(),
            period,
            gain: db_to_gain(request.level_db),
            sample_rate: request.sample_rate,
            ..Default::default()
        })
        .map_err(|error| error.to_string())?;
        let controls = noise.controls();
        let playing: Arc<[AtomicBool]> = request
            .speakers
            .iter()
            .map(|_| AtomicBool::new(true))
            .collect();

        let source = match &request.device {
            Some(device) => {
                let mut gates = Gates::new(
                    outputs,
                    Arc::clone(&playing),
                    (FADE.as_secs_f64() * request.sample_rate) as usize,
                );
                let averaged = Duration::from_secs_f64(
                    (request.periods * period) as f64 / request.sample_rate,
                );
                let config = LiveSessionConfig {
                    inputs: request.mics.iter().map(|(channel, _)| *channel).collect(),
                    num_output_channels: noise.num_output_channels(),
                    sample_rate: request.sample_rate,
                    history: averaged + Duration::from_secs(2),
                };
                let session = LiveSession::start(
                    move |output, reference| {
                        // the reference stays the noise itself: the response is relative to it
                        noise.fill(output, reference);
                        gates.apply(output);
                    },
                    &config,
                    device,
                )
                .map_err(|error| error.to_string())?;
                Source::Device(Box::new(session))
            }
            None => Source::Simulated(Box::new(Simulation::new(
                &request,
                &noise,
                controls.clone(),
                Arc::clone(&playing),
            )?)),
        };
        Ok(Self {
            request,
            source,
            noise: controls,
            playing,
            period,
            changed: Instant::now(),
            delays_ms: None,
        })
    }

    pub fn is_playing(&self, speaker: usize) -> bool {
        self.playing
            .get(speaker)
            .is_some_and(|playing| playing.load(Ordering::Relaxed))
    }

    /// Switches the `speaker`th speaker of the request on or off.
    pub fn set_playing(&mut self, speaker: usize, playing: bool) {
        if let Some(flag) = self.playing.get(speaker)
            && flag.swap(playing, Ordering::Relaxed) != playing
        {
            self.changed = Instant::now();
        }
    }

    /// The names of the speakers that play, joined: `Speaker 1 + Speaker 2`.
    pub fn playing_names(&self) -> String {
        let names: Vec<&str> = self
            .request
            .speakers
            .iter()
            .enumerate()
            .filter(|(index, _)| self.is_playing(*index))
            .map(|(_, (_, name))| name.as_str())
            .collect();
        if names.is_empty() {
            "No speaker".into()
        } else {
            names.join(" + ")
        }
    }

    pub fn set_level(&self, db: f64) {
        self.noise.set_gain(db_to_gain(db));
    }

    /// Finds the delays again with the next analysis.
    pub fn find_delays(&mut self) {
        self.delays_ms = None;
    }

    /// The latest recording to analyze: as many whole periods as were recorded since the
    /// speakers last changed, up to the request's. `None` while there isn't one yet.
    /// The curves are computed for each of `smoothings`.
    pub fn take(&mut self, smoothings: Vec<Option<f64>>) -> Option<Recording> {
        let fresh = self.changed.elapsed().saturating_sub(SETTLE).as_secs_f64();
        let fresh_frames = (fresh * self.request.sample_rate) as usize;
        let periods = self.request.periods.min(fresh_frames / self.period);
        if periods == 0 {
            return None;
        }
        let frames = periods * self.period;
        let snapshot = match &mut self.source {
            Source::Device(session) => session.latest(frames),
            Source::Simulated(simulation) => simulation.latest(frames),
        };
        if snapshot.inputs.num_time_steps() < frames {
            return None;
        }
        Some(Recording {
            snapshot,
            period: self.period,
            speakers: self.playing_names(),
            mics: self
                .request
                .mics
                .iter()
                .map(|(_, name)| name.clone())
                .collect(),
            delays_ms: self.delays_ms.clone(),
            range: self.request.range(),
            smoothings,
        })
    }

    /// Problems of the device: dropouts and why it stopped, if it did.
    pub fn status(&self) -> Status {
        match &self.source {
            Source::Device(session) => {
                let status = session.status();
                Status {
                    dropouts: status.xruns,
                    lost_frames: session.dropped_frames(),
                    error: status.error.map(|error| error.to_string()),
                }
            }
            Source::Simulated(_) => Status::default(),
        }
    }

    pub fn stop(self) {
        if let Source::Device(session) = self.source {
            // a device that won't stop cleanly is closed when the session is dropped anyway
            let _ = session.stop();
        }
    }
}

#[derive(Default)]
pub struct Status {
    /// Callbacks where the device lost audio.
    pub dropouts: u64,
    /// Frames lost because the app didn't pick them up in time.
    pub lost_frames: u64,
    pub error: Option<String>,
}

/// A recording to analyze in the background, with what the analysis needs to know.
pub struct Recording {
    snapshot: LiveSnapshot,
    period: usize,
    /// Who plays, for the names.
    speakers: String,
    mics: Vec<String>,
    delays_ms: Option<Vec<f64>>,
    range: (f64, f64),
    /// The smoothings to compute the curves for, one per frequency plot unless they agree.
    smoothings: Vec<Option<f64>>,
}

/// One response per mic, and the delays that were taken out of the phase.
pub struct Analysis {
    pub responses: Vec<Response>,
    /// The curves of each response for each smoothing the recording was taken with, computed
    /// here so the UI thread doesn't have to.
    pub curves: Vec<Vec<Curves>>,
    pub delays_ms: Vec<f64>,
    /// Whether the delays were found in this recording, rather than given.
    pub found_delays: bool,
}

/// The transfer function from the noise to each mic and its coherence. Blocks; run it in the
/// background.
pub fn analyze(recording: Recording) -> Result<Analysis, String> {
    let snapshot = &recording.snapshot;
    let config = TransferFunctionConfig::default().with_segment_len(recording.period);
    let response = transfer_function(&snapshot.inputs, &snapshot.reference, &config)
        .map_err(|error| error.to_string())?;
    let coherence = coherence(&snapshot.inputs, &snapshot.reference, &config)
        .map_err(|error| error.to_string())?;

    let sample_rate = response.sample_rate();
    let mut analysis = Analysis {
        responses: Vec::new(),
        curves: Vec::new(),
        delays_ms: Vec::new(),
        found_delays: recording.delays_ms.is_none(),
    };
    for (index, mic) in recording.mics.iter().enumerate() {
        let single = freq_channel(&response, index);
        // the impulse response, from the start of the noise period: it includes the delay
        let impulse_response = single.to_time();
        let delay_ms = match &recording.delays_ms {
            Some(delays) => delays.get(index).copied().unwrap_or(0.0),
            None => peak_index(&impulse_response) as f64 / sample_rate * 1000.0,
        };
        let peak = snapshot
            .inputs
            .channel(index)
            .iter()
            .fold(1e-12f64, |m, s| m.max(s.abs()));
        let frequency_response = without_delay(&single, delay_ms);
        let coherence = freq_channel(&coherence, index);
        analysis.curves.push(
            recording
                .smoothings
                .iter()
                .map(|&smoothing| {
                    Curves::new(
                        &frequency_response,
                        Some(&coherence),
                        recording.range,
                        smoothing,
                    )
                })
                .collect(),
        );
        analysis.responses.push(Response {
            speaker: recording.speakers.clone(),
            mic: mic.clone(),
            impulse_response,
            start: 0,
            frequency_response,
            delay: delay_ms / 1000.0 * sample_rate,
            coherence: Some(coherence),
            peak_db: Some(20.0 * peak.log10()),
            room: None,
        });
        analysis.delays_ms.push(delay_ms);
    }
    Ok(analysis)
}

/// One channel of `signal` as a signal of its own.
fn freq_channel(signal: &FreqSignal, index: usize) -> FreqSignal {
    let data = signal.channel(index).to_owned().insert_axis(Axis(0));
    FreqSignal::new(data, signal.sample_rate(), Some(signal.num_time_steps()))
        .expect("the sample rate was valid")
}

/// `response` with a delay of `ms` taken out of its phase.
fn without_delay(response: &FreqSignal, ms: f64) -> FreqSignal {
    let mut result = response.clone();
    let freqs = response.freq_bins().to_owned();
    for (h, f) in result.channel_mut(0).iter_mut().zip(freqs) {
        *h *= Complex64::from_polar(1.0, TAU * f * ms / 1000.0);
    }
    result
}

/// Switches speakers on and off on the audio thread, fading so it doesn't click.
struct Gates {
    outputs: Vec<u16>,
    playing: Arc<[AtomicBool]>,
    /// Current gain per output, from 0 to 1.
    gains: Vec<f32>,
    /// Gain change per frame.
    step: f32,
}

impl Gates {
    fn new(outputs: Vec<u16>, playing: Arc<[AtomicBool]>, fade_frames: usize) -> Self {
        Self {
            gains: vec![1.0; outputs.len()],
            outputs,
            playing,
            step: 1.0 / fade_frames.max(1) as f32,
        }
    }

    /// Real-time safe.
    fn apply(&mut self, output: &mut impl AudioBlockMut<f32>) {
        for (index, &channel) in self.outputs.iter().enumerate() {
            let target = if self.playing[index].load(Ordering::Relaxed) {
                1.0
            } else {
                0.0
            };
            let gain = &mut self.gains[index];
            if *gain == 1.0 && target == 1.0 || channel >= output.num_channels() {
                continue;
            }
            for frame in 0..output.num_frames() {
                *gain = if target > *gain {
                    (*gain + self.step).min(target)
                } else {
                    (*gain - self.step).max(target)
                };
                *output.sample_mut(channel, frame) *= *gain;
            }
        }
    }
}

/// A made-up room for trying live measurements without hardware, the one the simulated sweeps
/// measure: every mic hears the speakers that play, and a little background noise.
struct Simulation {
    /// One period of the noise, at full scale.
    noise: Vec<f64>,
    /// What each mic hears of each speaker during one period, in steady state: `[speaker][mic]`.
    heard: Vec<Vec<Vec<f64>>>,
    controls: NoiseControls,
    playing: Arc<[AtomicBool]>,
    sample_rate: f64,
    started: Instant,
    background: Lcg,
}

impl Simulation {
    fn new(
        request: &LiveRequest,
        noise: &NoiseSource,
        controls: NoiseControls,
        playing: Arc<[AtomicBool]>,
    ) -> Result<Self, String> {
        let sample_rate = request.sample_rate;
        let period = noise.period_signal();
        let len = period.num_time_steps();
        // three periods in a row: the middle of the third is steady for responses up to two
        // periods long
        let looped = TimeSignal::new(
            Array2::from_shape_fn((1, 3 * len), |(_, i)| period.channel(0)[i % len]),
            sample_rate,
        )
        .map_err(|error| error.to_string())?;
        let mut room_noise = Lcg(7);
        let heard = request
            .speakers
            .iter()
            .map(|&(speaker, _)| {
                request
                    .mics
                    .iter()
                    .map(|&(mic, _)| {
                        let ir = simulated_room(speaker, mic, sample_rate, &mut room_noise);
                        let heard = convolve(&looped, &ir, ConvolveMode::Full)
                            .map_err(|error| error.to_string())?;
                        Ok(heard
                            .channel(0)
                            .iter()
                            .skip(2 * len)
                            .take(len)
                            .copied()
                            .collect())
                    })
                    .collect::<Result<Vec<Vec<f64>>, String>>()
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            noise: period.channel(0).to_vec(),
            heard,
            controls,
            playing,
            sample_rate,
            started: Instant::now(),
            background: Lcg(1),
        })
    }

    /// Like [`LiveSession::latest`]: the most recent `num_frames`, or what there is so far.
    fn latest(&mut self, num_frames: usize) -> LiveSnapshot {
        let elapsed = (self.started.elapsed().as_secs_f64() * self.sample_rate) as usize;
        let n = num_frames.min(elapsed);
        let len = self.noise.len();
        // where in the loop the snapshot starts
        let start = (elapsed - n) % len;
        let gain = if self.controls.is_enabled() {
            f64::from(self.controls.gain())
        } else {
            0.0
        };
        let playing: Vec<bool> = self
            .playing
            .iter()
            .map(|playing| playing.load(Ordering::Relaxed))
            .collect();
        let num_mics = self.heard.first().map_or(0, Vec::len);
        let mut inputs = Array2::zeros((num_mics, n));
        for (mic, mut row) in inputs.rows_mut().into_iter().enumerate() {
            for (i, sample) in row.iter_mut().enumerate() {
                let at = (start + i) % len;
                let sound: f64 = self
                    .heard
                    .iter()
                    .zip(&playing)
                    .filter(|(_, playing)| **playing)
                    .map(|(heard, _)| heard[mic][at])
                    .sum();
                *sample = gain * sound + 1e-4 * self.background.next();
            }
        }
        let reference =
            Array2::from_shape_fn((1, n), |(_, i)| gain * self.noise[(start + i) % len]);
        LiveSnapshot {
            inputs: TimeSignal::new(inputs, self.sample_rate).expect("valid sample rate"),
            reference: TimeSignal::new(reference, self.sample_rate).expect("valid sample rate"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(speakers: &[u16]) -> LiveRequest {
        LiveRequest {
            device: None,
            speakers: speakers
                .iter()
                .map(|&s| (s, format!("Speaker {}", s + 1)))
                .collect(),
            mics: vec![(0, "Mic 1".into()), (1, "Mic 2".into())],
            sample_rate: 48_000.0,
            level_db: -20.0,
            periods: 4,
        }
    }

    /// Starts a simulated measurement that has been running for `seconds`.
    fn running(request: LiveRequest, seconds: f64) -> Live {
        let mut live = Live::start(request).unwrap();
        let ago = Duration::from_secs_f64(seconds);
        live.changed -= ago;
        if let Source::Simulated(simulation) = &mut live.source {
            simulation.started -= ago;
        }
        live
    }

    #[test]
    fn periods_are_powers_of_two_of_about_a_third_of_a_second() {
        assert_eq!(period_len(48_000.0), 16_384);
        assert_eq!(period_len(44_100.0), 16_384);
        assert_eq!(period_len(96_000.0), 32_768);
    }

    #[test]
    fn waits_for_a_whole_period_after_a_change() {
        let mut live = running(request(&[0, 1]), 5.0);
        assert!(live.take(vec![None]).is_some());
        live.set_playing(1, false);
        assert!(
            live.take(vec![None]).is_none(),
            "the recording is of both speakers"
        );
        assert_eq!(live.playing_names(), "Speaker 1");
    }

    #[test]
    fn finds_each_mics_response_and_delay() {
        let mut live = running(request(&[0]), 5.0);
        let analysis = analyze(live.take(vec![Some(6.0)]).unwrap()).unwrap();
        assert_eq!(analysis.responses.len(), 2);
        let response = &analysis.responses[1];
        assert_eq!(
            format!("{} → {}", response.speaker, response.mic),
            "Speaker 1 → Mic 2"
        );
        // the simulated room's latency to mic 2
        assert!(
            (120.0..123.0).contains(&analysis.delays_ms[1]),
            "{:?}",
            analysis.delays_ms
        );
        // a clean simulation: trusted almost everywhere
        let curves = &analysis.curves[1][0];
        for hz in [100.0, 1000.0, 10_000.0] {
            assert!(curves.trusted(hz, 0.9), "{hz} Hz");
        }
    }

    #[test]
    fn speakers_together_add_up() {
        // with the same delay taken out, the response of both is the sum of each alone
        let response = |playing: [bool; 2]| {
            let mut live = running(request(&[0, 1]), 5.0);
            live.delays_ms = Some(vec![0.0, 0.0]);
            for (speaker, on) in playing.into_iter().enumerate() {
                live.playing[speaker].store(on, Ordering::Relaxed);
            }
            let analysis = analyze(live.take(vec![None]).unwrap()).unwrap();
            analysis.responses[0].frequency_response.clone()
        };
        let (first, second, both) = (
            response([true, false]),
            response([false, true]),
            response([true, true]),
        );
        // over a band, relative to the parts: where they cancel, the sum is small and the
        // background noise relatively large
        let (mut error, mut size) = (0.0, 0.0);
        for bin in 100..4000 {
            let (a, b) = (first.channel(0)[bin], second.channel(0)[bin]);
            error += (both.channel(0)[bin] - (a + b)).norm();
            size += a.norm() + b.norm();
        }
        assert!(error / size < 0.01, "off by {:.2} %", error / size * 100.0);
    }
}
