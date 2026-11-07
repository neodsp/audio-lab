//! Runs a sweep measurement in the background and turns the recordings into responses.

use std::f64::consts::TAU;
use std::ops::Range;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use audio_lab::dsp::{ConvolveMode, DeconvolutionConfig, convolve, deconvolve, window_and_trim};
use audio_lab::signal::measure::{SequenceConfig, Step};
use audio_lab::signal::ndarray::{Array2, Axis};
use audio_lab::signal::playback::{DeviceConfig, play_and_record};
use audio_lab::signal::{FreqSignal, SweepConfig, SweepType, TimeSignal, generate_sweep};
use num::complex::Complex64;

/// Silence before the first sweep.
const WARMUP: Duration = Duration::from_millis(300);
/// Channels of the simulated device.
pub const SIMULATED_CHANNELS: u16 = 4;

/// One output or input channel in the setup: whether it takes part and its name.
#[derive(Clone)]
pub struct Channel {
    pub enabled: bool,
    pub name: String,
}

/// Everything the setup panel lets you choose.
pub struct Settings {
    /// Name of the next measurement.
    pub name: String,
    /// Measure a simulated room instead of a device, to try the app without hardware.
    pub simulated: bool,
    /// `None` for the system default.
    pub output_device: Option<String>,
    pub input_device: Option<String>,
    /// One per channel of the output device: the speakers.
    pub outputs: Vec<Channel>,
    /// One per channel of the input device: the microphones.
    pub inputs: Vec<Channel>,
    pub sample_rate: u32,
    /// The sweep is 2^n samples long.
    pub sweep_exponent: u32,
    /// Peak level of the sweep in dBFS.
    pub level_db: f64,
    pub freq_low: f64,
    pub freq_high: f64,
    /// Recorded after each sweep, for the reverb and the device latency.
    pub tail_ms: f64,
    /// How much of each impulse response is kept, after its start.
    pub ir_length_ms: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            name: "Measurement 1".into(),
            simulated: false,
            output_device: None,
            input_device: None,
            outputs: Vec::new(),
            inputs: Vec::new(),
            sample_rate: 48_000,
            sweep_exponent: 16,
            level_db: -12.0,
            freq_low: 20.0,
            freq_high: 20_000.0,
            tail_ms: 1500.0,
            ir_length_ms: 500.0,
        }
    }
}

impl Settings {
    /// Matches the channel lists to the devices' channel counts, keeping names and choices.
    /// The first channel of an empty list starts enabled.
    pub fn sync_channels(&mut self, num_outputs: u16, num_inputs: u16) {
        resize(&mut self.outputs, num_outputs, "Speaker");
        resize(&mut self.inputs, num_inputs, "Mic");
    }

    pub fn sweep_len(&self) -> usize {
        1 << self.sweep_exponent
    }

    /// Why the measurement can't start, if it can't.
    pub fn problem(&self) -> Option<String> {
        if !self.outputs.iter().any(|c| c.enabled) {
            return Some("Choose at least one speaker.".into());
        }
        if !self.inputs.iter().any(|c| c.enabled) {
            return Some("Choose at least one microphone.".into());
        }
        let nyquist = f64::from(self.sample_rate) / 2.0;
        if !(self.freq_low > 0.0 && self.freq_low < self.freq_high && self.freq_high <= nyquist) {
            return Some(format!(
                "The frequency range must lie between 0 and {nyquist:.0} Hz."
            ));
        }
        None
    }

    /// What to measure, for the background thread.
    pub fn request(&self) -> Request {
        let enabled = |channels: &[Channel]| {
            channels
                .iter()
                .enumerate()
                .filter(|(_, channel)| channel.enabled)
                .map(|(index, channel)| (index as u16, channel.name.clone()))
                .collect()
        };
        Request {
            device: (!self.simulated).then(|| DeviceConfig {
                input_device: self.input_device.clone(),
                output_device: self.output_device.clone(),
                ..Default::default()
            }),
            speakers: enabled(&self.outputs),
            mics: enabled(&self.inputs),
            sample_rate: f64::from(self.sample_rate),
            sweep_len: self.sweep_len(),
            level_db: self.level_db,
            range: self.freq_low..self.freq_high,
            tail: Duration::from_secs_f64(self.tail_ms / 1000.0),
            ir_length: self.ir_length_ms / 1000.0,
        }
    }
}

fn resize(channels: &mut Vec<Channel>, len: u16, prefix: &str) {
    let was_empty = channels.is_empty();
    let len = usize::from(len);
    while channels.len() < len {
        channels.push(Channel {
            enabled: was_empty && channels.is_empty(),
            name: format!("{prefix} {}", channels.len() + 1),
        });
    }
    channels.truncate(len);
}

/// A measurement to run: one sweep per speaker, all mics recording.
#[derive(Clone)]
pub struct Request {
    /// `None` for the simulated device.
    device: Option<DeviceConfig>,
    /// Channel and name of each speaker.
    speakers: Vec<(u16, String)>,
    mics: Vec<(u16, String)>,
    sample_rate: f64,
    sweep_len: usize,
    level_db: f64,
    pub range: Range<f64>,
    tail: Duration,
    /// Seconds.
    ir_length: f64,
}

impl Request {
    /// About how long the measurement takes.
    pub fn duration(&self) -> Duration {
        let Some(device) = &self.device else {
            return Duration::ZERO;
        };
        let sweep = Duration::from_secs_f64(self.sweep_len as f64 / self.sample_rate);
        device.settle + WARMUP + (sweep + self.tail) * self.speakers.len() as u32
    }
}

/// The response from one speaker to one mic.
pub struct Response {
    pub speaker: String,
    pub mic: String,
    /// The impulse response from its start, windowed.
    pub impulse_response: TimeSignal,
    /// Where `impulse_response` starts in the recording, in samples. Includes the latency.
    pub start: usize,
    /// The frequency response, with the phase relative to the impulse response's peak.
    pub frequency_response: FreqSignal,
    /// Peak level of the recording in dBFS.
    pub peak_db: f64,
}

pub struct Outcome {
    pub responses: Vec<Response>,
    /// Problems with single responses, e.g. clipping.
    pub warnings: Vec<String>,
}

/// A measurement running on a background thread.
pub struct Job {
    rx: Receiver<Result<Outcome, String>>,
    started: Instant,
    expected: Duration,
    pub request: Request,
}

impl Job {
    pub fn start(request: Request, ctx: &eframe::egui::Context) -> Self {
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let job_request = request.clone();
        std::thread::spawn(move || {
            let _ = tx.send(run(&job_request));
            ctx.request_repaint();
        });
        Self {
            rx,
            started: Instant::now(),
            expected: request.duration(),
            request,
        }
    }

    /// The result once the measurement is done.
    pub fn poll(&self) -> Option<Result<Outcome, String>> {
        match self.rx.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("the measurement thread crashed".into())),
        }
    }

    /// From 0 to 1, by the time it is expected to take. Stays below 1 until it is done.
    pub fn progress(&self) -> f32 {
        if self.expected.is_zero() {
            return 0.99;
        }
        (self.started.elapsed().as_secs_f32() / self.expected.as_secs_f32()).min(0.99)
    }

    pub fn remaining(&self) -> Duration {
        self.expected.saturating_sub(self.started.elapsed())
    }
}

fn run(request: &Request) -> Result<Outcome, String> {
    let sample_rate = request.sample_rate;
    let sweep = generate_sweep(
        request.sweep_len,
        request.range.clone(),
        SweepConfig {
            amplitude: 10f64.powf(request.level_db / 20.0),
            sample_rate,
            num_channels: 1,
            // a full cycle of 20 Hz, so the sweep doesn't start with a click
            fade_in: (0.05 * sample_rate) as usize,
            fade_out: 90,
            sweep_type: SweepType::Exponential,
        },
    )
    .map_err(|error| error.to_string())?;

    let config = SequenceConfig {
        inputs: request.mics.iter().map(|(channel, _)| *channel).collect(),
        warmup: WARMUP,
        tail: request.tail,
    };
    let recordings = match &request.device {
        Some(device) => {
            // one step per speaker: they play one after another
            let steps = request
                .speakers
                .iter()
                .map(|(output, _)| Step::new(vec![*output], sweep.clone()))
                .collect();
            play_and_record(steps, &config, device).map_err(|error| error.to_string())?
        }
        None => simulate(&sweep, request)?,
    };

    let deconvolution = DeconvolutionConfig::default().with_frequency_range(request.range.clone());
    let mut outcome = Outcome {
        responses: Vec::new(),
        warnings: Vec::new(),
    };
    for ((_, speaker), recording) in request.speakers.iter().zip(&recordings) {
        for (index, (_, mic)) in request.mics.iter().enumerate() {
            let label = format!("{speaker} → {mic}");
            let recorded = channel(recording, index);
            let peak = recorded
                .channel(0)
                .iter()
                .fold(1e-12f64, |m, s| m.max(s.abs()));
            if peak >= 1.0 {
                outcome.warnings.push(format!(
                    "{label}: the recording clips, lower the level or the input gain"
                ));
            }
            let ir = deconvolve(&recorded, &sweep, &deconvolution).map_err(|e| e.to_string())?;
            let (impulse_response, starts) = match window_and_trim(&ir, request.ir_length) {
                Ok(trimmed) => trimmed,
                Err(error) => {
                    outcome
                        .warnings
                        .push(format!("{label}: no impulse response found ({error})"));
                    continue;
                }
            };
            outcome.responses.push(Response {
                speaker: speaker.clone(),
                mic: mic.clone(),
                frequency_response: response_from_peak(&impulse_response),
                impulse_response,
                start: starts[0],
                peak_db: 20.0 * peak.log10(),
            });
        }
    }
    if outcome.responses.is_empty() {
        let mut message = "No impulse response found, check the routing and levels.".to_string();
        for warning in &outcome.warnings {
            message.push('\n');
            message.push_str(warning);
        }
        return Err(message);
    }
    Ok(outcome)
}

/// The frequency response of `ir`, with the phase relative to the impulse response's peak: the
/// delay up to it would turn the phase so quickly that it wraps every few Hz.
fn response_from_peak(ir: &TimeSignal) -> FreqSignal {
    let peak = peak_index(ir);
    let mut response = ir.to_freq();
    let freqs = response.freq_bins().to_owned();
    let sample_rate = ir.sample_rate();
    for (h, f) in response.channel_mut(0).iter_mut().zip(freqs) {
        *h *= Complex64::from_polar(1.0, TAU * f * peak as f64 / sample_rate);
    }
    response
}

/// Index of the largest sample of the first channel.
pub fn peak_index(signal: &TimeSignal) -> usize {
    signal
        .channel(0)
        .iter()
        .enumerate()
        .fold(
            (0, 0.0),
            |best, (i, s)| {
                if s.abs() > best.1 { (i, s.abs()) } else { best }
            },
        )
        .0
}

/// One channel of `signal` as a mono signal.
fn channel(signal: &TimeSignal, index: usize) -> TimeSignal {
    let data = signal.channel(index).to_owned().insert_axis(Axis(0));
    TimeSignal::new(data, signal.sample_rate()).expect("valid sample rate")
}

/// Recordings like `play_and_record` returns them, from a made-up room: each speaker reaches
/// each mic with its own delay, a speaker-like band limit, a few reflections and reverb.
fn simulate(sweep: &TimeSignal, request: &Request) -> Result<Vec<TimeSignal>, String> {
    let sample_rate = request.sample_rate;
    let len = request.sweep_len + (request.tail.as_secs_f64() * sample_rate) as usize;
    let mut noise = Lcg(1);
    request
        .speakers
        .iter()
        .map(|&(speaker, _)| {
            let mut recording = Array2::zeros((request.mics.len(), len));
            for (index, &(mic, _)) in request.mics.iter().enumerate() {
                let ir = simulated_room(speaker, mic, sample_rate, &mut noise);
                let heard = convolve(sweep, &ir, ConvolveMode::Full).map_err(|e| e.to_string())?;
                for (dst, src) in recording
                    .row_mut(index)
                    .iter_mut()
                    .zip(heard.channel(0).iter())
                {
                    // and a little background noise
                    *dst = src + 1e-4 * noise.next();
                }
            }
            TimeSignal::new(recording, sample_rate).map_err(|error| error.to_string())
        })
        .collect()
}

/// A made-up impulse response from `speaker` to `mic`.
fn simulated_room(speaker: u16, mic: u16, sample_rate: f64, noise: &mut Lcg) -> TimeSignal {
    let samples = |ms: f64| (ms / 1000.0 * sample_rate) as usize;
    let mut ir = vec![0.0; samples(400.0)];
    // device latency plus the distance, different for every pair
    let direct = samples(120.0 + 2.5 * f64::from(speaker) + 1.1 * f64::from(mic));
    ir[direct] = 0.5;
    for (delay_ms, gain) in [(1.9, -0.25), (4.7, 0.18), (9.3, -0.12), (17.0, 0.08)] {
        let at = direct + samples(delay_ms * (1.0 + 0.2 * f64::from(speaker)));
        ir[at] += gain;
    }
    // reverb: decaying noise, about 0.35 s to fall by 60 dB
    let decay = (-6.9 / (0.35 * sample_rate)).exp();
    let mut envelope = 0.005;
    for sample in &mut ir[direct + samples(3.0)..] {
        *sample += envelope * noise.next();
        envelope *= decay;
    }
    // a speaker: high-pass around 60 Hz, low-pass around 12 kHz, a bit different per speaker
    let one_pole = |cutoff: f64| (-TAU * cutoff / sample_rate).exp();
    let (high_pass, low_pass) = (
        one_pole(60.0 + 30.0 * f64::from(speaker)),
        one_pole(12_000.0),
    );
    let (mut x_last, mut high, mut low) = (0.0, 0.0, 0.0);
    for sample in &mut ir {
        high = high_pass * (high + *sample - x_last);
        x_last = *sample;
        low = (1.0 - low_pass) * high + low_pass * low;
        *sample = low;
    }
    TimeSignal::new(
        Array2::from_shape_vec((1, ir.len()), ir).unwrap(),
        sample_rate,
    )
    .unwrap()
}

/// Repeatable noise between -1 and 1.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulated_measurement_finds_every_response() {
        let mut settings = Settings {
            simulated: true,
            ..Default::default()
        };
        settings.sync_channels(SIMULATED_CHANNELS, SIMULATED_CHANNELS);
        settings.outputs[1].enabled = true;
        settings.inputs[2].enabled = true;
        let outcome = run(&settings.request()).unwrap();
        assert!(outcome.warnings.is_empty(), "{:?}", outcome.warnings);
        assert_eq!(outcome.responses.len(), 4);
        let names: Vec<_> = outcome
            .responses
            .iter()
            .map(|r| format!("{} → {}", r.speaker, r.mic))
            .collect();
        assert_eq!(
            names,
            [
                "Speaker 1 → Mic 1",
                "Speaker 1 → Mic 3",
                "Speaker 2 → Mic 1",
                "Speaker 2 → Mic 3"
            ]
        );
        // the latency of the simulated room
        let start_ms = outcome.responses[0].start as f64 / 48.0;
        assert!((115.0..121.0).contains(&start_ms), "{start_ms}");
    }

    #[test]
    fn channel_lists_follow_the_device_and_keep_names() {
        let mut settings = Settings::default();
        settings.sync_channels(2, 1);
        assert!(settings.outputs[0].enabled && !settings.outputs[1].enabled);
        settings.outputs[1].name = "Sub".into();
        settings.sync_channels(4, 1);
        assert_eq!(settings.outputs[1].name, "Sub");
        assert_eq!(settings.outputs[3].name, "Speaker 4");
        assert!(!settings.outputs[3].enabled);
        settings.sync_channels(1, 1);
        assert_eq!(settings.outputs.len(), 1);
    }
}
