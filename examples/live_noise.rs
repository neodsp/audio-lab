//! Live transfer function with pink noise on several speakers at once.
//!
//! Plays looped pink noise on all speakers while the microphones record, and shows the transfer
//! function from the noise to each mic, updated several times a second. Because it divides by
//! the noise, a flat system shows a flat line, whatever the color of the noise.
//!
//! The noise goes through a delay on the last speaker before it is played. The delay is part of
//! the measured chain, so moving its slider shows live how the speakers add up, e.g. a sub and a
//! main speaker around the crossover. Your own processing (a crossover, EQ) goes in the same
//! place.
//!
//! ```sh
//! # speakers on outputs 0 and 1, mics on inputs 0 and 1
//! cargo run --release --example live_noise -- --outputs 0,1 --inputs 0,1
//! ```
//!
//! Options (all optional): `--outputs 0,1` (default `0`), `--inputs 0,1` (default `0`),
//! `--input-device NAME`, `--output-device NAME`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use audio_lab::dsp::{
    FractionalOctaveSmoothingConfig, FractionalOctaveSmoothingMode, TransferFunctionConfig,
    smooth_fractional_octave, transfer_function,
};
use audio_lab::signal::audio_blocks::AudioBlockMut;
use audio_lab::signal::measure::{NoiseControls, NoiseSource, NoiseSourceConfig};
use audio_lab::signal::playback::{DeviceConfig, LiveSession, LiveSessionConfig};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};

const SAMPLE_RATE: f64 = 48_000.0;
/// Length of the looped noise and of each analysis segment: 341 ms at 48 kHz. It must be
/// longer than the device latency plus the room's reverb.
const PERIOD: usize = 16_384;
const MAX_DELAY_MS: f32 = 20.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;

    let mut noise = NoiseSource::new(&NoiseSourceConfig {
        outputs: args.outputs.clone(),
        period: PERIOD,
        gain: db_to_gain(-20.0),
        sample_rate: SAMPLE_RATE,
        ..Default::default()
    })?;
    let controls = noise.controls();

    // processing between the noise and the speakers: a delay on the last output
    let delayed_output = *args.outputs.last().expect("at least one output");
    let delay_samples = Arc::new(AtomicUsize::new(0));
    let mut delay = Delay::new(
        (MAX_DELAY_MS / 1000.0 * SAMPLE_RATE as f32) as usize,
        Arc::clone(&delay_samples),
    );

    let config = LiveSessionConfig {
        inputs: args.inputs.clone(),
        num_output_channels: noise.num_output_channels(),
        sample_rate: SAMPLE_RATE,
        ..Default::default()
    };
    let device = DeviceConfig {
        input_device: args.input_device,
        output_device: args.output_device,
        ..Default::default()
    };
    let session = LiveSession::start(
        move |output, reference| {
            // the reference stays the raw noise, so the delay is part of what we measure
            noise.fill(output, reference);
            delay.process(output, delayed_output);
        },
        &config,
        &device,
    )?;

    let app = LiveNoiseApp {
        session,
        controls,
        delay_samples,
        delayed_output,
        gain_db: -20.0,
        delay_ms: 0.0,
        num_fractions: 6.0,
        num_periods: 4,
        series: Vec::new(),
        peak_db: Vec::new(),
    };
    eframe::run_native(
        "Live transfer function",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )?;
    Ok(())
}

struct LiveNoiseApp {
    session: LiveSession,
    controls: NoiseControls,
    delay_samples: Arc<AtomicUsize>,
    delayed_output: u16,
    gain_db: f32,
    delay_ms: f32,
    /// Smoothing, 1/N octave.
    num_fractions: f64,
    /// Noise periods averaged per update.
    num_periods: usize,
    series: Vec<(String, Vec<PlotPoint>)>,
    /// Recorded peak level per mic.
    peak_db: Vec<f64>,
}

impl LiveNoiseApp {
    fn update_response(&mut self) {
        let snapshot = self.session.latest(self.num_periods * PERIOD);
        if snapshot.inputs.num_time_steps() < PERIOD {
            return; // still filling up
        }
        self.peak_db = snapshot
            .inputs
            .channel_iter()
            .map(|c| 20.0 * c.iter().fold(1e-12f64, |m, s| m.max(s.abs())).log10())
            .collect();

        let config = TransferFunctionConfig::default().with_segment_len(PERIOD);
        let Ok(response) = transfer_function(&snapshot.inputs, &snapshot.reference, &config) else {
            return;
        };
        let smoothing = FractionalOctaveSmoothingConfig {
            num_fractions: self.num_fractions,
            ..Default::default()
        }
        .with_mode(FractionalOctaveSmoothingMode::MagnitudeZeroPhase);
        let Ok((response, _)) = smooth_fractional_octave(&response, &smoothing) else {
            return;
        };

        let freqs = response.freq_bins();
        self.series = response
            .channel_iter()
            .enumerate()
            .map(|(channel, values)| {
                let points = freqs
                    .iter()
                    .zip(values.iter())
                    .filter(|&(&f, _)| (20.0..=20_000.0).contains(&f))
                    .map(|(&f, h)| PlotPoint::new(f.log10(), 20.0 * h.norm().max(1e-12).log10()))
                    .collect();
                let label = response.channel_label(channel).unwrap_or("mic").to_string();
                (label, points)
            })
            .collect();
    }
}

impl eframe::App for LiveNoiseApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update_response();

        egui::Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut enabled = self.controls.is_enabled();
                if ui.checkbox(&mut enabled, "Noise on").changed() {
                    self.controls.set_enabled(enabled);
                }
                let gain = egui::Slider::new(&mut self.gain_db, -60.0..=0.0).text("Level (dBFS)");
                if ui.add(gain).changed() {
                    self.controls.set_gain(db_to_gain(self.gain_db));
                }
                let delay = egui::Slider::new(&mut self.delay_ms, 0.0..=MAX_DELAY_MS)
                    .text(format!("Delay on output {} (ms)", self.delayed_output));
                if ui.add(delay).changed() {
                    let samples = (self.delay_ms / 1000.0 * SAMPLE_RATE as f32).round() as usize;
                    self.delay_samples.store(samples, Ordering::Relaxed);
                }
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.num_fractions, 1.0..=48.0)
                        .text("Smoothing (1/N octave)"),
                );
                ui.add(egui::Slider::new(&mut self.num_periods, 1..=8).text("Averaged periods"));
            });
            ui.horizontal(|ui| {
                for (label, peak) in self.series.iter().map(|(l, _)| l).zip(&self.peak_db) {
                    let text = format!("{label}: peak {peak:.1} dBFS");
                    if *peak >= -0.1 {
                        ui.colored_label(egui::Color32::RED, format!("{text}, clipping!"));
                    } else {
                        ui.label(text);
                    }
                }
                let status = self.session.status();
                ui.label(format!(
                    "dropouts: {}, dropped frames: {}",
                    status.xruns,
                    self.session.dropped_frames()
                ));
                if let Some(error) = status.error {
                    ui.colored_label(egui::Color32::RED, error.to_string());
                }
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            Plot::new("transfer_function")
                .x_axis_label("Frequency")
                .y_axis_label("Magnitude (dB)")
                .x_axis_formatter(|mark, _| {
                    let hz = 10f64.powf(mark.value);
                    if hz >= 1000.0 {
                        format!("{:.0} kHz", hz / 1000.0)
                    } else {
                        format!("{hz:.0} Hz")
                    }
                })
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    for (label, points) in &self.series {
                        plot_ui.line(Line::new(label.clone(), PlotPoints::Borrowed(points)));
                    }
                });
        });

        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));
    }
}

/// A delay on one output channel, changeable from the UI. Real-time safe.
struct Delay {
    buffer: Vec<f32>,
    write: usize,
    samples: Arc<AtomicUsize>,
}

impl Delay {
    fn new(max_samples: usize, samples: Arc<AtomicUsize>) -> Self {
        Self {
            buffer: vec![0.0; max_samples + 1],
            write: 0,
            samples,
        }
    }

    fn process(&mut self, block: &mut impl AudioBlockMut<f32>, channel: u16) {
        if channel >= block.num_channels() {
            return;
        }
        let len = self.buffer.len();
        let delay = self.samples.load(Ordering::Relaxed).min(len - 1);
        for sample in block.channel_iter_mut(channel) {
            self.buffer[self.write] = *sample;
            *sample = self.buffer[(self.write + len - delay) % len];
            self.write = (self.write + 1) % len;
        }
    }
}

fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

struct Args {
    outputs: Vec<u16>,
    inputs: Vec<u16>,
    input_device: Option<String>,
    output_device: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            outputs: vec![0],
            inputs: vec![0],
            input_device: None,
            output_device: None,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(flag) = iter.next() {
            let value = iter.next().ok_or(format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--outputs" => args.outputs = channels(&value)?,
                "--inputs" => args.inputs = channels(&value)?,
                "--input-device" => args.input_device = Some(value),
                "--output-device" => args.output_device = Some(value),
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        if args.outputs.is_empty() {
            return Err("--outputs needs at least one channel".into());
        }
        Ok(args)
    }
}

fn channels(list: &str) -> Result<Vec<u16>, String> {
    list.split(',')
        .map(|c| c.trim().parse().map_err(|_| format!("bad channel {c}")))
        .collect()
}
