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
//! "Hold" keeps the current curves next to the live ones, e.g. to compare delay settings.
//! Hover a curve to read its level; click its legend entry to hide it.
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
use audio_lab::signal::FreqSignal;
use audio_lab::signal::audio_blocks::AudioBlockMut;
use audio_lab::signal::measure::{NoiseControls, NoiseSource, NoiseSourceConfig};
use audio_lab::signal::playback::{DeviceConfig, LiveSession, LiveSessionConfig};
use eframe::egui;
use egui_plot::{HoverPosition, Legend, Line, Plot, PlotPoint, PlotPoints};

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
        response: None,
        series: Vec::new(),
        held: Vec::new(),
        num_held: 0,
        peak_db: Vec::new(),
        fit_y: true,
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
    /// The latest transfer function, unsmoothed.
    response: Option<FreqSignal>,
    /// The latest curve per mic.
    series: Vec<Series>,
    /// Curves kept with "Hold".
    held: Vec<Held>,
    /// Holds made so far, to number them.
    num_held: usize,
    /// Recorded peak level per mic.
    peak_db: Vec<f64>,
    /// Fit the magnitude axis to the data on the next frame.
    fit_y: bool,
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
        self.series = curves(&response, self.num_fractions, None);
        self.response = Some(response);

        // held curves follow the smoothing slider
        for held in &mut self.held {
            if held.num_fractions != self.num_fractions {
                held.series = curves(&held.response, self.num_fractions, Some(&held.name));
                held.num_fractions = self.num_fractions;
            }
        }
    }

    /// Keeps the current curves, named after the hold and the delay setting.
    fn hold(&mut self) {
        let Some(response) = self.response.clone() else {
            return;
        };
        self.num_held += 1;
        let name = format!("Hold {} (delay {:.1} ms)", self.num_held, self.delay_ms);
        self.held.push(Held {
            series: curves(&response, self.num_fractions, Some(&name)),
            name,
            response,
            num_fractions: self.num_fractions,
        });
    }
}

/// A legend label and the points of one curve.
type Series = (String, Vec<PlotPoint>);

/// Curves kept with "Hold". Unsmoothed, so the smoothing slider still applies to them.
struct Held {
    name: String,
    response: FreqSignal,
    series: Vec<Series>,
    /// The smoothing `series` was made with.
    num_fractions: f64,
}

/// One smoothed curve per mic, in dB over log frequency, labeled like the other plots:
/// `name · mic`.
fn curves(response: &FreqSignal, num_fractions: f64, name: Option<&str>) -> Vec<Series> {
    let smoothing = FractionalOctaveSmoothingConfig {
        num_fractions,
        ..Default::default()
    }
    .with_mode(FractionalOctaveSmoothingMode::MagnitudeZeroPhase);
    let Ok((response, _)) = smooth_fractional_octave(response, &smoothing) else {
        return Vec::new();
    };
    let freqs = response.freq_bins();
    response
        .channel_iter()
        .enumerate()
        .map(|(channel, values)| {
            let points = freqs
                .iter()
                .zip(values.iter())
                .filter(|&(&f, _)| (20.0..=20_000.0).contains(&f))
                .map(|(&f, h)| PlotPoint::new(f.log10(), 20.0 * h.norm().max(1e-12).log10()))
                .collect();
            let mic = response.channel_label(channel).unwrap_or("mic");
            let label = match name {
                Some(name) => format!("{name} · {mic}"),
                None => mic.to_string(),
            };
            (label, points)
        })
        .collect()
}

fn format_hz(hz: f64, precision: usize) -> String {
    if hz >= 1000.0 {
        format!("{:.*} kHz", precision, hz / 1000.0)
    } else {
        format!("{hz:.precision$} Hz")
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
                if ui.button("Fit").clicked() {
                    self.fit_y = true;
                }
                let hold = ui.add_enabled(self.response.is_some(), egui::Button::new("Hold"));
                if hold.on_hover_text("Keep the current curves").clicked() {
                    self.hold();
                }
                let clear = ui.add_enabled(!self.held.is_empty(), egui::Button::new("Clear held"));
                if clear.clicked() {
                    self.held.clear();
                }
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
            // fixed bounds, so the view doesn't jump with every update; double-click resets
            Plot::new("transfer_function")
                .default_x_bounds(20f64.log10(), 20_000f64.log10())
                .default_y_bounds(-60.0, 20.0)
                .x_axis_label("Frequency")
                .y_axis_label("Magnitude (dB)")
                .x_axis_formatter(|mark, _| format_hz(10f64.powf(mark.value), 0))
                // the curve, frequency and level under the mouse
                .label_formatter(|pos| {
                    let (name, point) = match pos {
                        HoverPosition::NearDataPoint {
                            plot_name,
                            position,
                            ..
                        } => (*plot_name, position),
                        HoverPosition::Elsewhere { position } => ("", position),
                    };
                    let text = format!("{}\n{:.1} dB", format_hz(10f64.powf(point.x), 1), point.y);
                    Some(if name.is_empty() {
                        text
                    } else {
                        format!("{name}\n{text}")
                    })
                })
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    let held = self.held.iter().flat_map(|h| &h.series);
                    if self.fit_y && !self.series.is_empty() {
                        let (lo, hi) = self
                            .series
                            .iter()
                            .chain(held.clone())
                            .flat_map(|(_, points)| points.iter().map(|p| p.y))
                            .fold((f64::MAX, f64::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
                        if lo <= hi {
                            plot_ui.set_plot_bounds_y(lo - 6.0..=hi + 6.0);
                            self.fit_y = false;
                        }
                    }
                    // held first, so the live curves stay on top
                    for (label, points) in held.chain(&self.series) {
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
