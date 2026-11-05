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
//! The phase plot above the magnitude shows the phase wrapped to ±180°. Each mic's delay
//! (device latency plus the sound's travel time) is taken out first, otherwise the phase would
//! wrap every few Hz. It is found from the impulse response at the start and with "Find", and
//! can be nudged by hand; between finds it stays fixed, so a change such as the delay slider
//! shows up in the phase.
//!
//! "Hold" keeps the current curves next to the live ones, e.g. to compare delay settings.
//! Hover a curve to read its value; click its legend entry to hide it in both plots.
//!
//! ```sh
//! # speakers on outputs 0 and 1, mics on inputs 0 and 1
//! cargo run --release --example live_noise -- --outputs 0,1 --inputs 0,1
//! ```
//!
//! Options (all optional): `--outputs 0,1` (default `0`), `--inputs 0,1` (default `0`),
//! `--input-device NAME`, `--output-device NAME`.

use std::f64::consts::TAU;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use audio_lab::dsp::{
    FractionalOctaveSmoothingConfig, FractionalOctaveSmoothingMode, TransferFunctionConfig,
    coherence, smooth_fractional_octave, transfer_function,
};
use audio_lab::signal::FreqSignal;
use audio_lab::signal::audio_blocks::AudioBlockMut;
use audio_lab::signal::measure::{NoiseControls, NoiseSource, NoiseSourceConfig};
use audio_lab::signal::playback::{DeviceConfig, LiveSession, LiveSessionConfig};
use eframe::egui;
use egui_plot::{
    ColorConflictHandling, HoverPosition, ItemId, Legend, Line, Plot, PlotMemory, PlotPoint,
    PlotPoints, uniform_grid_spacer,
};
use num::complex::Complex64;

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
        compensation_ms: Vec::new(),
        min_coherence: 0.5,
        measurement: None,
        curves: Vec::new(),
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
    /// Delay taken out of the phase per mic: the device latency plus the sound's travel time.
    /// Without it the phase wraps every few Hz.
    compensation_ms: Vec<f64>,
    /// Below this coherence the phase is hidden and the magnitude faded.
    min_coherence: f64,
    /// The latest measurement, unsmoothed.
    measurement: Option<Measurement>,
    /// The latest curve per mic.
    curves: Vec<Curve>,
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
        let (Ok(response), Ok(coherence)) = (
            transfer_function(&snapshot.inputs, &snapshot.reference, &config),
            coherence(&snapshot.inputs, &snapshot.reference, &config),
        ) else {
            return;
        };
        if self.compensation_ms.len() != response.num_channels() {
            self.compensation_ms = find_delays_ms(&response);
        }
        let measurement = Measurement {
            response,
            coherence,
        };
        let settings = self.settings();
        self.curves = curves(&measurement, &settings, None, 0);
        self.measurement = Some(measurement);

        // held curves follow the smoothing, the delay compensation and the coherence limit
        for held in &mut self.held {
            if held.settings != settings {
                held.curves = curves(&held.measurement, &settings, Some(&held.name), held.color);
                held.settings = settings.clone();
            }
        }
    }

    fn settings(&self) -> CurveSettings {
        CurveSettings {
            num_fractions: self.num_fractions,
            compensation_ms: self.compensation_ms.clone(),
            min_coherence: self.min_coherence,
        }
    }

    /// Keeps the current curves, named after the hold and the delay setting.
    fn hold(&mut self) {
        let Some(measurement) = self.measurement.clone() else {
            return;
        };
        self.num_held += 1;
        let name = format!("Hold {} (delay {:.1} ms)", self.num_held, self.delay_ms);
        // colors after the live curves and the earlier holds
        let color = self.num_held * measurement.response.num_channels();
        let settings = self.settings();
        self.held.push(Held {
            curves: curves(&measurement, &settings, Some(&name), color),
            name,
            measurement,
            color,
            settings,
        });
    }
}

/// One mic's curve, in dB and wrapped degrees over log frequency.
struct Curve {
    label: String,
    color: egui::Color32,
    /// Pieces of the magnitude, and whether their coherence is high enough to trust.
    magnitude: Vec<(bool, Vec<PlotPoint>)>,
    /// Pieces of the phase where the coherence is high enough, split at the wraps so a wrap
    /// from -180° to 180° is not drawn as a line across the plot.
    phase: Vec<Vec<PlotPoint>>,
}

/// A transfer function and its coherence, unsmoothed.
#[derive(Clone)]
struct Measurement {
    response: FreqSignal,
    coherence: FreqSignal,
}

/// What the curves are computed with.
#[derive(Clone, PartialEq)]
struct CurveSettings {
    num_fractions: f64,
    compensation_ms: Vec<f64>,
    min_coherence: f64,
}

/// Curves kept with "Hold". Unsmoothed, so the smoothing and the delay compensation still
/// apply to them.
struct Held {
    name: String,
    measurement: Measurement,
    /// Color of the first mic's curve.
    color: usize,
    curves: Vec<Curve>,
    /// What `curves` were computed with.
    settings: CurveSettings,
}

/// The delay of each mic, from the peak of its impulse response.
fn find_delays_ms(response: &FreqSignal) -> Vec<f64> {
    let impulse_response = response.to_time();
    impulse_response
        .channel_iter()
        .map(|ir| {
            let peak = ir.iter().enumerate().fold((0, 0.0), |best, (i, s)| {
                if s.abs() > best.1 { (i, s.abs()) } else { best }
            });
            peak.0 as f64 / SAMPLE_RATE * 1000.0
        })
        .collect()
}

/// One curve per mic, labeled like the other plots: `name · mic`. The magnitude is smoothed on
/// its own; the phase is smoothed as complex values after the delay compensation is taken out.
/// The coherence is smoothed like the magnitude and decides which parts to trust.
fn curves(
    measurement: &Measurement,
    settings: &CurveSettings,
    name: Option<&str>,
    first_color: usize,
) -> Vec<Curve> {
    let response = &measurement.response;
    let smoothing =
        |mode| FractionalOctaveSmoothingConfig::new(settings.num_fractions).with_mode(mode);
    let magnitude_only = smoothing(FractionalOctaveSmoothingMode::MagnitudeZeroPhase);
    let (Ok((magnitude, _)), Ok((coherence, _))) = (
        smooth_fractional_octave(response, &magnitude_only),
        smooth_fractional_octave(&measurement.coherence, &magnitude_only),
    ) else {
        return Vec::new();
    };

    let freqs = response.freq_bins();
    let mut compensated = response.clone();
    for (mut channel, &ms) in compensated
        .channel_iter_mut()
        .zip(&settings.compensation_ms)
    {
        for (h, &f) in channel.iter_mut().zip(freqs.iter()) {
            *h *= Complex64::from_polar(1.0, TAU * f * ms / 1000.0);
        }
    }
    let Ok((phase, _)) = smooth_fractional_octave(
        &compensated,
        &smoothing(FractionalOctaveSmoothingMode::Complex),
    ) else {
        return Vec::new();
    };

    let in_range = |f: &f64| (20.0..=20_000.0).contains(f);
    (0..response.num_channels())
        .map(|channel| {
            let bins = || {
                freqs
                    .iter()
                    .zip(coherence.channel(channel))
                    .enumerate()
                    .filter(|(_, (f, _))| in_range(f))
                    .map(|(bin, (&f, c))| (bin, f.log10(), c.re >= settings.min_coherence))
            };

            // trusted and faded pieces, sharing their end points so the line stays connected
            let mut magnitude_pieces: Vec<(bool, Vec<PlotPoint>)> = Vec::new();
            for (bin, x, good) in bins() {
                let h = magnitude.channel(channel)[bin];
                let point = PlotPoint::new(x, 20.0 * h.norm().max(1e-12).log10());
                match magnitude_pieces.last_mut() {
                    Some((piece_good, piece)) if *piece_good == good => piece.push(point),
                    last => {
                        let start = last.and_then(|(_, piece)| piece.last().copied());
                        magnitude_pieces.push((good, start.into_iter().chain([point]).collect()));
                    }
                }
            }

            // the phase only where it is trusted
            let mut phase_pieces = Vec::new();
            let mut run = Vec::new();
            for (bin, x, good) in bins() {
                if good {
                    let degrees = phase.channel(channel)[bin].arg().to_degrees();
                    run.push(PlotPoint::new(x, degrees));
                } else if !run.is_empty() {
                    phase_pieces.extend(split_at_wraps(std::mem::take(&mut run).into_iter()));
                }
            }
            phase_pieces.extend(split_at_wraps(run.into_iter()));

            let mic = response.channel_label(channel).unwrap_or("mic");
            Curve {
                label: match name {
                    Some(name) => format!("{name} · {mic}"),
                    None => mic.to_string(),
                },
                color: color(first_color + channel),
                magnitude: magnitude_pieces,
                phase: phase_pieces,
            }
        })
        .collect()
}

/// Splits a wrapped phase into pieces that each run to the edge where the phase wraps and
/// continue from the other edge, so the curve looks continuous instead of having gaps.
fn split_at_wraps(points: impl Iterator<Item = PlotPoint>) -> Vec<Vec<PlotPoint>> {
    let mut pieces = vec![Vec::new()];
    let mut last: Option<PlotPoint> = None;
    for point in points {
        if let Some(last) = last {
            let step = point.y - last.y;
            // a jump of more than half a turn is a wrap: the phase went the short way round
            if step.abs() > 180.0 {
                let unwrapped = point.y - 360.0 * step.signum();
                let edge = if unwrapped > last.y { 180.0 } else { -180.0 };
                let x = last.x + (point.x - last.x) * (edge - last.y) / (unwrapped - last.y);
                pieces.last_mut().unwrap().push(PlotPoint::new(x, edge));
                pieces.push(vec![PlotPoint::new(x, -edge)]);
            }
        }
        pieces.last_mut().unwrap().push(point);
        last = Some(point);
    }
    pieces
}

/// The `index`th of a series of well distinguishable colors, as egui_plot picks them.
fn color(index: usize) -> egui::Color32 {
    let golden_ratio = (5f32.sqrt() - 1.0) / 2.0;
    egui::epaint::Hsva::new(index as f32 * golden_ratio, 0.85, 0.5, 1.0).into()
}

fn format_hz(hz: f64, precision: usize) -> String {
    if hz >= 1000.0 {
        format!("{:.*} kHz", precision, hz / 1000.0)
    } else {
        format!("{hz:.precision$} Hz")
    }
}

/// Shows the curve, frequency and value under the mouse.
fn hover_label(pos: &HoverPosition, unit: &str) -> Option<String> {
    let (name, point) = match pos {
        HoverPosition::NearDataPoint {
            plot_name,
            position,
            ..
        } => (*plot_name, position),
        HoverPosition::Elsewhere { position } => ("", position),
    };
    let text = format!(
        "{}\n{:.1} {unit}",
        format_hz(10f64.powf(point.x), 1),
        point.y
    );
    Some(if name.is_empty() {
        text
    } else {
        format!("{name}\n{text}")
    })
}

/// A frequency plot: log axis from 20 Hz to 20 kHz, linked to the other plot's zoom and cursor.
fn frequency_plot(id: &str) -> Plot<'static> {
    let link = egui::Id::unique("frequency");
    Plot::new(id)
        .id(egui::Id::unique(id))
        .default_x_bounds(20f64.log10(), 20_000f64.log10())
        .x_axis_formatter(|mark, _| format_hz(10f64.powf(mark.value), 0))
        .link_axis(link, [true, false])
        .link_cursor(link, [true, false])
}

const MAGNITUDE_PLOT: &str = "magnitude";

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
                let coherence =
                    egui::Slider::new(&mut self.min_coherence, 0.0..=1.0).text("Min. coherence");
                ui.add(coherence).on_hover_text(
                    "Hides the phase and fades the magnitude where the result changes from one \
                     noise period to the next, e.g. room noise. Needs at least 2 averaged \
                     periods; with n periods pure noise reads about 1/n.",
                );
                if ui.button("Fit").clicked() {
                    self.fit_y = true;
                }
                let hold = ui.add_enabled(self.measurement.is_some(), egui::Button::new("Hold"));
                if hold.on_hover_text("Keep the current curves").clicked() {
                    self.hold();
                }
                let clear = ui.add_enabled(!self.held.is_empty(), egui::Button::new("Clear held"));
                if clear.clicked() {
                    self.held.clear();
                }
            });
            ui.horizontal(|ui| {
                ui.label("Phase delay compensation:");
                for (curve, ms) in self.curves.iter().zip(&mut self.compensation_ms) {
                    ui.label(&curve.label);
                    let drag = egui::DragValue::new(ms)
                        .speed(0.01)
                        .range(0.0..=PERIOD as f64 / SAMPLE_RATE * 1000.0)
                        .fixed_decimals(2)
                        .suffix(" ms");
                    ui.add(drag);
                }
                let find = ui.add_enabled(self.measurement.is_some(), egui::Button::new("Find"));
                let find = find.on_hover_text(
                    "Take each mic's delay from the peak of its impulse response. \
                     Find it once, then keep it, so changes show up in the phase.",
                );
                if find.clicked()
                    && let Some(measurement) = &self.measurement
                {
                    self.compensation_ms = find_delays_ms(&measurement.response);
                }
            });
            ui.horizontal(|ui| {
                for (curve, peak) in self.curves.iter().zip(&self.peak_db) {
                    let text = format!("{}: peak {peak:.1} dBFS", curve.label);
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
            // held first, so the live curves stay on top
            let all_curves = || self.held.iter().flat_map(|h| &h.curves).chain(&self.curves);
            // curves hidden in the magnitude plot's legend are hidden in the phase plot too
            let hidden = PlotMemory::load(ui.ctx(), egui::Id::unique(MAGNITUDE_PLOT))
                .map(|memory| memory.hidden_items)
                .unwrap_or_default();

            // fixed bounds, so the view doesn't jump with every update; double-click resets
            frequency_plot("phase")
                .height(ui.available_height() * 0.35)
                .default_y_bounds(-180.0, 180.0)
                // the phase always shows the full turn; only the frequency zooms and pans
                .allow_zoom([true, false])
                .allow_drag([true, false])
                .allow_scroll([true, false])
                .allow_axis_zoom_drag([true, false])
                .allow_boxed_zoom(false)
                .y_axis_label("Phase (°)")
                .y_grid_spacer(uniform_grid_spacer(|_| [15.0, 45.0, 90.0]))
                .y_axis_formatter(|mark, _| format!("{:.0}°", mark.value))
                .label_formatter(|pos| hover_label(pos, "°"))
                .show(ui, |plot_ui| {
                    plot_ui.set_plot_bounds_y(-180.0..=180.0);
                    for curve in all_curves() {
                        if hidden.contains(&ItemId::new(&curve.label)) {
                            continue;
                        }
                        for piece in curve.phase.iter().filter(|piece| piece.len() > 1) {
                            let line = Line::new(&curve.label, PlotPoints::Borrowed(piece));
                            plot_ui.line(line.color(curve.color));
                        }
                    }
                });

            frequency_plot(MAGNITUDE_PLOT)
                .default_y_bounds(-60.0, 20.0)
                .x_axis_label("Frequency")
                .y_axis_label("Magnitude (dB)")
                .label_formatter(|pos| hover_label(pos, "dB"))
                // a curve's faded pieces must not take the color out of its legend entry
                .legend(Legend::default().color_conflict_handling(ColorConflictHandling::PickFirst))
                .show(ui, |plot_ui| {
                    if self.fit_y && !self.curves.is_empty() {
                        let (lo, hi) = all_curves()
                            .flat_map(|curve| &curve.magnitude)
                            .flat_map(|(_, piece)| piece.iter().map(|p| p.y))
                            .fold((f64::MAX, f64::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
                        if lo <= hi {
                            plot_ui.set_plot_bounds_y(lo - 6.0..=hi + 6.0);
                            self.fit_y = false;
                        }
                    }
                    for curve in all_curves() {
                        // trusted pieces first: the legend takes the first piece's color
                        let mut pieces: Vec<_> = curve.magnitude.iter().collect();
                        pieces.sort_by_key(|(good, _)| !good);
                        for (good, piece) in pieces {
                            let color = if *good {
                                curve.color
                            } else {
                                curve.color.gamma_multiply(0.3)
                            };
                            let line = Line::new(&curve.label, PlotPoints::Borrowed(piece));
                            plot_ui.line(line.color(color));
                        }
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
