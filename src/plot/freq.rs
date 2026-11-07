use num::complex::Complex64;

use crate::dsp::{
    FractionalOctaveSmoothingConfig, FractionalOctaveSmoothingMode, smooth_fractional_octave,
};
use crate::signal::FreqSignal;
use eframe::egui;
use egui_plot::{GridInput, GridMark, Legend, Line, Plot, PlotPoint, PlotPoints};

use crate::plot::native_options_any_thread;
use crate::plot::save::SavePlotState;

const DB_FLOOR: f64 = -120.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreqValue {
    Magnitude,
    MagnitudeDb,
    PhaseRadians,
    PhaseDegrees,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreqPlotOptions {
    pub log_freq: bool,
    pub value: FreqValue,
    /// Only show this range in Hz, e.g. the range a sweep covered. `None` for everything above
    /// 0 Hz (on a log axis) up to half the sample rate.
    pub freq_range: Option<(f64, f64)>,
    /// Fractional-octave smoothing to start with, `Some(3.0)` for 1/3 octave. With `Some`, the
    /// window has controls to change it or switch it off, and smooths the signals itself: pass
    /// them unsmoothed. The magnitude is smoothed on its own, the phase as complex values.
    pub smoothing: Option<f64>,
}

impl Default for FreqPlotOptions {
    fn default() -> Self {
        Self {
            log_freq: true,
            value: FreqValue::MagnitudeDb,
            freq_range: None,
            smoothing: None,
        }
    }
}

/// Points drawn per screen point along a log frequency axis. The interpolation between the bins
/// is smooth, so more wouldn't look different.
const POINTS_PER_SCREEN_POINT: f64 = 1.0;

impl FreqValue {
    fn y_label(&self) -> &'static str {
        match self {
            FreqValue::MagnitudeDb => "Magnitude (dB)",
            FreqValue::Magnitude => "Magnitude",
            FreqValue::PhaseRadians => "Phase (rad)",
            FreqValue::PhaseDegrees => "Phase (deg)",
        }
    }

    fn format_y(&self, y: f64) -> String {
        match self {
            FreqValue::MagnitudeDb => format!("{y:.1} dB"),
            FreqValue::Magnitude => format!("{y:.4}"),
            FreqValue::PhaseRadians => format!("{y:.4} rad"),
            FreqValue::PhaseDegrees => format!("{y:.2} deg"),
        }
    }
}

/// Returns spectrum-style grid marks in log10 space.
/// Three tiers:
///   - Major (step 1.0):  decade boundaries — 10, 100, 1k, 10k, 100k
///   - Medium (step 0.3): ×2 and ×5 per decade — 20, 50, 200, 500, …
///   - Minor (step 0.1):  remaining integers per decade — 30, 40, 60, 70, …
fn spectrum_grid_spacer(input: GridInput) -> Vec<GridMark> {
    let (lo, hi) = (input.bounds.0, input.bounds.1);
    let mut marks = Vec::new();

    let start_exp = lo.floor() as i32 - 1;
    let end_exp = hi.ceil() as i32 + 1;

    for exp in start_exp..=end_exp {
        let decade = 10f64.powi(exp);
        for mult in 1..10 {
            let hz = decade * mult as f64;
            let log_pos = hz.log10();
            if log_pos < lo || log_pos > hi {
                continue;
            }
            let step_size = match mult {
                1 => 1.0,
                2 | 5 => 0.3,
                _ => 0.1,
            };
            marks.push(GridMark {
                value: log_pos,
                step_size,
            });
        }
    }
    marks
}

pub(crate) struct FreqSignalPlot {
    series: Vec<Series>,
    options: FreqPlotOptions,
    /// The frequency range shown at the start and after a double-click, on the plot's x axis.
    x_bounds: Option<(f64, f64)>,
    smoothing: Option<SmoothingControl>,
    save: SavePlotState,
}

impl FreqSignalPlot {
    pub(crate) fn new(signals: &[&FreqSignal], title: &str, options: FreqPlotOptions) -> Self {
        let smoothing = SmoothingControl::new(signals, options);
        let smoothed = smoothing.as_ref().map(SmoothingControl::smoothed);
        let (series, x_bounds) = series(&plotted(signals, &smoothed), options);
        Self {
            series,
            options,
            x_bounds,
            smoothing,
            save: SavePlotState::new(title),
        }
    }
}

/// `smoothed` if there is a smoothing control, else `signals`.
fn plotted<'a>(
    signals: &[&'a FreqSignal],
    smoothed: &'a Option<Vec<FreqSignal>>,
) -> Vec<&'a FreqSignal> {
    match smoothed {
        Some(smoothed) => smoothed.iter().collect(),
        None => signals.to_vec(),
    }
}

/// One channel's curve with its legend label and color.
struct Series {
    label: String,
    color: egui::Color32,
    curve: Curve,
}

/// The curve of every channel of `signals` for `options.value`, and the frequency range they
/// cover on the plot's x axis. Each channel gets its own color, the same for the same channel
/// whatever the value, so a magnitude and a phase plot match.
fn series(signals: &[&FreqSignal], options: FreqPlotOptions) -> (Vec<Series>, Option<(f64, f64)>) {
    let mut series = Vec::new();
    let mut shown = (f64::INFINITY, f64::NEG_INFINITY);
    for (signal_index, signal) in signals.iter().enumerate() {
        let num_channels = signal.num_channels();
        for channel_index in 0..num_channels {
            let label = crate::plot::legend::build_channel_label(
                signal.channel_label(channel_index),
                signal.comment(),
                signal_index,
                channel_index,
                num_channels,
            );
            let curve = Curve::new(signal, channel_index, options);
            if let Some((low, high)) = curve.x_range() {
                shown = (shown.0.min(low), shown.1.max(high));
            }
            series.push(Series {
                label,
                color: color(series.len()),
                curve,
            });
        }
    }
    (series, (shown.0 < shown.1).then_some(shown))
}

/// The `index`th of a series of well distinguishable colors, as egui_plot picks them.
fn color(index: usize) -> egui::Color32 {
    let golden_ratio = (5f32.sqrt() - 1.0) / 2.0;
    egui::epaint::Hsva::new(index as f32 * golden_ratio, 0.85, 0.5, 1.0).into()
}

/// A channel's values, ready to draw.
enum Curve {
    /// Drawn as they are: the bins on a linear axis, in pieces between the phase's wraps.
    Points(Vec<Vec<PlotPoint>>),
    /// Drawn at the resolution of the screen on a log axis: interpolated between the bins at
    /// points evenly spaced on the axis, for the part that is in view. So the curves look the
    /// same at the low end, where the bins lie far apart, as at the top, where many bins share
    /// a pixel, and stay smooth however far you zoom in.
    Log(LogCurve),
}

struct LogCurve {
    first_bin: f64,
    bin_width: f64,
    /// Where there is data to draw, on the log axis.
    log_range: (f64, f64),
    values: LogValues,
}

enum LogValues {
    /// Values per bin, e.g. the magnitude, interpolated without overshoot.
    Real(Vec<f64>),
    /// The complex values per bin, drawn as their phase wrapped to ±`half_turn` (180 for
    /// degrees, π for radians). Complex values interpolate across the phase's wraps.
    Phase {
        re: Vec<f64>,
        im: Vec<f64>,
        half_turn: f64,
    },
}

impl Curve {
    fn new(signal: &FreqSignal, channel: usize, options: FreqPlotOptions) -> Self {
        let freq_bins = signal.freq_bins();
        let half_turn = match options.value {
            FreqValue::PhaseDegrees => Some(180.0),
            FreqValue::PhaseRadians => Some(std::f64::consts::PI),
            FreqValue::Magnitude | FreqValue::MagnitudeDb => None,
        };
        let values: Vec<f64> = match options.value {
            FreqValue::Magnitude => signal.to_magnitude().channel(channel).to_vec(),
            FreqValue::MagnitudeDb => signal
                .to_magnitude_db()
                .channel(channel)
                .iter()
                .map(|value| value.max(DB_FLOOR))
                .collect(),
            FreqValue::PhaseRadians => signal.to_phase().channel(channel).to_vec(),
            FreqValue::PhaseDegrees => signal
                .to_phase()
                .channel(channel)
                .iter()
                .map(|value| value.to_degrees())
                .collect(),
        };

        if !options.log_freq {
            let points: Vec<PlotPoint> = freq_bins
                .iter()
                .zip(&values)
                .filter(|&(&f, _)| {
                    options
                        .freq_range
                        .is_none_or(|(low, high)| (low..=high).contains(&f))
                })
                .map(|(&f, &y)| PlotPoint::new(f, y))
                .collect();
            return Self::Points(match half_turn {
                Some(half_turn) => split_at_wraps(&points, half_turn),
                None => vec![points],
            });
        }

        let first_bin = freq_bins.first().copied().unwrap_or(0.0);
        // 0 Hz has no place on a log axis
        let lowest_bin = freq_bins.iter().copied().find(|&f| f > 0.0);
        let highest_bin = freq_bins.last().copied();
        let log_range = match (lowest_bin, highest_bin) {
            (Some(lowest), Some(highest)) => {
                let (low, high) = match options.freq_range {
                    Some((low, high)) => (low.max(lowest), high.min(highest)),
                    None => (lowest, highest),
                };
                (low.log10(), high.log10())
            }
            _ => (0.0, 0.0),
        };
        let channel_values = signal.channel(channel);
        Self::Log(LogCurve {
            first_bin,
            bin_width: freq_bins.get(1).map_or(1.0, |f| f - first_bin),
            log_range,
            values: match half_turn {
                Some(half_turn) => LogValues::Phase {
                    re: channel_values.iter().map(|c| c.re).collect(),
                    im: channel_values.iter().map(|c| c.im).collect(),
                    half_turn,
                },
                None => LogValues::Real(values),
            },
        })
    }

    /// The frequency range it covers on the plot's x axis.
    fn x_range(&self) -> Option<(f64, f64)> {
        let (low, high) = match self {
            Self::Points(pieces) => {
                let mut points = pieces.iter().flatten();
                let first = points.next()?.x;
                (first, points.last().map_or(first, |p| p.x))
            }
            Self::Log(curve) => curve.log_range,
        };
        (low < high).then_some((low, high))
    }

    /// Adds the curve to the plot, for the plot's current view.
    fn draw<'a>(
        &'a self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        label: &'a str,
        color: egui::Color32,
    ) {
        match self {
            Self::Points(pieces) => {
                for piece in pieces.iter().filter(|piece| piece.len() > 1) {
                    plot_ui.line(Line::new(label, PlotPoints::Borrowed(piece)).color(color));
                }
            }
            Self::Log(curve) => {
                let positions = positions_in_view(plot_ui, curve.log_range);
                for piece in curve.pieces(&positions) {
                    if piece.len() > 1 {
                        plot_ui.line(Line::new(label, PlotPoints::Owned(piece)).color(color));
                    }
                }
            }
        }
    }
}

impl LogCurve {
    /// The curve at `positions` on the log axis, in pieces between the phase's wraps.
    fn pieces(&self, positions: &[f64]) -> Vec<Vec<PlotPoint>> {
        let bin = |x: f64| (10f64.powf(x) - self.first_bin) / self.bin_width;
        match &self.values {
            LogValues::Real(values) => vec![
                positions
                    .iter()
                    .map(|&x| PlotPoint::new(x, interpolate_monotone(values, bin(x))))
                    .collect(),
            ],
            LogValues::Phase { re, im, half_turn } => {
                let points: Vec<PlotPoint> = positions
                    .iter()
                    .map(|&x| {
                        let position = bin(x);
                        let phase = interpolate_monotone(im, position)
                            .atan2(interpolate_monotone(re, position));
                        PlotPoint::new(x, phase / std::f64::consts::PI * half_turn)
                    })
                    .collect();
                split_at_wraps(&points, *half_turn)
            }
        }
    }
}

/// Where to draw a curve on a log axis: evenly spaced, [`POINTS_PER_SCREEN_POINT`] per screen
/// point of the `visible` range, `width` screen points wide. Also half the visible width beyond
/// each side, so the curve doesn't end at the old edges in the frame in which the view is
/// panned or zoomed (the plot only knows the last frame's view). Only within `log_range`,
/// where there is data.
fn draw_positions(visible: Option<(f64, f64)>, width: f64, log_range: (f64, f64)) -> Vec<f64> {
    let (low, high) = log_range;
    if low >= high {
        return Vec::new();
    }
    let (start, end, step) = match visible {
        Some((left, right)) if left < right && left < high && low < right && width >= 1.0 => {
            let margin = 0.5 * (right - left);
            let step = (right - left) / (width * POINTS_PER_SCREEN_POINT);
            (low.max(left - margin), high.min(right + margin), step)
        }
        // nothing to go by yet, e.g. in the first frame
        _ => (low, high, (high - low) / 2000.0),
    };
    let num_points = ((end - start) / step).ceil() as usize + 1;
    (0..num_points)
        .map(|i| (start + i as f64 * step).min(end))
        .collect()
}

/// [`draw_positions`] for the plot's current view.
fn positions_in_view(plot_ui: &egui_plot::PlotUi, log_range: (f64, f64)) -> Vec<f64> {
    let bounds = plot_ui.plot_bounds();
    let visible = bounds
        .is_valid_x()
        .then(|| (bounds.min()[0], bounds.max()[0]));
    let width = f64::from(plot_ui.transform().frame().width());
    draw_positions(visible, width, log_range)
}

/// The signals as given and the smoothing chosen in the window.
struct SmoothingControl {
    signals: Vec<FreqSignal>,
    enabled: bool,
    num_fractions: f64,
}

impl SmoothingControl {
    /// `None` without `options.smoothing`: then the window has no smoothing controls.
    fn new(signals: &[&FreqSignal], options: FreqPlotOptions) -> Option<Self> {
        options.smoothing.map(|num_fractions| Self {
            signals: signals.iter().map(|&signal| signal.clone()).collect(),
            enabled: true,
            num_fractions,
        })
    }

    /// The signals with the chosen smoothing. A signal too short for it stays as it is.
    fn smoothed(&self) -> Vec<FreqSignal> {
        if !self.enabled {
            return self.signals.clone();
        }
        let config =
            |mode| FractionalOctaveSmoothingConfig::new(self.num_fractions).with_mode(mode);
        self.signals
            .iter()
            .map(|signal| {
                let magnitude = smooth_fractional_octave(
                    signal,
                    &config(FractionalOctaveSmoothingMode::MagnitudeZeroPhase),
                );
                let phase = smooth_fractional_octave(
                    signal,
                    &config(FractionalOctaveSmoothingMode::Complex),
                );
                let (Ok((magnitude, _)), Ok((mut phase, _))) = (magnitude, phase) else {
                    return signal.clone();
                };
                phase
                    .freq_data_mut()
                    .zip_mut_with(&magnitude.freq_data(), |p, m| {
                        *p = Complex64::from_polar(m.norm(), p.arg());
                    });
                phase
            })
            .collect()
    }

    /// The checkbox and slider. Returns whether the smoothing changed.
    fn show(&mut self, ui: &mut egui::Ui) -> bool {
        ui.horizontal(|ui| {
            let enabled = ui.checkbox(&mut self.enabled, "Smoothing").changed();
            let slider = egui::Slider::new(&mut self.num_fractions, 1.0..=48.0)
                .text("1/N octave")
                .logarithmic(true);
            let fractions = ui.add_enabled(self.enabled, slider).changed();
            enabled || fractions
        })
        .inner
    }
}

/// `values` at the fractional index `position`, by monotone cubic interpolation: a smooth curve
/// through the values that never goes above or below its neighbors, so bins that lie far apart
/// don't turn into lobes and notches that aren't in the data.
pub(crate) fn interpolate_monotone(values: &[f64], position: f64) -> f64 {
    let Some(last) = values.len().checked_sub(1) else {
        return f64::NAN;
    };
    let index = (position.max(0.0).floor() as usize).min(last);
    let t = (position - index as f64).clamp(0.0, 1.0);
    let at = |i: isize| values[i.clamp(0, last as isize) as usize];
    let i = index as isize;
    let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
    // slopes at both ends: flat at a peak or dip, else the harmonic mean of the neighbors
    let slope = |before: f64, after: f64| {
        if before * after <= 0.0 {
            0.0
        } else {
            2.0 / (1.0 / before + 1.0 / after)
        }
    };
    let (m1, m2) = (slope(p1 - p0, p2 - p1), slope(p2 - p1, p3 - p2));
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * p1
        + (t3 - 2.0 * t2 + t) * m1
        + (3.0 * t2 - 2.0 * t3) * p2
        + (t3 - t2) * m2
}

impl eframe::App for FreqSignalPlot {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("controls").show(ui, |ui| {
            self.save.show_panel(ui);
            if let Some(smoothing) = &mut self.smoothing
                && smoothing.show(ui)
            {
                let smoothed = smoothing.smoothed();
                self.series = series(&smoothed.iter().collect::<Vec<_>>(), self.options).0;
            }
        });
        egui::CentralPanel::default().show(ui, |ui| {
            let value = self.options.value;
            let plot = Plot::new("freq_signal")
                .x_axis_label("Frequency (Hz)")
                .y_axis_label(value.y_label())
                .legend(Legend::default());
            let plot = frequency_axis(plot, self.options.log_freq, self.x_bounds, move |y| {
                value.format_y(y)
            });
            plot.show(ui, |plot_ui| {
                for series in &self.series {
                    series.curve.draw(plot_ui, &series.label, series.color);
                }
            });
        });
        self.save.handle_screenshot(ui.ctx());
    }
}

/// Sets up the frequency axis of `plot`: the range shown at the start and after a
/// double-click, and on a log axis the grid, the tick labels in Hz and the readout under the
/// mouse, with `format_y` for the value.
fn frequency_axis<'a>(
    mut plot: Plot<'a>,
    log_freq: bool,
    x_bounds: Option<(f64, f64)>,
    format_y: impl Fn(f64) -> String + 'a,
) -> Plot<'a> {
    // exactly the data, without the margins auto-fitting adds
    if let Some((low, high)) = x_bounds {
        plot = plot.default_x_bounds(low, high);
    }
    if !log_freq {
        return plot;
    }
    let format_hz = |hz: f64, decimals: usize| {
        if hz >= 1000.0 {
            format!("{:.*} kHz", decimals, hz / 1000.0)
        } else {
            format!("{hz:.decimals$} Hz")
        }
    };
    plot.x_grid_spacer(spectrum_grid_spacer)
        .x_axis_formatter(move |mark, _range| format_hz(10f64.powf(mark.value), 0))
        .label_formatter(move |pos| {
            let (name, point) = match pos {
                egui_plot::HoverPosition::NearDataPoint {
                    plot_name,
                    position,
                    ..
                } => (*plot_name, position),
                egui_plot::HoverPosition::Elsewhere { position } => ("", position),
            };
            let text = format!(
                "{}\n{}",
                format_hz(10f64.powf(point.x), 1),
                format_y(point.y)
            );
            Some(if name.is_empty() {
                text
            } else {
                format!("{name}\n{text}")
            })
        })
}

/// Magnitude and phase of the same signals in two plots, phase on top, like a measurement
/// program shows a transfer function.
pub(crate) struct MagnitudePhasePlot {
    magnitude: Vec<Series>,
    /// The same channels as `magnitude`, with the same labels and colors.
    phase: Vec<Series>,
    options: FreqPlotOptions,
    x_bounds: Option<(f64, f64)>,
    smoothing: Option<SmoothingControl>,
    save: SavePlotState,
}

impl MagnitudePhasePlot {
    pub(crate) fn new(signals: &[&FreqSignal], title: &str, options: FreqPlotOptions) -> Self {
        let smoothing = SmoothingControl::new(signals, options);
        let smoothed = smoothing.as_ref().map(SmoothingControl::smoothed);
        let mut plot = Self {
            magnitude: Vec::new(),
            phase: Vec::new(),
            options,
            x_bounds: None,
            smoothing: None,
            save: SavePlotState::new(title),
        };
        plot.x_bounds = plot.set_curves(&plotted(signals, &smoothed));
        plot.smoothing = smoothing;
        plot
    }

    /// Draws `signals` from now on. Returns the frequency range they cover.
    fn set_curves(&mut self, signals: &[&FreqSignal]) -> Option<(f64, f64)> {
        let with_value = |value| FreqPlotOptions {
            value,
            ..self.options
        };
        let (magnitude, x_bounds) = series(signals, with_value(FreqValue::MagnitudeDb));
        self.magnitude = magnitude;
        self.phase = series(signals, with_value(FreqValue::PhaseDegrees)).0;
        x_bounds
    }
}

/// Splits a wrapped phase (±`half_turn`, 180 for degrees) into pieces that each run to the edge
/// where the phase wraps and continue from the other edge, so it looks continuous instead of
/// having a line across the plot at every wrap.
fn split_at_wraps(points: &[PlotPoint], half_turn: f64) -> Vec<Vec<PlotPoint>> {
    let mut pieces = vec![Vec::new()];
    for (index, &point) in points.iter().enumerate() {
        if let Some(&last) = index.checked_sub(1).map(|i| &points[i]) {
            let step = point.y - last.y;
            // a jump of more than half a turn is a wrap: the phase went the short way round
            if step.abs() > half_turn {
                let unwrapped = point.y - 2.0 * half_turn * step.signum();
                let edge = if unwrapped > last.y {
                    half_turn
                } else {
                    -half_turn
                };
                let x = last.x + (point.x - last.x) * (edge - last.y) / (unwrapped - last.y);
                pieces.last_mut().unwrap().push(PlotPoint::new(x, edge));
                pieces.push(vec![PlotPoint::new(x, -edge)]);
            }
        }
        pieces.last_mut().unwrap().push(point);
    }
    pieces
}

impl eframe::App for MagnitudePhasePlot {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("controls").show(ui, |ui| {
            self.save.show_panel(ui);
            if let Some(smoothing) = &mut self.smoothing
                && smoothing.show(ui)
            {
                let smoothed = smoothing.smoothed();
                self.set_curves(&smoothed.iter().collect::<Vec<_>>());
            }
        });
        egui::CentralPanel::default().show(ui, |ui| {
            let link = egui::Id::unique("audio_lab_magnitude_phase");
            let magnitude_id = egui::Id::unique("audio_lab_magnitude");
            // curves hidden in the magnitude plot's legend are hidden in the phase plot too
            let hidden = egui_plot::PlotMemory::load(ui.ctx(), magnitude_id)
                .map(|memory| memory.hidden_items)
                .unwrap_or_default();

            let phase = Plot::new("phase")
                .height(ui.available_height() * 0.35)
                .y_axis_label("Phase (°)")
                .link_axis(link, [true, false])
                .link_cursor(link, [true, false])
                .default_y_bounds(-180.0, 180.0)
                // the phase always shows the full turn; only the frequency zooms and pans
                .allow_zoom([true, false])
                .allow_drag([true, false])
                .allow_scroll([true, false])
                .allow_axis_zoom_drag([true, false])
                .allow_boxed_zoom(false)
                .y_grid_spacer(egui_plot::uniform_grid_spacer(|_| [15.0, 45.0, 90.0]))
                .y_axis_formatter(|mark, _| format!("{:.0}°", mark.value));
            frequency_axis(phase, self.options.log_freq, self.x_bounds, |y| {
                format!("{y:.1}°")
            })
            .show(ui, |plot_ui| {
                plot_ui.set_plot_bounds_y(-180.0..=180.0);
                for series in &self.phase {
                    if !hidden.contains(&egui_plot::ItemId::new(&series.label)) {
                        series.curve.draw(plot_ui, &series.label, series.color);
                    }
                }
            });

            let magnitude = Plot::new("magnitude")
                .id(magnitude_id)
                .x_axis_label("Frequency (Hz)")
                .y_axis_label(FreqValue::MagnitudeDb.y_label())
                .link_axis(link, [true, false])
                .link_cursor(link, [true, false])
                .legend(Legend::default());
            frequency_axis(magnitude, self.options.log_freq, self.x_bounds, |y| {
                FreqValue::MagnitudeDb.format_y(y)
            })
            .show(ui, |plot_ui| {
                for series in &self.magnitude {
                    series.curve.draw(plot_ui, &series.label, series.color);
                }
            });
        });
        self.save.handle_screenshot(ui.ctx());
    }
}

/// Shows the magnitude and phase of `signals` in one window, phase on top, e.g. measured
/// transfer functions. Both plots zoom and pan together along the frequency axis; the phase is
/// wrapped to ±180°. `options.value` is not used. Blocks until the window is closed.
///
/// Remove the delay first, e.g. up to the peak of the impulse response: a delay turns the
/// phase so quickly that it wraps every few Hz.
pub fn show_magnitude_and_phase(
    title: &str,
    signals: &[&FreqSignal],
    options: FreqPlotOptions,
) -> Result<(), crate::plot::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(MagnitudePhasePlot::new(signals, title, options)))),
    )?)
}

pub fn show_freq_with_options(
    title: &str,
    signals: &[&FreqSignal],
    options: FreqPlotOptions,
) -> Result<(), crate::plot::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(FreqSignalPlot::new(signals, title, options)))),
    )?)
}

pub fn show_freq(title: &str, signals: &[&FreqSignal]) -> Result<(), crate::plot::Error> {
    show_freq_with_options(title, signals, Default::default())
}

#[cfg(test)]
mod tests {
    use crate::signal::{TimeSignal, ndarray::arr2};

    use super::*;

    fn make_signal() -> FreqSignal {
        let sample_rate = 44100.0;
        TimeSignal::new(
            arr2(&[
                std::array::from_fn::<f64, 4096, _>(|i| {
                    (i as f64 / sample_rate * 440.0 * std::f64::consts::TAU).sin()
                }),
                std::array::from_fn::<f64, 4096, _>(|i| {
                    (i as f64 / sample_rate * 880.0 * std::f64::consts::TAU).sin()
                }),
            ]),
            sample_rate,
        )
        .unwrap()
        .into_freq()
    }

    #[test]
    fn log_axis_draws_within_the_range() {
        let signal = make_signal();
        let options = FreqPlotOptions {
            freq_range: Some((20.0, 20_000.0)),
            ..Default::default()
        };
        let plot = FreqSignalPlot::new(&[&signal], "test", options);
        let Curve::Log(curve) = &plot.series[0].curve else {
            panic!("a log axis draws at the resolution of the screen");
        };
        let (low, high) = curve.log_range;
        assert!((low - 20f64.log10()).abs() < 1e-9 && (high - 20_000f64.log10()).abs() < 1e-9);
        assert_eq!(plot.x_bounds, Some(curve.log_range));

        // drawn through the bins: at a bin's frequency, the curve is the bin's value
        let bin = 100;
        let x = signal.freq_bins()[bin].log10();
        let piece = &curve.pieces(&[x])[0];
        let expected = signal.to_magnitude_db().channel(0)[bin];
        assert!((piece[0].y - expected).abs() < 1e-9);
    }

    #[test]
    fn linear_axis_draws_the_bins() {
        let signal = make_signal();
        let options = FreqPlotOptions {
            log_freq: false,
            freq_range: Some((1000.0, 2000.0)),
            ..Default::default()
        };
        let plot = FreqSignalPlot::new(&[&signal], "test", options);
        let Curve::Points(pieces) = &plot.series[0].curve else {
            panic!("a linear axis draws the bins");
        };
        let bins: Vec<f64> = signal
            .freq_bins()
            .iter()
            .copied()
            .filter(|f| (1000.0..=2000.0).contains(f))
            .collect();
        let xs: Vec<f64> = pieces[0].iter().map(|p| p.x).collect();
        assert_eq!(xs, bins);
    }

    #[test]
    fn positions_follow_the_view_at_screen_resolution() {
        let range = (20f64.log10(), 20_000f64.log10());
        // zoomed in to 200..400 Hz on 600 screen points
        let visible = (200f64.log10(), 400f64.log10());
        let positions = draw_positions(Some(visible), 600.0, range);
        let step = positions[1] - positions[0];
        assert!((step - (visible.1 - visible.0) / 600.0).abs() < 1e-12);
        // half a view beyond each side
        let margin = 0.5 * (visible.1 - visible.0);
        assert!((positions[0] - (visible.0 - margin)).abs() < 1e-9);
        assert!((positions.last().unwrap() - (visible.1 + margin)).abs() < 1e-9);

        // zoomed out beyond the data: only where there is data, still one per screen point
        let positions = draw_positions(Some((0.0, 6.0)), 600.0, range);
        assert_eq!(positions[0], range.0);
        assert_eq!(*positions.last().unwrap(), range.1);
        assert!(positions.len() < 400);

        // no view yet: the whole range
        let positions = draw_positions(None, 0.0, range);
        assert_eq!((positions[0], *positions.last().unwrap()), range);
        assert!(draw_positions(Some(visible), 600.0, (1.0, 1.0)).is_empty());
    }

    #[test]
    fn phase_in_radians_wraps_at_pi() {
        // a delayed impulse: its phase turns steadily and wraps many times
        let mut data = ndarray::Array2::zeros((1, 4096));
        data[[0, 50]] = 1.0;
        let signal = TimeSignal::new(data, 48_000.0).unwrap().into_freq();
        let options = FreqPlotOptions {
            value: FreqValue::PhaseRadians,
            ..Default::default()
        };
        let plot = FreqSignalPlot::new(&[&signal], "test", options);
        let Curve::Log(curve) = &plot.series[0].curve else {
            panic!("log axis");
        };
        let positions = draw_positions(None, 0.0, curve.log_range);
        let pieces = curve.pieces(&positions);
        // 50 samples of delay turn the phase 25 times up to half the sample rate
        assert!(pieces.len() >= 24, "{}", pieces.len());
        let pi = std::f64::consts::PI;
        assert!(pieces.iter().flatten().all(|p| p.y.abs() <= pi + 1e-9));
    }

    #[test]
    fn monotone_interpolation_passes_through_and_never_overshoots() {
        let values = [0.0, 10.0, 0.0, -3.0, -2.9, 5.0, 5.0, 4.0, -30.0, 0.0];
        for i in 0..values.len() - 1 {
            assert_eq!(interpolate_monotone(&values, i as f64), values[i]);
            let (low, high) = (values[i].min(values[i + 1]), values[i].max(values[i + 1]));
            for k in 1..100 {
                let y = interpolate_monotone(&values, i as f64 + k as f64 / 100.0);
                assert!((low - 1e-9..=high + 1e-9).contains(&y), "{i} + {k}: {y}");
            }
        }
    }

    #[test]
    fn phase_pieces_meet_at_the_edges() {
        let points: Vec<PlotPoint> = (0..200)
            .map(|i| {
                let unwrapped = -7.0 * i as f64;
                let wrapped = (unwrapped + 180.0f64).rem_euclid(360.0) - 180.0;
                PlotPoint::new(i as f64, wrapped)
            })
            .collect();
        let pieces = split_at_wraps(&points, 180.0);
        assert_eq!(pieces.len(), 1 + (7.0 * 199.0 + 180.0) as usize / 360);
        for pair in pieces.windows(2) {
            let (end, start) = (pair[0].last().unwrap(), pair[1][0]);
            assert_eq!((end.x, end.y), (start.x, -start.y));
            assert_eq!(end.y, -180.0);
        }
        let originals = pieces
            .iter()
            .flatten()
            .filter(|p| p.x.fract() == 0.0 && p.y.abs() < 180.0);
        assert_eq!(
            originals.count(),
            points.iter().filter(|p| p.y.abs() < 180.0).count()
        );
    }

    #[test]
    fn magnitude_and_phase_share_labels_and_colors() {
        let signal = make_signal();
        let plot = MagnitudePhasePlot::new(&[&signal], "test", FreqPlotOptions::default());
        assert_eq!(plot.magnitude.len(), 2);
        assert_eq!(plot.phase.len(), 2);
        for (magnitude, phase) in plot.magnitude.iter().zip(&plot.phase) {
            assert_eq!(magnitude.label, phase.label);
            assert_eq!(magnitude.color, phase.color);
        }
        assert_ne!(plot.magnitude[0].color, plot.magnitude[1].color);
    }

    #[test]
    fn smoothing_control_smooths_and_switches_off() {
        let mut signal = make_signal();
        signal.set_comment(Some("sines"));
        let options = FreqPlotOptions {
            smoothing: Some(3.0),
            ..Default::default()
        };
        assert!(SmoothingControl::new(&[&signal], FreqPlotOptions::default()).is_none());
        let mut control = SmoothingControl::new(&[&signal], options).unwrap();

        let smoothed = control.smoothed();
        assert_eq!(smoothed[0].comment(), Some("sines"));
        assert_eq!(smoothed[0].num_freq_bins(), signal.num_freq_bins());
        // the sine's peak gets spread over its neighbors
        let peak = |s: &FreqSignal| s.channel(0).iter().map(|c| c.norm()).fold(0.0, f64::max);
        assert!(peak(&smoothed[0]) < 0.5 * peak(&signal));

        control.enabled = false;
        assert_eq!(control.smoothed()[0], signal);
    }

    #[test]
    #[ignore]
    fn test_show_magnitude_and_phase() {
        show_magnitude_and_phase("Magnitude and phase", &[&make_signal()], Default::default())
            .unwrap();
    }

    #[test]
    #[ignore]
    fn test_show_freq_signal_log_db() {
        show_freq("Freq Signal (Log, dB)", &[&make_signal()]).unwrap();
    }

    #[test]
    #[ignore]
    fn test_show_freq_signal_phase() {
        show_freq_with_options(
            "Freq Signal (Phase)",
            &[&make_signal()],
            FreqPlotOptions {
                value: FreqValue::PhaseDegrees,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    #[ignore]
    fn test_show_freq_signal_phase_degrees() {
        show_freq_with_options(
            "Freq Signal (Phase, Degrees)",
            &[&make_signal()],
            FreqPlotOptions {
                value: FreqValue::PhaseDegrees,
                ..Default::default()
            },
        )
        .unwrap();
    }
}
