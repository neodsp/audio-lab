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

/// Points per octave that magnitudes are drawn at on a log axis.
const POINTS_PER_OCTAVE: f64 = 96.0;

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
    series: Vec<(String, Vec<PlotPoint>)>,
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

/// A label and points per channel, and the frequency range they cover on the plot's x axis.
type Series = (Vec<(String, Vec<PlotPoint>)>, Option<(f64, f64)>);

/// The curves of `signals` for `options.value`.
fn series(signals: &[&FreqSignal], options: FreqPlotOptions) -> Series {
    let in_range = |f: f64| {
        (!options.log_freq || f > 0.0)
            && options
                .freq_range
                .is_none_or(|(low, high)| (low..=high).contains(&f))
    };
    // on a log axis, magnitudes are drawn at points evenly spaced on the axis, so they look
    // the same at the low end, where the bins lie far apart, as at the top, where many bins
    // share a pixel. A phase can't be interpolated across its wraps, so it keeps the bins.
    let resample =
        options.log_freq && matches!(options.value, FreqValue::Magnitude | FreqValue::MagnitudeDb);

    let mut series = Vec::new();
    let mut shown = (f64::INFINITY, f64::NEG_INFINITY);
    for (signal_index, signal) in signals.iter().enumerate() {
        let freq_bins = signal.freq_bins();
        let derived = match options.value {
            FreqValue::Magnitude => signal.to_magnitude(),
            FreqValue::MagnitudeDb => signal.to_magnitude_db(),
            FreqValue::PhaseRadians | FreqValue::PhaseDegrees => signal.to_phase(),
        };
        let display_freqs = resample.then(|| log_spaced(&freq_bins.view(), options.freq_range));
        let num_channels = signal.num_channels();
        for channel_index in 0..num_channels {
            let label = crate::plot::legend::build_channel_label(
                signal.channel_label(channel_index),
                signal.comment(),
                signal_index,
                channel_index,
                num_channels,
            );
            let values: Vec<f64> = derived
                .channel(channel_index)
                .iter()
                .map(|&value| match options.value {
                    FreqValue::MagnitudeDb => value.max(DB_FLOOR),
                    FreqValue::PhaseDegrees => value.to_degrees(),
                    _ => value,
                })
                .collect();
            let points: Vec<PlotPoint> = match &display_freqs {
                Some(display_freqs) => {
                    let bin_width = freq_bins.get(1).map_or(1.0, |f| f - freq_bins[0]);
                    display_freqs
                        .iter()
                        .map(|&f| {
                            let position = (f - freq_bins[0]) / bin_width;
                            PlotPoint::new(f.log10(), interpolate_monotone(&values, position))
                        })
                        .collect()
                }
                None => freq_bins
                    .iter()
                    .zip(&values)
                    .filter(|&(&f, _)| in_range(f))
                    .map(|(&f, &y)| PlotPoint::new(if options.log_freq { f.log10() } else { f }, y))
                    .collect(),
            };
            if let (Some(first), Some(last)) = (points.first(), points.last()) {
                shown = (shown.0.min(first.x), shown.1.max(last.x));
            }
            series.push((label, points));
        }
    }
    (series, (shown.0 < shown.1).then_some(shown))
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

/// Frequencies evenly spaced on a log axis, [`POINTS_PER_OCTAVE`] per octave, from the lowest
/// bin above 0 Hz (or the range's start) to the highest bin (or the range's end).
fn log_spaced(freq_bins: &ndarray::ArrayView1<f64>, range: Option<(f64, f64)>) -> Vec<f64> {
    let lowest_bin = freq_bins.iter().copied().find(|&f| f > 0.0);
    let highest_bin = freq_bins.last().copied();
    let (Some(lowest_bin), Some(highest_bin)) = (lowest_bin, highest_bin) else {
        return Vec::new();
    };
    let (low, high) = match range {
        Some((low, high)) => (low.max(lowest_bin), high.min(highest_bin)),
        None => (lowest_bin, highest_bin),
    };
    if low >= high {
        return Vec::new();
    }
    let num_points = (POINTS_PER_OCTAVE * (high / low).log2()).ceil() as usize + 1;
    (0..num_points)
        .map(|i| (low * 2f64.powf(i as f64 / POINTS_PER_OCTAVE)).min(high))
        .collect()
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
                for (label, points) in &self.series {
                    plot_ui.line(Line::new(label.clone(), PlotPoints::Borrowed(points)));
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
    /// Label, color and points per channel.
    magnitude: Vec<(String, egui::Color32, Vec<PlotPoint>)>,
    /// Label, color and the pieces between the phase's wraps per channel.
    phase: Vec<(String, egui::Color32, Vec<Vec<PlotPoint>>)>,
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
        let magnitude_options = FreqPlotOptions {
            value: FreqValue::MagnitudeDb,
            ..self.options
        };
        let (magnitude, x_bounds) = series(signals, magnitude_options);
        let phase = phase_series(signals, self.options);
        // the same color for a channel in both plots
        let color = |index: usize| {
            let golden_ratio = (5f32.sqrt() - 1.0) / 2.0;
            egui::epaint::Hsva::new(index as f32 * golden_ratio, 0.85, 0.5, 1.0).into()
        };
        self.phase = phase
            .into_iter()
            .zip(&magnitude)
            .enumerate()
            .map(|(index, (pieces, (label, _)))| (label.clone(), color(index), pieces))
            .collect();
        self.magnitude = magnitude
            .into_iter()
            .enumerate()
            .map(|(index, (label, points))| (label, color(index), points))
            .collect();
        x_bounds
    }
}

/// The phase in degrees of every channel, wrapped to ±180° and split into pieces at the wraps.
/// On a log axis it is drawn at evenly spaced points like the magnitude, interpolated as
/// complex values so the wraps don't get in the way.
fn phase_series(signals: &[&FreqSignal], options: FreqPlotOptions) -> Vec<Vec<Vec<PlotPoint>>> {
    let mut series = Vec::new();
    for signal in signals {
        let freq_bins = signal.freq_bins();
        let display_freqs = options
            .log_freq
            .then(|| log_spaced(&freq_bins.view(), options.freq_range));
        for channel in signal.channel_iter() {
            let points: Vec<PlotPoint> = match &display_freqs {
                Some(display_freqs) => {
                    let re: Vec<f64> = channel.iter().map(|c| c.re).collect();
                    let im: Vec<f64> = channel.iter().map(|c| c.im).collect();
                    let bin_width = freq_bins.get(1).map_or(1.0, |f| f - freq_bins[0]);
                    display_freqs
                        .iter()
                        .map(|&f| {
                            let position = (f - freq_bins[0]) / bin_width;
                            let phase = interpolate_monotone(&im, position)
                                .atan2(interpolate_monotone(&re, position));
                            PlotPoint::new(f.log10(), phase.to_degrees())
                        })
                        .collect()
                }
                None => freq_bins
                    .iter()
                    .zip(channel.iter())
                    .filter(|&(&f, _)| {
                        options
                            .freq_range
                            .is_none_or(|(low, high)| (low..=high).contains(&f))
                    })
                    .map(|(&f, c)| PlotPoint::new(f, c.arg().to_degrees()))
                    .collect(),
            };
            series.push(split_at_wraps(&points));
        }
    }
    series
}

/// Splits a wrapped phase in degrees into pieces that each run to the edge where the phase
/// wraps and continue from the other edge, so it looks continuous instead of having a line
/// across the plot at every wrap.
fn split_at_wraps(points: &[PlotPoint]) -> Vec<Vec<PlotPoint>> {
    let mut pieces = vec![Vec::new()];
    for (index, &point) in points.iter().enumerate() {
        if let Some(&last) = index.checked_sub(1).map(|i| &points[i]) {
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
                for (label, color, pieces) in &self.phase {
                    if hidden.contains(&egui_plot::ItemId::new(label)) {
                        continue;
                    }
                    for piece in pieces.iter().filter(|piece| piece.len() > 1) {
                        let line = Line::new(label, PlotPoints::Borrowed(piece));
                        plot_ui.line(line.color(*color));
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
                for (label, color, points) in &self.magnitude {
                    let line = Line::new(label, PlotPoints::Borrowed(points));
                    plot_ui.line(line.color(*color));
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
    fn log_axis_draws_magnitudes_evenly_spaced_within_the_range() {
        let signal = make_signal();
        let options = FreqPlotOptions {
            freq_range: Some((20.0, 20_000.0)),
            ..Default::default()
        };
        let plot = FreqSignalPlot::new(&[&signal], "test", options);
        let points = &plot.series[0].1;
        let steps: Vec<f64> = points.windows(2).map(|w| w[1].x - w[0].x).collect();
        let step = 2f64.log10() / POINTS_PER_OCTAVE;
        assert!(
            steps[..steps.len() - 1]
                .iter()
                .all(|s| (s - step).abs() < 1e-9)
        );
        assert!((points[0].x - 20f64.log10()).abs() < 1e-9);
        assert!((points.last().unwrap().x - 20_000f64.log10()).abs() < 1e-9);
        let (low, high) = plot.x_bounds.unwrap();
        assert!((low - 20f64.log10()).abs() < 1e-9 && (high - 20_000f64.log10()).abs() < 1e-9);
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
        let pieces = split_at_wraps(&points);
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
        for ((m_label, m_color, _), (p_label, p_color, _)) in plot.magnitude.iter().zip(&plot.phase)
        {
            assert_eq!(m_label, p_label);
            assert_eq!(m_color, p_color);
        }
        assert_ne!(plot.magnitude[0].1, plot.magnitude[1].1);
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
