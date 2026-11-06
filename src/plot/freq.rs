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
}

impl Default for FreqPlotOptions {
    fn default() -> Self {
        Self {
            log_freq: true,
            value: FreqValue::MagnitudeDb,
            freq_range: None,
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
    save: SavePlotState,
}

impl FreqSignalPlot {
    pub(crate) fn new(signals: &[&FreqSignal], title: &str, options: FreqPlotOptions) -> Self {
        let in_range = |f: f64| {
            (!options.log_freq || f > 0.0)
                && options
                    .freq_range
                    .is_none_or(|(low, high)| (low..=high).contains(&f))
        };
        // on a log axis, magnitudes are drawn at points evenly spaced on the axis, so they look
        // the same at the low end, where the bins lie far apart, as at the top, where many bins
        // share a pixel. A phase can't be interpolated across its wraps, so it keeps the bins.
        let resample = options.log_freq
            && matches!(options.value, FreqValue::Magnitude | FreqValue::MagnitudeDb);

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
                        .map(|(&f, &y)| {
                            PlotPoint::new(if options.log_freq { f.log10() } else { f }, y)
                        })
                        .collect(),
                };
                if let (Some(first), Some(last)) = (points.first(), points.last()) {
                    shown = (shown.0.min(first.x), shown.1.max(last.x));
                }
                series.push((label, points));
            }
        }
        Self {
            series,
            options,
            x_bounds: (shown.0 < shown.1).then_some(shown),
            save: SavePlotState::new(title),
        }
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
        });
        egui::CentralPanel::default().show(ui, |ui| {
            let mut plot = Plot::new("freq_signal")
                .x_axis_label("Frequency (Hz)")
                .y_axis_label(self.options.value.y_label())
                .legend(Legend::default());
            // exactly the data, without the margins auto-fitting adds
            if let Some((low, high)) = self.x_bounds {
                plot = plot.default_x_bounds(low, high);
            }

            if self.options.log_freq {
                let value = self.options.value;
                plot = plot
                    .x_grid_spacer(spectrum_grid_spacer)
                    .x_axis_formatter(|mark, _range| {
                        let hz = 10f64.powf(mark.value);
                        if hz >= 1000.0 {
                            format!("{:.0} kHz", hz / 1000.0)
                        } else {
                            format!("{hz:.0} Hz")
                        }
                    })
                    .label_formatter(move |pos| {
                        let (name, point) = match pos {
                            egui_plot::HoverPosition::NearDataPoint {
                                plot_name,
                                position,
                                ..
                            } => (*plot_name, position),
                            egui_plot::HoverPosition::Elsewhere { position } => ("", position),
                        };
                        let hz = 10f64.powf(point.x);
                        let freq_str = if hz >= 1000.0 {
                            format!("{:.1} kHz", hz / 1000.0)
                        } else {
                            format!("{hz:.1} Hz")
                        };
                        let y_str = value.format_y(point.y);
                        Some(if name.is_empty() {
                            format!("{freq_str}\n{y_str}")
                        } else {
                            format!("{name}\n{freq_str}\n{y_str}")
                        })
                    });
            }

            plot.show(ui, |plot_ui| {
                for (label, points) in &self.series {
                    plot_ui.line(Line::new(label.clone(), PlotPoints::Borrowed(points)));
                }
            });
        });
        self.save.handle_screenshot(ui.ctx());
    }
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
