use audio_signal::{math::gain_to_db, signal::freq_signal::FreqSignal};
use eframe::egui;
use egui_plot::{GridInput, GridMark, Legend, Line, Plot, PlotPoint, PlotPoints};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

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
}

impl Default for FreqPlotOptions {
    fn default() -> Self {
        Self {
            log_freq: true,
            value: FreqValue::MagnitudeDb,
        }
    }
}

impl FreqValue {
    fn value(&self, magnitude: f64, phase_radians: f64) -> f64 {
        match self {
            FreqValue::MagnitudeDb => gain_to_db(magnitude).max(DB_FLOOR),
            FreqValue::Magnitude => magnitude,
            FreqValue::PhaseRadians => phase_radians,
            FreqValue::PhaseDegrees => phase_radians.to_degrees(),
        }
    }

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
    channels: Vec<Vec<PlotPoint>>,
    options: FreqPlotOptions,
    save: SavePlotState,
}

impl FreqSignalPlot {
    pub(crate) fn new(signal: &FreqSignal, title: &str, options: FreqPlotOptions) -> Self {
        let freq_bins = signal.freq_bins();
        let derived = match options.value {
            FreqValue::Magnitude => Some(signal.to_magnitude()),
            FreqValue::MagnitudeDb => Some(signal.to_magnitude_db()),
            FreqValue::PhaseRadians | FreqValue::PhaseDegrees => Some(signal.to_phase()),
        };
        let channels = signal
            .channel_iter()
            .enumerate()
            .map(|(ch_index, ch)| {
                let derived_channel = derived.as_ref().map(|data| data.channel(ch_index));
                freq_bins
                    .iter()
                    .zip(ch.iter().enumerate())
                    .filter(|&(&f, _)| !options.log_freq || f > 0.0)
                    .map(|(&f, (bin_index, c))| {
                        let x = if options.log_freq { f.log10() } else { f };
                        let y = if let Some(channel) = &derived_channel {
                            let value = channel[bin_index];
                            if matches!(options.value, FreqValue::PhaseDegrees) {
                                value.to_degrees()
                            } else {
                                value
                            }
                        } else {
                            options.value.value(c.norm(), c.arg())
                        };
                        PlotPoint::new(x, y)
                    })
                    .collect()
            })
            .collect();
        Self {
            channels,
            options,
            save: SavePlotState::new(title),
        }
    }
}

impl eframe::App for FreqSignalPlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            self.save.show_panel(ui);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut plot = Plot::new("freq_signal")
                .x_axis_label("Frequency (Hz)")
                .y_axis_label(self.options.value.y_label())
                .legend(Legend::default());

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
                    .label_formatter(move |name, point| {
                        let hz = 10f64.powf(point.x);
                        let freq_str = if hz >= 1000.0 {
                            format!("{:.1} kHz", hz / 1000.0)
                        } else {
                            format!("{hz:.1} Hz")
                        };
                        let y_str = value.format_y(point.y);
                        if name.is_empty() {
                            format!("{freq_str}\n{y_str}")
                        } else {
                            format!("{name}\n{freq_str}\n{y_str}")
                        }
                    });
            }

            plot.show(ui, |plot_ui| {
                for (i, points) in self.channels.iter().enumerate() {
                    plot_ui.line(Line::new(
                        format!("Channel {i}"),
                        PlotPoints::Owned(points.clone()),
                    ));
                }
            });
        });
        self.save.handle_screenshot(ctx);
    }
}

pub fn show_freq_signal(
    title: &str,
    signal: &FreqSignal,
    options: FreqPlotOptions,
) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(FreqSignalPlot::new(signal, title, options)))),
    )?)
}

#[cfg(test)]
mod tests {
    use audio_signal::{ndarray::arr2, signal::time_signal::TimeSignal};

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
    #[ignore]
    fn test_show_freq_signal_log_db() {
        show_freq_signal(
            "Freq Signal (Log, dB)",
            &make_signal(),
            FreqPlotOptions::default(),
        )
        .unwrap();
    }

    #[test]
    #[ignore]
    fn test_show_freq_signal_phase() {
        show_freq_signal(
            "Freq Signal (Phase)",
            &make_signal(),
            FreqPlotOptions {
                log_freq: true,
                value: FreqValue::PhaseDegrees,
            },
        )
        .unwrap();
    }

    #[test]
    #[ignore]
    fn test_show_freq_signal_phase_degrees() {
        show_freq_signal(
            "Freq Signal (Phase, Degrees)",
            &make_signal(),
            FreqPlotOptions {
                log_freq: true,
                value: FreqValue::PhaseDegrees,
            },
        )
        .unwrap();
    }
}
