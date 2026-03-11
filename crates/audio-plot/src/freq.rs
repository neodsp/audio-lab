use audio_signal::signal::freq_signal::FreqSignal;
use eframe::egui;
use egui_plot::{GridInput, GridMark, Legend, Line, Plot, PlotPoint, PlotPoints};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

const DB_FLOOR: f64 = -120.0;

pub enum PhaseUnit {
    Radians,
    Degrees,
}

pub enum FreqDisplay {
    Magnitude { db: bool },
    Phase(PhaseUnit),
}

impl FreqDisplay {
    fn value(&self, magnitude: f64, phase_radians: f64) -> f64 {
        match self {
            FreqDisplay::Magnitude { db: true } => (20.0 * magnitude.log10()).max(DB_FLOOR),
            FreqDisplay::Magnitude { db: false } => magnitude,
            FreqDisplay::Phase(PhaseUnit::Radians) => phase_radians,
            FreqDisplay::Phase(PhaseUnit::Degrees) => phase_radians.to_degrees(),
        }
    }

    fn y_label(&self) -> &'static str {
        match self {
            FreqDisplay::Magnitude { db: true } => "Magnitude (dB)",
            FreqDisplay::Magnitude { db: false } => "Magnitude",
            FreqDisplay::Phase(PhaseUnit::Radians) => "Phase (rad)",
            FreqDisplay::Phase(PhaseUnit::Degrees) => "Phase (deg)",
        }
    }

    fn format_y(&self, y: f64) -> String {
        match self {
            FreqDisplay::Magnitude { db: true } => format!("{y:.1} dB"),
            FreqDisplay::Magnitude { db: false } => format!("{y:.4}"),
            FreqDisplay::Phase(PhaseUnit::Radians) => format!("{y:.4} rad"),
            FreqDisplay::Phase(PhaseUnit::Degrees) => format!("{y:.2} deg"),
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
    log_freq: bool,
    display: FreqDisplay,
    save: SavePlotState,
}

impl FreqSignalPlot {
    pub(crate) fn new(signal: &FreqSignal, title: &str, log_freq: bool, display: FreqDisplay) -> Self {
        let freq_bins = signal.freq_bins();
        let channels = signal
            .channel_iter()
            .map(|ch| {
                freq_bins
                    .iter()
                    .zip(ch.iter())
                    .filter(|&(&f, _)| !log_freq || f > 0.0)
                    .map(|(&f, c)| {
                        let x = if log_freq { f.log10() } else { f };
                        let y = display.value(c.norm(), c.arg());
                        PlotPoint::new(x, y)
                    })
                    .collect()
            })
            .collect();
        Self {
            channels,
            log_freq,
            display,
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
                .y_axis_label(self.display.y_label())
                .legend(Legend::default());

            if self.log_freq {
                let display = &self.display;
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
                        let y_str = display.format_y(point.y);
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
    log_freq: bool,
    display: FreqDisplay,
) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(FreqSignalPlot::new(signal, title, log_freq, display)))),
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
    #[cfg(not(target_os = "macos"))]
    fn test_show_freq_signal_log_db() {
        show_freq_signal(
            "Freq Signal (Log, dB)",
            &make_signal(),
            true,
            FreqDisplay::Magnitude { db: true },
        )
        .unwrap();
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_show_freq_signal_phase() {
        show_freq_signal(
            "Freq Signal (Phase)",
            &make_signal(),
            true,
            FreqDisplay::Phase(PhaseUnit::Radians),
        )
        .unwrap();
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_show_freq_signal_phase_degrees() {
        show_freq_signal(
            "Freq Signal (Phase, Degrees)",
            &make_signal(),
            true,
            FreqDisplay::Phase(PhaseUnit::Degrees),
        )
        .unwrap();
    }
}
