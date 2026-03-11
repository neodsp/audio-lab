use audio_signal::signal::freq_signal::FreqSignal;
use audio_signal::signal::time_signal::TimeSignal;
use eframe::egui;
use egui_plot::{GridInput, GridMark, Legend, Line, Plot, PlotPoint, PlotPoints};

/// Returns spectrum-style grid marks in log10 space.
/// Three tiers:
///   - Major (step 1.0):  decade boundaries — 10, 100, 1k, 10k, 100k
///   - Medium (step 0.3): ×2 and ×5 per decade — 20, 50, 200, 500, …
///   - Minor (step 0.1):  remaining integers per decade — 30, 40, 60, 70, …
fn spectrum_grid_spacer(input: GridInput) -> Vec<GridMark> {
    let (lo, hi) = (input.bounds.0, input.bounds.1);
    let mut marks = Vec::new();

    // Iterate over decades that could be visible
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
                1 => 1.0,     // decade boundary: 10, 100, 1k, …
                2 | 5 => 0.3, // ×2 and ×5: 20, 50, 200, 500, …
                _ => 0.1,     // remaining: 30, 40, 60, 70, …
            };
            marks.push(GridMark {
                value: log_pos,
                step_size,
            });
        }
    }
    marks
}

struct TimeSignalPlot {
    channels: Vec<Vec<PlotPoint>>,
}

impl TimeSignalPlot {
    fn new(signal: &TimeSignal) -> Self {
        let time_steps = signal.time_steps();
        let channels = signal
            .channel_iter()
            .map(|ch| {
                time_steps
                    .iter()
                    .zip(ch.iter())
                    .map(|(&t, &y)| PlotPoint::new(t, y))
                    .collect()
            })
            .collect();
        Self { channels }
    }
}

impl eframe::App for TimeSignalPlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("time_signal")
                .x_axis_label("Time (s)")
                .y_axis_label("Amplitude")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    for (i, points) in self.channels.iter().enumerate() {
                        plot_ui.line(Line::new(
                            format!("Channel {i}"),
                            PlotPoints::Owned(points.clone()),
                        ));
                    }
                });
        });
    }
}

const DB_FLOOR: f64 = -120.0;

struct FreqSignalPlot {
    channels: Vec<Vec<PlotPoint>>,
    log_freq: bool,
    db: bool,
}

impl FreqSignalPlot {
    fn new(signal: &FreqSignal, log_freq: bool, db: bool) -> Self {
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
                        let y = if db {
                            (20.0 * c.norm().log10()).max(DB_FLOOR)
                        } else {
                            c.norm()
                        };
                        PlotPoint::new(x, y)
                    })
                    .collect()
            })
            .collect();
        Self {
            channels,
            log_freq,
            db,
        }
    }
}

impl eframe::App for FreqSignalPlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let y_label = if self.db {
                "Magnitude (dB)"
            } else {
                "Magnitude"
            };
            let mut plot = Plot::new("freq_signal")
                .x_axis_label("Frequency (Hz)")
                .y_axis_label(y_label)
                .legend(Legend::default());

            if self.log_freq {
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
                    .label_formatter({
                        let db = self.db;
                        move |name, point| {
                            let hz = 10f64.powf(point.x);
                            let freq_str = if hz >= 1000.0 {
                                format!("{:.1} kHz", hz / 1000.0)
                            } else {
                                format!("{hz:.1} Hz")
                            };
                            let y_str = if db {
                                format!("{:.1} dB", point.y)
                            } else {
                                format!("{:.4}", point.y)
                            };
                            if name.is_empty() {
                                format!("{freq_str}\n{y_str}")
                            } else {
                                format!("{name}\n{freq_str}\n{y_str}")
                            }
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
    }
}

fn native_options_any_thread() -> eframe::NativeOptions {
    eframe::NativeOptions {
        event_loop_builder: Some(Box::new(|builder| {
            #[cfg(target_os = "linux")]
            if std::env::var("WAYLAND_DISPLAY").is_ok() {
                use winit::platform::wayland::EventLoopBuilderExtWayland;
                builder.with_any_thread(true);
            } else {
                use winit::platform::x11::EventLoopBuilderExtX11;
                builder.with_any_thread(true);
            }
            #[cfg(target_os = "windows")]
            {
                use winit::platform::windows::EventLoopBuilderExtWindows;
                builder.with_any_thread(true);
            }
        })),
        ..Default::default()
    }
}

pub fn show_time_signal(title: &str, signal: &TimeSignal) -> eframe::Result {
    eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(TimeSignalPlot::new(signal)))),
    )
}

pub fn show_freq_signal(
    title: &str,
    signal: &FreqSignal,
    log_freq: bool,
    db: bool,
) -> eframe::Result {
    eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(FreqSignalPlot::new(signal, log_freq, db)))),
    )
}

#[cfg(test)]
mod tests {
    use audio_signal::arr2;
    use audio_signal::signal::time_signal::TimeSignal;

    use super::*;

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_show_time_signal() {
        let sample_rate = 44100.0;
        let signal = TimeSignal::new(
            arr2(&[
                std::array::from_fn::<f64, 1000, _>(|i| {
                    (i as f64 / sample_rate * 440.0 * std::f64::consts::TAU).sin()
                }),
                std::array::from_fn::<f64, 1000, _>(|i| {
                    (i as f64 / sample_rate * 880.0 * std::f64::consts::TAU).sin()
                }),
            ]),
            sample_rate,
        )
        .unwrap();
        show_time_signal("Time Signal Test", &signal).unwrap();
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_show_freq_signal_log_db() {
        let sample_rate = 44100.0;
        let signal = TimeSignal::new(
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
        .unwrap();
        show_freq_signal("Freq Signal Test (Log dB)", &signal.into_freq(), true, true).unwrap();
    }
}
