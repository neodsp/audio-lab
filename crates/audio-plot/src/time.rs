use audio_signal::signal::time_signal::TimeSignal;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

pub(crate) struct TimeSignalPlot {
    channels: Vec<Vec<PlotPoint>>,
    save: SavePlotState,
}

impl TimeSignalPlot {
    pub(crate) fn new(signal: &TimeSignal, title: &str) -> Self {
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
        Self {
            channels,
            save: SavePlotState::new(title),
        }
    }
}

impl eframe::App for TimeSignalPlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            self.save.show_panel(ui);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            let inner = Plot::new("time_signal")
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
            self.save.set_rect(inner.response.rect);
        });
        self.save.handle_screenshot(ctx);
    }
}

pub fn show_time_signal(title: &str, signal: &TimeSignal) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(TimeSignalPlot::new(signal, title)))),
    )?)
}

#[cfg(test)]
mod tests {
    use audio_signal::signal::time_signal::TimeSignal;

    use super::*;

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_show_time_signal() {
        use audio_signal::ndarray::arr2;

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
}
