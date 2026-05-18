use audio_signal::{math::gain_to_db, signal::time_signal::TimeSignal};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

const DB_FLOOR: f64 = -120.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeValue {
    Amplitude,
    AmplitudeDb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimePlotOptions {
    pub value: TimeValue,
}

impl Default for TimePlotOptions {
    fn default() -> Self {
        Self {
            value: TimeValue::Amplitude,
        }
    }
}

impl TimeValue {
    fn value(&self, sample: f64) -> f64 {
        match self {
            Self::Amplitude => sample,
            Self::AmplitudeDb => gain_to_db(sample.abs()).max(DB_FLOOR),
        }
    }

    fn y_label(&self) -> &'static str {
        match self {
            Self::Amplitude => "Amplitude",
            Self::AmplitudeDb => "Amplitude (dB)",
        }
    }
}

pub(crate) struct TimeSignalPlot {
    series: Vec<(String, Vec<PlotPoint>)>,
    options: TimePlotOptions,
    save: SavePlotState,
}

impl TimeSignalPlot {
    pub(crate) fn new(signals: &[&TimeSignal], title: &str, options: TimePlotOptions) -> Self {
        let mut series = Vec::new();
        for (signal_index, signal) in signals.iter().enumerate() {
            let time_steps = signal.time_steps();
            let num_channels = signal.num_channels();
            for (channel_index, channel) in signal.channel_iter().enumerate() {
                let label = crate::legend::build_channel_label(
                    signal.channel_label(channel_index),
                    signal.comment(),
                    signal_index,
                    channel_index,
                    num_channels,
                );
                let points = time_steps
                    .iter()
                    .zip(channel.iter())
                    .map(|(&t, &y)| PlotPoint::new(t, options.value.value(y)))
                    .collect();
                series.push((label, points));
            }
        }
        Self {
            series,
            options,
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
            Plot::new("time_signal")
                .x_axis_label("Time (s)")
                .y_axis_label(self.options.value.y_label())
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    for (label, points) in &self.series {
                        plot_ui.line(Line::new(label.clone(), PlotPoints::Owned(points.clone())));
                    }
                });
        });
        self.save.handle_screenshot(ctx);
    }
}

pub fn show_time_with_options(
    title: &str,
    signals: &[&TimeSignal],
    options: TimePlotOptions,
) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(TimeSignalPlot::new(signals, title, options)))),
    )?)
}

pub fn show_time(title: &str, signals: &[&TimeSignal]) -> Result<(), crate::Error> {
    show_time_with_options(title, signals, Default::default())
}

#[cfg(test)]
mod tests {
    use audio_signal::signal::time_signal::TimeSignal;

    use super::*;

    #[test]
    #[ignore]
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
        show_time("Time Signal Test", &[&signal]).unwrap();
    }

    #[test]
    #[ignore]
    fn test_show_time_signal_db() {
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
        show_time_with_options(
            "Time Signal Test (dB)",
            &[&signal],
            TimePlotOptions {
                value: TimeValue::AmplitudeDb,
            },
        )
        .unwrap();
    }
}
