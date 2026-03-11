use audio_signal::data::real_data::RealData;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

pub(crate) struct RealDataPlot {
    channels: Vec<Vec<PlotPoint>>,
    x_label: String,
    y_label: String,
    save: SavePlotState,
}

impl RealDataPlot {
    pub(crate) fn new(
        data: &RealData,
        title: &str,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        let x_data = data.x_data();
        let channels = data
            .channel_iter()
            .map(|ch| {
                x_data
                    .iter()
                    .zip(ch.iter())
                    .map(|(&x, &y)| PlotPoint::new(x, y))
                    .collect()
            })
            .collect();
        Self {
            channels,
            x_label: x_label.into(),
            y_label: y_label.into(),
            save: SavePlotState::new(title),
        }
    }
}

impl eframe::App for RealDataPlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            self.save.show_panel(ui);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("real_data")
                .x_axis_label(&self.x_label)
                .y_axis_label(&self.y_label)
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
        self.save.handle_screenshot(ctx);
    }
}

pub fn show_real_data(
    title: &str,
    data: &RealData,
    x_label: impl Into<String>,
    y_label: impl Into<String>,
) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(RealDataPlot::new(data, title, x_label, y_label)))),
    )?)
}

#[cfg(test)]
mod tests {
    use audio_signal::data::real_data::RealData;

    use super::*;

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_show_real_data() {
        use audio_signal::ndarray::{Array1, Axis, stack};

        let x = Array1::linspace(0.0, 1.0, 500);
        let data = RealData::new(
            x.clone(),
            stack![Axis(0), x.mapv(|t| t.sin()), x.mapv(|t| t.cos())],
        )
        .unwrap();
        show_real_data("Real Data Test", &data, "x", "y").unwrap();
    }
}
