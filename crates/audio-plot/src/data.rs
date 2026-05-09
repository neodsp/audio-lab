use audio_signal::data::real_data::RealData;
use eframe::egui;
use egui_plot::{GridInput, GridMark, Legend, Line, Plot, PlotPoint, PlotPoints};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct RealDataPlotOptions {
    pub log_x: bool,
}

/// Returns log10-spaced grid marks (same tiers as spectrum_grid_spacer in freq.rs).
fn log_grid_spacer(input: GridInput) -> Vec<GridMark> {
    let (lo, hi) = (input.bounds.0, input.bounds.1);
    let mut marks = Vec::new();

    let start_exp = lo.floor() as i32 - 1;
    let end_exp = hi.ceil() as i32 + 1;

    for exp in start_exp..=end_exp {
        let decade = 10f64.powi(exp);
        for mult in 1..10 {
            let val = decade * mult as f64;
            let log_pos = val.log10();
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

pub(crate) struct RealDataPlot {
    channels: Vec<Vec<PlotPoint>>,
    options: RealDataPlotOptions,
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
        options: RealDataPlotOptions,
    ) -> Self {
        let x_data = data.x_data();
        let channels = data
            .channel_iter()
            .map(|ch| {
                x_data
                    .iter()
                    .zip(ch.iter())
                    .filter(|&(&x, _)| !options.log_x || x > 0.0)
                    .map(|(&x, &y)| {
                        let px = if options.log_x { x.log10() } else { x };
                        PlotPoint::new(px, y)
                    })
                    .collect()
            })
            .collect();
        Self {
            channels,
            options,
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
            let mut plot = Plot::new("real_data")
                .x_axis_label(&self.x_label)
                .y_axis_label(&self.y_label)
                .legend(Legend::default());

            if self.options.log_x {
                plot = plot
                    .x_grid_spacer(log_grid_spacer)
                    .x_axis_formatter(|mark, _range| {
                        let v = 10f64.powf(mark.value);
                        if v >= 1000.0 {
                            format!("{:.3}k", v / 1000.0)
                        } else {
                            format!("{v:.4}")
                        }
                    })
                    .label_formatter(|name, point| {
                        let v = 10f64.powf(point.x);
                        let x_str = if v >= 1000.0 {
                            format!("{:.3}k", v / 1000.0)
                        } else {
                            format!("{v:.4}")
                        };
                        if name.is_empty() {
                            format!("{x_str}\n{:.4}", point.y)
                        } else {
                            format!("{name}\n{x_str}\n{:.4}", point.y)
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

pub fn show_real_data(
    title: &str,
    data: &RealData,
    x_label: impl Into<String>,
    y_label: impl Into<String>,
    options: RealDataPlotOptions,
) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| {
            Ok(Box::new(RealDataPlot::new(
                data, title, x_label, y_label, options,
            )))
        }),
    )?)
}

#[cfg(test)]
mod tests {
    use audio_signal::data::real_data::RealData;

    use super::*;

    #[test]
    #[ignore]
    fn test_show_real_data() {
        use audio_signal::ndarray::{Array1, Axis, stack};

        let x = Array1::linspace(0.0, 1.0, 500);
        let data = RealData::new(
            x.clone(),
            stack![Axis(0), x.mapv(|t| t.sin()), x.mapv(|t| t.cos())],
        )
        .unwrap();
        show_real_data(
            "Real Data Test",
            &data,
            "x",
            "y",
            RealDataPlotOptions::default(),
        )
        .unwrap();
    }
}
