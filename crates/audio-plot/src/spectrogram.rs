use audio_signal::{ndarray, signal::Spectrogram};
use eframe::egui;
use egui_plot::{GridInput, GridMark, Plot, PlotImage, PlotPoint};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

const TURBO_PALETTE: &[egui::Color32] = &[
    egui::Color32::from_rgb(48, 18, 59),
    egui::Color32::from_rgb(72, 42, 121),
    egui::Color32::from_rgb(55, 87, 176),
    egui::Color32::from_rgb(35, 135, 207),
    egui::Color32::from_rgb(30, 178, 185),
    egui::Color32::from_rgb(45, 209, 145),
    egui::Color32::from_rgb(88, 223, 98),
    egui::Color32::from_rgb(164, 227, 39),
    egui::Color32::from_rgb(228, 207, 14),
    egui::Color32::from_rgb(250, 170, 10),
    egui::Color32::from_rgb(246, 117, 6),
    egui::Color32::from_rgb(220, 62, 2),
    egui::Color32::from_rgb(122, 4, 2),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpectrogramScale {
    Linear,
    Log,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrogramPlotOptions {
    pub log_freq: bool,
    pub db_floor: f64,
}

impl Default for SpectrogramPlotOptions {
    fn default() -> Self {
        Self {
            log_freq: false,
            db_floor: -80.0,
        }
    }
}

fn apply_colormap(t: f32, palette: &[egui::Color32]) -> egui::Color32 {
    if palette.len() < 2 {
        return palette.first().copied().unwrap_or(egui::Color32::BLACK);
    }
    let v = t.clamp(0.0, 1.0);
    let idx = v * (palette.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(palette.len() - 1);
    let t = idx - lo as f32;
    let c0 = palette[lo];
    let c1 = palette[hi];
    egui::Color32::from_rgb(
        (c0.r() as f32 * (1.0 - t) + c1.r() as f32 * t).round() as u8,
        (c0.g() as f32 * (1.0 - t) + c1.g() as f32 * t).round() as u8,
        (c0.b() as f32 * (1.0 - t) + c1.b() as f32 * t).round() as u8,
    )
}

fn draw_colorbar(ui: &mut egui::Ui, db_min: f64, db_max: f64) {
    const BAR_WIDTH: f32 = 16.0;
    const TICK_LEN: f32 = 5.0;
    const LABEL_GAP: f32 = 4.0;
    const MARGIN_Y: f32 = 10.0;

    let available = ui.available_size();
    let (response, painter) = ui.allocate_painter(available, egui::Sense::hover());

    let bar_rect = egui::Rect::from_min_max(
        egui::pos2(response.rect.min.x, response.rect.min.y + MARGIN_Y),
        egui::pos2(
            response.rect.min.x + BAR_WIDTH,
            response.rect.max.y - MARGIN_Y,
        ),
    );
    let bar_height = bar_rect.height();

    for i in 0..128usize {
        let t = 1.0 - (i as f32 + 0.5) / 128.0;
        let y_top = bar_rect.min.y + i as f32 * bar_height / 128.0;
        let y_bot = bar_rect.min.y + (i + 1) as f32 * bar_height / 128.0;
        let strip = egui::Rect::from_min_max(
            egui::pos2(bar_rect.min.x, y_top),
            egui::pos2(bar_rect.max.x, y_bot),
        );
        painter.rect_filled(strip, 0.0, apply_colormap(t, TURBO_PALETTE));
    }

    let stroke = egui::Stroke::new(1.0, ui.visuals().text_color());
    painter.rect_stroke(bar_rect, 0.0, stroke, egui::StrokeKind::Outside);

    let font_id = egui::TextStyle::Small.resolve(ui.style());
    let text_color = ui.visuals().text_color();
    for tick in 0..=4 {
        let t = tick as f32 / 4.0;
        let db = db_min + t as f64 * (db_max - db_min);
        let y = bar_rect.min.y + (1.0 - t) * bar_height;
        painter.line_segment(
            [
                egui::pos2(bar_rect.max.x, y),
                egui::pos2(bar_rect.max.x + TICK_LEN, y),
            ],
            egui::Stroke::new(1.0, text_color),
        );
        painter.text(
            egui::pos2(bar_rect.max.x + TICK_LEN + LABEL_GAP, y),
            egui::Align2::LEFT_CENTER,
            format!("{db:.0} dB"),
            font_id.clone(),
            text_color,
        );
    }
}

fn format_hz(freq_hz: f64) -> String {
    if freq_hz >= 1000.0 {
        format!("{:.2} kHz", freq_hz / 1000.0)
    } else {
        format!("{freq_hz:.0} Hz")
    }
}

fn format_hz_tick(freq_hz: f64) -> String {
    if freq_hz >= 1000.0 {
        format!("{:.0} kHz", freq_hz / 1000.0)
    } else {
        format!("{freq_hz:.0} Hz")
    }
}

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

fn positive_freq_bins(spec: &Spectrogram) -> Vec<f64> {
    spec.freq_bins()
        .iter()
        .copied()
        .filter(|&freq| freq > 0.0)
        .collect()
}

fn make_log_freq_grid(freq_bins: &[f64]) -> Vec<f64> {
    if freq_bins.is_empty() {
        return Vec::new();
    }
    if freq_bins.len() == 1 {
        return vec![freq_bins[0]];
    }

    let start = freq_bins[0].log10();
    let end = freq_bins[freq_bins.len() - 1].log10();
    (0..freq_bins.len())
        .map(|idx| {
            let t = idx as f64 / (freq_bins.len() - 1) as f64;
            10.0_f64.powf(start + t * (end - start))
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InterpolationStep {
    lower: usize,
    upper: usize,
    weight: f64,
}

fn make_interpolation_plan(source_freqs: &[f64], target_freqs: &[f64]) -> Vec<InterpolationStep> {
    target_freqs
        .iter()
        .map(|&target_freq| {
            match source_freqs.binary_search_by(|probe| probe.total_cmp(&target_freq)) {
                Ok(index) => InterpolationStep {
                    lower: index,
                    upper: index,
                    weight: 0.0,
                },
                Err(0) => InterpolationStep {
                    lower: 0,
                    upper: 0,
                    weight: 0.0,
                },
                Err(index) if index >= source_freqs.len() => {
                    let last = source_freqs.len() - 1;
                    InterpolationStep {
                        lower: last,
                        upper: last,
                        weight: 0.0,
                    }
                }
                Err(upper) => {
                    let lower = upper - 1;
                    let f0 = source_freqs[lower];
                    let f1 = source_freqs[upper];
                    let weight = if f1 > f0 {
                        (target_freq - f0) / (f1 - f0)
                    } else {
                        0.0
                    };
                    InterpolationStep {
                        lower,
                        upper,
                        weight,
                    }
                }
            }
        })
        .collect()
}

#[cfg(test)]
fn interpolate_row(source_freqs: &[f64], source_values: &[f64], target_freq: f64) -> f64 {
    let step = make_interpolation_plan(source_freqs, &[target_freq])[0];
    let v0 = source_values[step.lower];
    let v1 = source_values[step.upper];
    v0 + step.weight * (v1 - v0)
}

fn make_linear_values(
    spec: &Spectrogram,
    mag_db: &ndarray::Array3<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let num_channels = spec.num_channels();
    let num_frames = spec.num_frames();
    let num_freq_bins = spec.num_freq_bins();

    let values = (0..num_channels)
        .map(|ch| {
            let mut v = Vec::with_capacity(num_frames * num_freq_bins);
            for freq_bin in 0..num_freq_bins {
                for frame in 0..num_frames {
                    v.push(mag_db[[ch, frame, freq_bin]]);
                }
            }
            v
        })
        .collect();

    (values, spec.freq_bins().to_vec())
}

fn make_log_values(spec: &Spectrogram, mag_db: &ndarray::Array3<f64>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let source_freqs = positive_freq_bins(spec);
    if source_freqs.len() < 2 {
        return make_linear_values(spec, mag_db);
    }

    let target_freqs = make_log_freq_grid(&source_freqs);
    let interpolation_plan = make_interpolation_plan(&source_freqs, &target_freqs);
    let num_channels = spec.num_channels();
    let num_frames = spec.num_frames();

    let values = (0..num_channels)
        .map(|ch| {
            let mut v = Vec::with_capacity(num_frames * target_freqs.len());
            for step in &interpolation_plan {
                for frame in 0..num_frames {
                    let v0 = mag_db[[ch, frame, step.lower + 1]];
                    let v1 = mag_db[[ch, frame, step.upper + 1]];
                    v.push(v0 + step.weight * (v1 - v0));
                }
            }
            v
        })
        .collect();

    (values, target_freqs)
}

fn values_to_images(
    values: &[Vec<f64>],
    num_frames: usize,
    num_freq_bins: usize,
    db_floor: f64,
    db_peak: f64,
) -> Vec<egui::ColorImage> {
    let db_span = (db_peak - db_floor).max(f64::EPSILON);
    values
        .iter()
        .map(|channel_values| {
            let mut pixels = Vec::with_capacity(num_frames * num_freq_bins);
            for freq_bin in (0..num_freq_bins).rev() {
                let row_start = freq_bin * num_frames;
                for frame in 0..num_frames {
                    let db = channel_values[row_start + frame];
                    let t = ((db - db_floor) / db_span).clamp(0.0, 1.0) as f32;
                    pixels.push(apply_colormap(t, TURBO_PALETTE));
                }
            }
            egui::ColorImage::new([num_frames, num_freq_bins], pixels)
        })
        .collect()
}

pub(crate) struct SpectrogramPlot {
    values: Vec<Vec<f64>>,
    images: Vec<egui::ColorImage>,
    textures: Vec<Option<egui::TextureHandle>>,
    num_frames: usize,
    num_freq_bins: usize,
    db_floor: f64,
    db_peak: f64,
    pos: PlotPoint,
    tile_size: (f32, f32),
    image_center: PlotPoint,
    image_size: egui::Vec2,
    num_channels: usize,
    current_channel: usize,
    channel_labels: Vec<String>,
    save: SavePlotState,
    scale: SpectrogramScale,
}

impl SpectrogramPlot {
    pub(crate) fn new(spec: &Spectrogram, title: &str, options: SpectrogramPlotOptions) -> Self {
        let scale = if options.log_freq {
            SpectrogramScale::Log
        } else {
            SpectrogramScale::Linear
        };
        let db_floor = options.db_floor;
        let num_channels = spec.num_channels();
        let num_frames = spec.num_frames();
        let frame_times = spec.frame_times();
        let mag_db = spec.amplitude_spectrum_db(db_floor);
        let db_peak = mag_db.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let (values, plotted_freqs_hz) = match scale {
            SpectrogramScale::Linear => make_linear_values(spec, &mag_db),
            SpectrogramScale::Log => make_log_values(spec, &mag_db),
        };
        let num_freq_bins = plotted_freqs_hz.len();
        let images = values_to_images(&values, num_frames, num_freq_bins, db_floor, db_peak);

        let dt = if num_frames > 1 {
            (frame_times[num_frames - 1] - frame_times[0]) / (num_frames - 1) as f64
        } else {
            spec.hop_size() as f64 / spec.sample_rate()
        };

        let plotted_y: Vec<f64> = match scale {
            SpectrogramScale::Linear => plotted_freqs_hz.clone(),
            SpectrogramScale::Log => plotted_freqs_hz.iter().map(|freq| freq.log10()).collect(),
        };
        let dy = if num_freq_bins > 1 {
            (plotted_y[num_freq_bins - 1] - plotted_y[0]) / (num_freq_bins - 1) as f64
        } else if scale == SpectrogramScale::Log {
            0.1
        } else {
            spec.sample_rate() / spec.window_size() as f64
        };

        let pos = PlotPoint {
            x: frame_times.first().copied().unwrap_or(0.0) - dt / 2.0,
            y: plotted_y.first().copied().unwrap_or(0.0) - dy / 2.0,
        };
        let image_size = egui::vec2(
            (num_frames as f64 * dt) as f32,
            (num_freq_bins as f64 * dy) as f32,
        );
        let image_center = PlotPoint::new(
            pos.x + f64::from(image_size.x) / 2.0,
            pos.y + f64::from(image_size.y) / 2.0,
        );

        let channel_labels: Vec<String> = (0..num_channels)
            .map(|ch| match spec.channel_label(ch) {
                Some(label) => crate::legend::clip_label(label),
                None => format!("Channel {ch}"),
            })
            .collect();

        Self {
            values,
            images,
            textures: vec![None; num_channels],
            num_frames,
            num_freq_bins,
            db_floor,
            db_peak,
            pos,
            tile_size: (dt as f32, dy as f32),
            image_center,
            image_size,
            num_channels,
            current_channel: 0,
            channel_labels,
            save: SavePlotState::new(title),
            scale,
        }
    }

    fn freq_from_plot_y(&self, y: f64) -> f64 {
        match self.scale {
            SpectrogramScale::Linear => y,
            SpectrogramScale::Log => 10.0_f64.powf(y),
        }
    }

    fn axis_label(&self) -> &'static str {
        match self.scale {
            SpectrogramScale::Linear => "Frequency (Hz)",
            SpectrogramScale::Log => "Frequency (Hz, log)",
        }
    }
}

impl eframe::App for SpectrogramPlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            self.save.show_panel(ui);
            if self.num_channels > 1 {
                ui.horizontal(|ui| {
                    for ch in 0..self.num_channels {
                        ui.selectable_value(&mut self.current_channel, ch, &self.channel_labels[ch]);
                    }
                });
            }
        });

        egui::SidePanel::right("colorbar")
            .exact_width(80.0)
            .resizable(false)
            .show(ctx, |ui| {
                draw_colorbar(ui, self.db_floor, self.db_peak);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let texture = self.textures[self.current_channel].get_or_insert_with(|| {
                ctx.load_texture(
                    format!(
                        "spectrogram-{:?}-channel-{}",
                        self.scale, self.current_channel
                    ),
                    self.images[self.current_channel].clone(),
                    egui::TextureOptions::NEAREST,
                )
            });
            let image = PlotImage::new(
                self.channel_labels[self.current_channel].clone(),
                texture.id(),
                self.image_center,
                self.image_size,
            );

            let scale = self.scale;
            let mut plot = Plot::new("spectrogram")
                .x_axis_label("Time (s)")
                .y_axis_label(self.axis_label())
                .y_axis_formatter(move |mark, _range| {
                    let freq_hz = match scale {
                        SpectrogramScale::Linear => mark.value,
                        SpectrogramScale::Log => 10.0_f64.powf(mark.value),
                    };
                    format_hz_tick(freq_hz)
                })
                .set_margin_fraction(egui::vec2(0.0, 0.0));
            if self.scale == SpectrogramScale::Log {
                plot = plot.y_grid_spacer(spectrum_grid_spacer);
            }

            let mut hovered_coord = None;
            let inner = plot.show(ui, |plot_ui| {
                if plot_ui.response().hovered() {
                    hovered_coord = plot_ui.pointer_coordinate();
                }
                plot_ui.image(image);
            });

            if let Some(coord) = hovered_coord {
                let freq_hz = self.freq_from_plot_y(coord.y);
                let frame = ((coord.x - self.pos.x) / self.tile_size.0 as f64).floor() as isize;
                let freq_bin = ((coord.y - self.pos.y) / self.tile_size.1 as f64).floor() as isize;
                let db_value = if frame >= 0 && (frame as usize) < self.num_frames && freq_bin >= 0
                {
                    if (freq_bin as usize) < self.num_freq_bins {
                        let idx = freq_bin as usize * self.num_frames + frame as usize;
                        Some(self.values[self.current_channel][idx])
                    } else {
                        None
                    }
                } else {
                    None
                };

                let text = match db_value {
                    Some(db) => format!(
                        "t = {:.3} s\nf = {}\n{:.1} dB",
                        coord.x,
                        format_hz(freq_hz),
                        db
                    ),
                    None => format!("t = {:.3} s\nf = {}", coord.x, format_hz(freq_hz)),
                };
                let mut tooltip = egui::Tooltip::always_open(
                    ui.ctx().clone(),
                    inner.response.layer_id,
                    inner.response.id.with("hover_tooltip"),
                    egui::PopupAnchor::Pointer,
                );
                tooltip.popup = tooltip.popup.gap(12.0).width(f32::INFINITY);
                tooltip.show(|ui: &mut egui::Ui| {
                    ui.add(egui::Label::new(text).wrap_mode(egui::TextWrapMode::Extend));
                });
            }
        });

        self.save.handle_screenshot(ctx);
    }
}

pub fn show_spectrogram(
    title: &str,
    spectrogram: &Spectrogram,
    options: SpectrogramPlotOptions,
) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(SpectrogramPlot::new(spectrogram, title, options)))),
    )?)
}

#[cfg(test)]
mod tests {
    use audio_signal::{
        ndarray::{Array1, Array3},
        signal::{Spectrogram, SpectrogramNormalization},
    };
    use num::complex::Complex64;

    use super::*;

    fn make_spectrogram() -> Spectrogram {
        let num_channels = 2;
        let num_frames = 50;
        let num_freq_bins = 17;
        let sample_rate = 16_000.0;
        let data =
            Array3::from_shape_fn((num_channels, num_frames, num_freq_bins), |(ch, f, b)| {
                Complex64::new((ch as f64 + 1.0) * (f + b) as f64, 0.0)
            });
        let frame_times = Array1::linspace(0.0, 1.5, num_frames);
        let freq_bins = Array1::linspace(0.0, 8000.0, num_freq_bins);
        Spectrogram::new(
            data,
            frame_times,
            freq_bins,
            sample_rate,
            256,
            128,
            SpectrogramNormalization::new(1.0, 256.0),
        )
    }

    #[test]
    fn log_grid_uses_positive_frequencies() {
        let freqs = vec![62.5, 125.0, 250.0, 500.0, 1000.0];
        let grid = make_log_freq_grid(&freqs);

        assert_eq!(grid.len(), freqs.len());
        assert!((grid[0] - 62.5).abs() < 1e-12);
        assert!((grid[grid.len() - 1] - 1000.0).abs() < 1e-12);
        assert!(grid.windows(2).all(|window| window[1] > window[0]));
    }

    #[test]
    fn interpolation_hits_exact_bin_values() {
        let freqs = [100.0, 200.0, 400.0];
        let values = [-60.0, -20.0, -10.0];

        assert!((interpolate_row(&freqs, &values, 200.0) + 20.0).abs() < 1e-12);
        assert!((interpolate_row(&freqs, &values, 300.0) + 15.0).abs() < 1e-12);
    }

    #[test]
    fn interpolation_plan_reuses_bin_mapping() {
        let freqs = [100.0, 200.0, 400.0];
        let plan = make_interpolation_plan(&freqs, &[100.0, 300.0, 500.0]);

        assert_eq!(
            plan,
            vec![
                InterpolationStep {
                    lower: 0,
                    upper: 0,
                    weight: 0.0
                },
                InterpolationStep {
                    lower: 1,
                    upper: 2,
                    weight: 0.5
                },
                InterpolationStep {
                    lower: 2,
                    upper: 2,
                    weight: 0.0
                }
            ]
        );
    }

    #[test]
    #[ignore]
    fn test_show_spectrogram() {
        show_spectrogram(
            "Spectrogram Test",
            &make_spectrogram(),
            SpectrogramPlotOptions::default(),
        )
        .unwrap();
    }
}
