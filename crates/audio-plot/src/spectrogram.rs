use audio_signal::signal::Spectrogram;
use eframe::egui;

use egui_plot::{Plot, PlotImage, PlotPoint};

use crate::native_options_any_thread;
use crate::save::SavePlotState;

// ─── Turbo colormap ───────────────────────────────────────────────────────────

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

// ─── Colorbar ─────────────────────────────────────────────────────────────────

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

    // Gradient: 128 horizontal strips, top = db_max (t=1), bottom = db_min (t=0).
    let n = 128usize;
    for i in 0..n {
        let t = 1.0 - (i as f32 + 0.5) / n as f32;
        let y_top = bar_rect.min.y + i as f32 * bar_height / n as f32;
        let y_bot = bar_rect.min.y + (i + 1) as f32 * bar_height / n as f32;
        let strip = egui::Rect::from_min_max(
            egui::pos2(bar_rect.min.x, y_top),
            egui::pos2(bar_rect.max.x, y_bot),
        );
        painter.rect_filled(strip, 0.0, apply_colormap(t, TURBO_PALETTE));
    }

    // Border.
    let stroke = egui::Stroke::new(1.0, ui.visuals().text_color());
    painter.rect_stroke(bar_rect, 0.0, stroke, egui::StrokeKind::Outside);

    // Five tick marks with dB labels.
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

// ─── App ─────────────────────────────────────────────────────────────────────

pub(crate) struct SpectrogramPlot {
    /// Flat dB values per channel in row-major order:
    /// `values[freq_bin * num_frames + frame]`.
    /// Row 0 (freq_bin = 0) sits at the bottom of the plot = lowest frequency.
    values: Vec<Vec<f64>>,
    images: Vec<egui::ColorImage>,
    textures: Vec<Option<egui::TextureHandle>>,
    num_frames: usize,
    num_freq_bins: usize,
    db_floor: f64,
    db_peak: f64,
    /// Lower-left corner of the image in plot (time, freq) coordinates.
    pos: PlotPoint,
    /// Width × height of one tile in plot coordinates (seconds × Hz).
    tile_size: (f32, f32),
    /// Center of the full image in plot coordinates.
    image_center: PlotPoint,
    /// Width × height of the full image in plot coordinates (seconds × Hz).
    image_size: egui::Vec2,
    num_channels: usize,
    current_channel: usize,
    save: SavePlotState,
}

impl SpectrogramPlot {
    pub(crate) fn new(spec: &Spectrogram, title: &str, db_floor: f64) -> Self {
        let num_channels = spec.num_channels();
        let num_frames = spec.num_frames();
        let num_freq_bins = spec.num_freq_bins();
        let frame_times = spec.frame_times();
        let freq_bins = spec.freq_bins();

        let mag_db = spec.amplitude_spectrum_db(db_floor);
        let db_peak = mag_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Flat value arrays: row = freq_bin (0 = lowest), col = frame.
        let values: Vec<Vec<f64>> = (0..num_channels)
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
        let db_span = (db_peak - db_floor).max(f64::EPSILON);

        let images = values
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
            .collect();

        // Tile dimensions in plot-coordinate units.
        let dt = if num_frames > 1 {
            (frame_times[num_frames - 1] - frame_times[0]) / (num_frames - 1) as f64
        } else {
            spec.hop_size() as f64 / spec.sample_rate()
        };
        let df = if num_freq_bins > 1 {
            (freq_bins[num_freq_bins - 1] - freq_bins[0]) / (num_freq_bins - 1) as f64
        } else {
            spec.sample_rate() / spec.window_size() as f64
        };

        // Lower-left corner = half a tile before the first frame/bin centre.
        let pos = PlotPoint {
            x: frame_times.first().copied().unwrap_or(0.0) - dt / 2.0,
            y: freq_bins.first().copied().unwrap_or(0.0) - df / 2.0,
        };
        let image_size = egui::vec2(
            (num_frames as f64 * dt) as f32,
            (num_freq_bins as f64 * df) as f32,
        );
        let image_center = PlotPoint::new(
            pos.x + f64::from(image_size.x) / 2.0,
            pos.y + f64::from(image_size.y) / 2.0,
        );

        Self {
            values,
            images,
            textures: vec![None; num_channels],
            num_frames,
            num_freq_bins,
            db_floor,
            db_peak,
            pos,
            tile_size: (dt as f32, df as f32),
            image_center,
            image_size,
            num_channels,
            current_channel: 0,
            save: SavePlotState::new(title),
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
                        ui.selectable_value(&mut self.current_channel, ch, format!("Channel {ch}"));
                    }
                });
            }
        });

        // Colorbar on the right; must be declared before CentralPanel.
        egui::SidePanel::right("colorbar")
            .exact_width(80.0)
            .resizable(false)
            .show(ctx, |ui| {
                draw_colorbar(ui, self.db_floor, self.db_peak);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let texture = self.textures[self.current_channel].get_or_insert_with(|| {
                ctx.load_texture(
                    format!("spectrogram-channel-{}", self.current_channel),
                    self.images[self.current_channel].clone(),
                    egui::TextureOptions::NEAREST,
                )
            });
            let image = PlotImage::new(
                format!("Channel {}", self.current_channel),
                texture.id(),
                self.image_center,
                self.image_size,
            );

            let mut hovered_coord = None;
            let inner = Plot::new("spectrogram")
                .x_axis_label("Time (s)")
                .y_axis_label("Frequency (Hz)")
                .y_axis_formatter(|mark, _range| {
                    let hz = mark.value;
                    if hz >= 1000.0 {
                        format!("{:.0} kHz", hz / 1000.0)
                    } else {
                        format!("{hz:.0} Hz")
                    }
                })
                .set_margin_fraction(egui::vec2(0.0, 0.0))
                .show(ui, |plot_ui| {
                    if plot_ui.response().hovered() {
                        hovered_coord = plot_ui.pointer_coordinate();
                    }
                    plot_ui.image(image);
                });

            if let Some(coord) = hovered_coord {
                let freq_hz = coord.y;

                // Map plot coordinates back to the nearest tile to read its dB value.
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

                let freq_str = if freq_hz >= 1000.0 {
                    format!("{:.2} kHz", freq_hz / 1000.0)
                } else {
                    format!("{:.0} Hz", freq_hz)
                };
                let text = match db_value {
                    Some(db) => format!("t = {:.3} s\nf = {}\n{:.1} dB", coord.x, freq_str, db),
                    None => format!("t = {:.3} s\nf = {}", coord.x, freq_str),
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

// ─── Public API ───────────────────────────────────────────────────────────────

/// Show a spectrogram as a color image using the Turbo colormap.
///
/// Values are displayed in dB re. calibrated one-sided amplitude spectrum
/// (20 × log₁₀(amplitude), floored at `db_floor`).
/// `db_floor` also sets the cold end of the color scale (e.g. `-80.0`);
/// the hot end is the peak value across all channels.
///
/// The plot supports zoom (scroll wheel) and pan (drag).
pub fn show_spectrogram(
    title: &str,
    spectrogram: &Spectrogram,
    db_floor: f64,
) -> Result<(), crate::Error> {
    Ok(eframe::run_native(
        title,
        native_options_any_thread(),
        Box::new(|_cc| Ok(Box::new(SpectrogramPlot::new(spectrogram, title, db_floor)))),
    )?)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

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
    #[ignore]
    fn test_show_spectrogram() {
        show_spectrogram("Spectrogram Test", &make_spectrogram(), -80.0).unwrap();
    }
}
