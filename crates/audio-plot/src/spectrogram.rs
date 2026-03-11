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

fn apply_colormap(value: f32, palette: &[egui::Color32]) -> egui::Color32 {
    if palette.len() < 2 {
        return palette.first().copied().unwrap_or(egui::Color32::BLACK);
    }
    let v = value.clamp(0.0, 1.0);
    let idx = v * (palette.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(palette.len() - 1);
    let t = idx - lo as f32;
    let c0 = palette[lo];
    let c1 = palette[hi];
    egui::Color32::from_rgb(
        lerp_u8(c0.r(), c1.r(), t),
        lerp_u8(c0.g(), c1.g(), t),
        lerp_u8(c0.b(), c1.b(), t),
    )
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t).round() as u8
}

// ─── App ─────────────────────────────────────────────────────────────────────

pub(crate) struct SpectrogramPlot {
    /// Pre-computed color images indexed by channel.
    images: Vec<egui::ColorImage>,
    /// Lazily loaded textures, one per channel.
    textures: Vec<Option<egui::TextureHandle>>,
    frame_times: Vec<f64>,
    freq_bins: Vec<f64>,
    num_channels: usize,
    current_channel: usize,
    save: SavePlotState,
}

impl SpectrogramPlot {
    pub(crate) fn new(spec: &Spectrogram, title: &str, db_floor: f64) -> Self {
        let num_channels = spec.num_channels();
        let num_frames = spec.num_frames();
        let num_freq_bins = spec.num_freq_bins();

        let mag_db = spec.magnitude_db(db_floor);

        // Normalize [db_floor, db_peak] → [0, 1] across all channels.
        let db_peak = mag_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let db_range = (db_peak - db_floor).max(f64::EPSILON);

        // Build one ColorImage per channel.
        // Image width = num_frames, height = num_freq_bins.
        // Row 0 = highest frequency so that image-top aligns with plot-top
        // (egui_plot has y increasing upward; PlotImage maps image row 0 to the
        // top of the plot rect, i.e. the highest y = highest frequency).
        let images = (0..num_channels)
            .map(|ch| {
                let mut pixels = Vec::with_capacity(num_frames * num_freq_bins);
                // Row 0 = highest frequency (image-top = plot-top = high freq).
                for freq_bin in (0..num_freq_bins).rev() {
                    for frame in 0..num_frames {
                        let db = mag_db[[ch, frame, freq_bin]];
                        let norm = ((db - db_floor) / db_range).clamp(0.0, 1.0) as f32;
                        pixels.push(apply_colormap(norm, TURBO_PALETTE));
                    }
                }
                egui::ColorImage::new([num_frames.max(1), num_freq_bins.max(1)], pixels)
            })
            .collect();

        Self {
            images,
            textures: vec![None; num_channels.max(1)],
            frame_times: spec.frame_times().to_vec(),
            freq_bins: spec.freq_bins().to_vec(),
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

        egui::CentralPanel::default().show(ctx, |ui| {
            // Lazily upload the texture for the selected channel.
            if self.textures[self.current_channel].is_none() {
                let handle = ctx.load_texture(
                    format!("spectrogram_ch{}", self.current_channel),
                    self.images[self.current_channel].clone(),
                    egui::TextureOptions::LINEAR,
                );
                self.textures[self.current_channel] = Some(handle);
            }

            let Some(texture) = &self.textures[self.current_channel] else {
                return;
            };

            let (Some(&t0), Some(&t1)) = (self.frame_times.first(), self.frame_times.last()) else {
                return;
            };
            let (Some(&f0), Some(&f1)) = (self.freq_bins.first(), self.freq_bins.last()) else {
                return;
            };

            let dt = (t1 - t0).max(f64::EPSILON);
            let df = (f1 - f0).max(f64::EPSILON);

            let center = PlotPoint::new((t0 + t1) / 2.0, (f0 + f1) / 2.0);
            let size = egui::vec2(dt as f32, df as f32);
            let image = PlotImage::new("spectrogram", texture.id(), center, size);

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
                .label_formatter(|_name, point| {
                    let freq_str = if point.y >= 1000.0 {
                        format!("{:.2} kHz", point.y / 1000.0)
                    } else {
                        format!("{:.1} Hz", point.y)
                    };
                    format!("{:.3} s\n{freq_str}", point.x)
                })
                .set_margin_fraction(egui::vec2(0.0, 0.0))
                .show(ui, |plot_ui| {
                    plot_ui.image(image);
                });

            self.save.set_rect(inner.response.rect);
        });

        self.save.handle_screenshot(ctx);
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Show a spectrogram as a color heatmap using the Turbo colormap.
///
/// `db_floor` sets the lower dB bound for the color scale (e.g. `-80.0`).
/// The upper bound is the peak magnitude across all channels.
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
        signal::Spectrogram,
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
        Spectrogram::new(data, frame_times, freq_bins, sample_rate, 256, 128)
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_show_spectrogram() {
        show_spectrogram("Spectrogram Test", &make_spectrogram(), -80.0).unwrap();
    }
}
