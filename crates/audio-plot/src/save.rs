use eframe::egui;

pub(crate) struct SavePlotState {
    plot_rect: Option<egui::Rect>,
    title: String,
    default_filename: String,
}

impl SavePlotState {
    pub(crate) fn new(title: impl Into<String>) -> Self {
        let title = title.into();
        let default_filename = format!("{title}.png");
        Self {
            plot_rect: None,
            title,
            default_filename,
        }
    }

    pub(crate) fn set_rect(&mut self, rect: egui::Rect) {
        self.plot_rect = Some(rect);
    }

    pub(crate) fn show_panel(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Button on the left
            if ui.button("Save Plot").clicked() {
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            }

            // Paint the heading centered over the full panel width
            let center_x = ui.clip_rect().center().x;
            let center_y = ui.max_rect().center().y;
            ui.painter().text(
                egui::pos2(center_x, center_y),
                egui::Align2::CENTER_CENTER,
                &self.title,
                egui::TextStyle::Heading.resolve(ui.style()),
                ui.visuals().text_color(),
            );

            // Consume remaining space so the layout doesn't shrink
            ui.allocate_space(ui.available_size());
        });
    }

    pub(crate) fn handle_screenshot(&self, ctx: &egui::Context) {
        let screenshot = ctx.input(|i| {
            for event in &i.raw.events {
                if let egui::Event::Screenshot { image, .. } = event {
                    return Some(image.clone());
                }
            }
            None
        });
        if let (Some(screenshot), Some(plot_rect)) = (screenshot, self.plot_rect) {
            if let Some(mut path) = rfd::FileDialog::new()
                .set_file_name(&self.default_filename)
                .add_filter("PNG Image", &["png"])
                .save_file()
            {
                path.set_extension("png");
                let pixels_per_point = ctx.pixels_per_point();
                let plot = screenshot.region(&plot_rect, Some(pixels_per_point));
                let result = image::save_buffer(
                    &path,
                    plot.as_raw(),
                    plot.width() as u32,
                    plot.height() as u32,
                    image::ColorType::Rgba8,
                );
                match result {
                    Ok(()) => eprintln!("Image saved to {}", path.display()),
                    Err(err) => eprintln!("Failed to save image to {}: {err}", path.display()),
                }
            }
        }
    }
}
