use eframe::egui;

pub(crate) struct SavePlotState {
    button_rect: Option<egui::Rect>,
    title: String,
    default_filename: String,
}

impl SavePlotState {
    pub(crate) fn new(title: impl Into<String>) -> Self {
        let title = title.into();
        let default_filename = format!("{title}.png");
        Self {
            button_rect: None,
            title,
            default_filename,
        }
    }

    pub(crate) fn show_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Button on the left
            let response = ui.button("Save Plot");
            self.button_rect = Some(response.rect);
            if response.clicked() {
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
        if let Some(screenshot) = screenshot {
            if let Some(mut path) = rfd::FileDialog::new()
                .set_file_name(&self.default_filename)
                .add_filter("PNG Image", &["png"])
                .save_file()
            {
                path.set_extension("png");
                let pixels_per_point = ctx.pixels_per_point();
                let width = screenshot.width() as u32;
                let height = screenshot.height() as u32;

                let mut img = image::RgbaImage::from_raw(width, height, screenshot.as_raw().to_vec())
                    .expect("screenshot buffer size mismatch");

                // Paint over the save button with the panel background color
                if let Some(button_rect) = self.button_rect {
                    let bg = ctx.style().visuals.panel_fill;
                    let pixel = image::Rgba([bg.r(), bg.g(), bg.b(), bg.a()]);

                    let min_x = (button_rect.min.x * pixels_per_point) as u32;
                    let min_y = (button_rect.min.y * pixels_per_point) as u32;
                    let max_x = ((button_rect.max.x * pixels_per_point) as u32).min(width);
                    let max_y = ((button_rect.max.y * pixels_per_point) as u32).min(height);

                    for y in min_y..max_y {
                        for x in min_x..max_x {
                            img.put_pixel(x, y, pixel);
                        }
                    }
                }

                let result = img.save(&path);
                match result {
                    Ok(()) => eprintln!("Image saved to {}", path.display()),
                    Err(err) => eprintln!("Failed to save image to {}: {err}", path.display()),
                }
            }
        }
    }
}
