//! A measurement app: measure impulse and frequency responses with sweeps and compare them.
//!
//! The left side bar lists all measurements. Those taken together, one per speaker and mic,
//! are grouped in a folder that can be expanded, hidden or deleted as a whole. Tick a
//! measurement to show it, click its color to change it, right-click it for more.
//!
//! "New measurement" opens the setup in the side bar: choose the devices, the speakers
//! (outputs) and microphones (inputs), the sweep, and start. Each speaker plays a sweep, one
//! after another, while all mics record. When it is done, the side bar goes back to the list,
//! with the new measurements in a new folder.
//!
//! The tabs show the magnitude, the phase and the impulse responses. The phase is relative to
//! each impulse response's peak, so the delays are taken out; the impulse responses keep them,
//! on a time axis from the start of the recording. Double-click a plot to reset its zoom.
//!
//! "Simulated device" in the setup measures a made-up room instead, to try the app without
//! audio hardware.
//!
//! ```sh
//! cargo run --release --example measurement_app
//! ```

mod devices;
mod measure;
mod views;

use devices::DeviceList;
use eframe::egui;
use measure::{Job, Outcome, Response, Settings};
use views::{Curves, FreqTrace, TimeTrace};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("audio-lab measurements"),
        ..Default::default()
    };
    eframe::run_native(
        "audio-lab measurements",
        options,
        Box::new(|cc| Ok(Box::new(App::new(&cc.egui_ctx)))),
    )
}

/// What the side bar shows.
#[derive(PartialEq)]
enum SideBar {
    Measurements,
    Setup,
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Magnitude,
    Phase,
    Impulse,
}

/// Measurements taken together: one per speaker and mic.
struct Group {
    id: u64,
    name: String,
    /// The sweep's frequency range: outside of it there is nothing to show.
    range: (f64, f64),
    measurements: Vec<Measurement>,
}

struct Measurement {
    /// `speaker → mic`.
    name: String,
    /// `group · speaker → mic`, for the plots.
    full_name: String,
    response: Response,
    visible: bool,
    color: egui::Color32,
    /// The frequency curves for the current smoothing, computed when first shown.
    curves: Option<Curves>,
    impulse: Vec<egui_plot::PlotPoint>,
}

/// Something to do to the list after drawing it.
enum Action {
    DeleteGroup(u64),
    Delete(u64, usize),
    /// Show only this measurement.
    Solo(u64, usize),
}

struct App {
    groups: Vec<Group>,
    next_group_id: u64,
    /// Measurements made so far, to give each a new color.
    num_measurements: usize,
    side_bar: SideBar,
    tab: Tab,
    smoothing_enabled: bool,
    /// 1/N octave.
    smoothing: f64,
    settings: Settings,
    devices: DeviceList,
    job: Option<Job>,
    /// Why the last measurement failed.
    error: Option<String>,
    /// Problems with single responses of the last measurement.
    warnings: Vec<String>,
}

impl App {
    fn new(ctx: &egui::Context) -> Self {
        Self {
            groups: Vec::new(),
            next_group_id: 1,
            num_measurements: 0,
            side_bar: SideBar::Measurements,
            tab: Tab::Magnitude,
            smoothing_enabled: true,
            smoothing: 6.0,
            settings: Settings::default(),
            devices: DeviceList::scan(ctx),
            job: None,
            error: None,
            warnings: Vec::new(),
        }
    }

    fn smoothing(&self) -> Option<f64> {
        self.smoothing_enabled.then_some(self.smoothing)
    }

    /// Picks up a finished measurement and goes back to the list.
    fn poll_job(&mut self) {
        let Some(result) = self.job.as_ref().and_then(Job::poll) else {
            return;
        };
        let job = self.job.take().expect("polled a job");
        match result {
            Ok(outcome) => {
                self.add_group(outcome, &job);
                self.side_bar = SideBar::Measurements;
                self.error = None;
            }
            Err(error) => self.error = Some(error),
        }
    }

    fn add_group(&mut self, outcome: Outcome, job: &Job) {
        let name = self.settings.name.trim().to_string();
        let name = if name.is_empty() {
            format!("Measurement {}", self.next_group_id)
        } else {
            name
        };
        let measurements = outcome
            .responses
            .into_iter()
            .map(|response| {
                let short = format!("{} → {}", response.speaker, response.mic);
                self.num_measurements += 1;
                Measurement {
                    full_name: format!("{name} · {short}"),
                    name: short,
                    impulse: views::impulse_points(&response.impulse_response, response.start),
                    response,
                    visible: true,
                    color: color(self.num_measurements - 1),
                    curves: None,
                }
            })
            .collect();
        self.groups.push(Group {
            id: self.next_group_id,
            name,
            range: (job.request.range.start, job.request.range.end),
            measurements,
        });
        self.next_group_id += 1;
        self.warnings = outcome.warnings;
        self.settings.name = next_name(&self.settings.name, self.next_group_id);
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::DeleteGroup(id) => self.groups.retain(|group| group.id != id),
            Action::Delete(id, index) => {
                if let Some(group) = self.groups.iter_mut().find(|group| group.id == id) {
                    group.measurements.remove(index);
                }
                self.groups.retain(|group| !group.measurements.is_empty());
            }
            Action::Solo(id, index) => {
                for group in &mut self.groups {
                    for (i, measurement) in group.measurements.iter_mut().enumerate() {
                        measurement.visible = group.id == id && i == index;
                    }
                }
            }
        }
    }

    fn measurements_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        let new = egui::Button::new("➕ New measurement")
            .min_size(egui::vec2(ui.available_width(), 28.0));
        if ui.add(new).clicked() {
            self.side_bar = SideBar::Setup;
        }
        if !self.warnings.is_empty() {
            ui.add_space(4.0);
            for warning in &self.warnings {
                ui.colored_label(egui::Color32::from_rgb(220, 140, 0), format!("⚠ {warning}"));
            }
            if ui.small_button("Dismiss").clicked() {
                self.warnings.clear();
            }
        }
        ui.separator();
        if self.groups.is_empty() {
            ui.weak("No measurements yet.");
            return;
        }

        let mut action = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            // newest on top
            for group in self.groups.iter_mut().rev() {
                let id = ui.make_persistent_id(("group", group.id));
                egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    id,
                    true,
                )
                .show_header(ui, |ui| {
                    let mut all = group.measurements.iter().all(|m| m.visible);
                    let some = group.measurements.iter().any(|m| m.visible);
                    let mixed = some && !all;
                    let checkbox = egui::Checkbox::new(&mut all, "").indeterminate(mixed);
                    if ui.add(checkbox).on_hover_text("Show or hide all").changed() {
                        group.measurements.iter_mut().for_each(|m| m.visible = all);
                    }
                    ui.label(egui::RichText::new(format!("📁 {}", group.name)).strong());
                    ui.weak(format!("({})", group.measurements.len()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("🗑")
                            .on_hover_text("Delete the folder")
                            .clicked()
                        {
                            action = Some(Action::DeleteGroup(group.id));
                        }
                    });
                })
                .body(|ui| {
                    for (index, m) in group.measurements.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut m.visible, "");
                            ui.scope(|ui| {
                                // a small swatch
                                ui.spacing_mut().interact_size.x = 18.0;
                                ui.color_edit_button_srgba(&mut m.color);
                            });
                            let label = ui
                                .add(egui::Label::new(&m.name).sense(egui::Sense::click()))
                                .on_hover_text(details(&m.response));
                            if label.clicked() {
                                m.visible = !m.visible;
                            }
                            label.context_menu(|ui| {
                                if ui.button("Show only this").clicked() {
                                    action = Some(Action::Solo(group.id, index));
                                    ui.close();
                                }
                                if ui.button("Delete").clicked() {
                                    action = Some(Action::Delete(group.id, index));
                                    ui.close();
                                }
                            });
                        });
                    }
                });
            }
        });
        if let Some(action) = action {
            self.apply(action);
        }
    }

    fn setup_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let back = ui.add_enabled(self.job.is_none(), egui::Button::new("⬅ Measurements"));
            if back.clicked() {
                self.side_bar = SideBar::Measurements;
            }
        });
        ui.separator();

        if let Some(job) = &self.job {
            ui.heading("Measuring…");
            ui.add_space(8.0);
            ui.add(egui::ProgressBar::new(job.progress()).animate(true));
            let remaining = job.remaining().as_secs();
            if remaining > 0 {
                ui.label(format!("About {remaining} s left. Keep it quiet."));
            } else {
                ui.label("Analyzing…");
            }
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
            return;
        }

        self.devices.poll();
        let (num_outputs, num_inputs) = if self.settings.simulated {
            (measure::SIMULATED_CHANNELS, measure::SIMULATED_CHANNELS)
        } else {
            let devices = self.devices.devices().cloned().unwrap_or_default();
            (
                devices.num_outputs(self.settings.output_device.as_deref()),
                devices.num_inputs(self.settings.input_device.as_deref()),
            )
        };
        self.settings.sync_channels(num_outputs, num_inputs);

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.settings.name)
                        .desired_width(f32::INFINITY),
                );
            });

            section(ui, "Devices");
            self.device_choice(ui);

            section(ui, "Speakers (outputs)");
            channel_choice(ui, "outputs", &mut self.settings.outputs);

            section(ui, "Microphones (inputs)");
            channel_choice(ui, "inputs", &mut self.settings.inputs);

            section(ui, "Sweep");
            sweep_settings(ui, &mut self.settings);

            ui.add_space(12.0);
            let problem = self.settings.problem();
            let request = self.settings.request();
            let start = egui::Button::new(egui::RichText::new("▶ Start").strong())
                .min_size(egui::vec2(ui.available_width(), 32.0));
            if ui.add_enabled(problem.is_none(), start).clicked() {
                self.error = None;
                self.job = Some(Job::start(request.clone(), ui.ctx()));
            }
            match problem {
                Some(problem) => {
                    ui.weak(problem);
                }
                None if !self.settings.simulated => {
                    ui.weak(format!(
                        "Takes about {:.0} s.",
                        request.duration().as_secs_f64()
                    ));
                }
                None => {}
            }
            if let Some(error) = &self.error {
                ui.add_space(4.0);
                ui.colored_label(egui::Color32::RED, error);
            }
        });
    }

    fn device_choice(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.settings.simulated,
            "Simulated device (no hardware)",
        );
        if self.settings.simulated {
            return;
        }
        let devices = match &self.devices {
            DeviceList::Scanning(_) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Looking for devices…");
                });
                return;
            }
            DeviceList::Failed(error) => {
                ui.colored_label(egui::Color32::RED, format!("No devices: {error}"));
                if ui.button("Retry").clicked() {
                    self.devices = DeviceList::scan(ui.ctx());
                }
                return;
            }
            DeviceList::Ready(devices) => devices.clone(),
        };
        egui::Grid::new("devices").num_columns(2).show(ui, |ui| {
            ui.label("Output");
            device_combo(
                ui,
                "output",
                &mut self.settings.output_device,
                &devices.outputs,
                &devices.default_output,
            );
            ui.end_row();
            ui.label("Input");
            device_combo(
                ui,
                "input",
                &mut self.settings.input_device,
                &devices.inputs,
                &devices.default_input,
            );
            ui.end_row();
        });
        if ui.small_button("⟳ Refresh devices").clicked() {
            self.devices = DeviceList::scan(ui.ctx());
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Magnitude, "Magnitude");
            ui.selectable_value(&mut self.tab, Tab::Phase, "Phase");
            ui.selectable_value(&mut self.tab, Tab::Impulse, "Impulse response");
            ui.separator();
            ui.add_enabled_ui(self.tab != Tab::Impulse, |ui| {
                ui.checkbox(&mut self.smoothing_enabled, "Smoothing");
                let slider = egui::Slider::new(&mut self.smoothing, 1.0..=48.0)
                    .text("1/N octave")
                    .logarithmic(true);
                ui.add_enabled(self.smoothing_enabled, slider);
            });
        });
    }

    fn plots(&mut self, ui: &mut egui::Ui) {
        if !self
            .groups
            .iter()
            .any(|g| g.measurements.iter().any(|m| m.visible))
        {
            ui.centered_and_justified(|ui| {
                ui.weak(if self.groups.is_empty() {
                    "No measurements yet. Click \"New measurement\" to start one."
                } else {
                    "Tick a measurement in the side bar to show it."
                });
            });
            return;
        }

        // compute the curves of what is shown, for the current smoothing
        let smoothing = self.smoothing();
        if self.tab != Tab::Impulse {
            for group in &mut self.groups {
                for m in group.measurements.iter_mut().filter(|m| m.visible) {
                    if m.curves.as_ref().is_none_or(|c| c.smoothing != smoothing) {
                        m.curves = Some(views::curves(
                            &m.response.frequency_response,
                            group.range,
                            smoothing,
                        ));
                    }
                }
            }
        }

        let visible = || {
            self.groups
                .iter()
                .flat_map(|g| &g.measurements)
                .filter(|m| m.visible)
        };
        let range = self
            .groups
            .iter()
            .filter(|g| g.measurements.iter().any(|m| m.visible))
            .fold((f64::INFINITY, 0.0f64), |(low, high), g| {
                (low.min(g.range.0), high.max(g.range.1))
            });
        let freq_traces = || -> Vec<FreqTrace> {
            visible()
                .filter_map(|m| {
                    Some(FreqTrace {
                        name: &m.full_name,
                        color: m.color,
                        curves: m.curves.as_ref()?,
                    })
                })
                .collect()
        };
        match self.tab {
            Tab::Magnitude => views::magnitude_plot(ui, &freq_traces(), range),
            Tab::Phase => views::phase_plot(ui, &freq_traces(), range),
            Tab::Impulse => {
                let traces: Vec<_> = visible()
                    .map(|m| TimeTrace {
                        name: &m.full_name,
                        color: m.color,
                        points: &m.impulse,
                    })
                    .collect();
                views::impulse_plot(ui, &traces);
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_job();

        egui::Panel::left("side_bar")
            .resizable(true)
            .default_size(320.0)
            .min_size(260.0)
            .show(ui, |ui| match self.side_bar {
                SideBar::Measurements => self.measurements_panel(ui),
                SideBar::Setup => self.setup_panel(ui),
            });
        egui::Panel::top("tabs").show(ui, |ui| {
            ui.add_space(2.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        egui::CentralPanel::default().show(ui, |ui| self.plots(ui));
    }
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(title).strong());
    ui.separator();
}

fn device_combo(
    ui: &mut egui::Ui,
    id: &str,
    chosen: &mut Option<String>,
    devices: &[devices::Device],
    default: &str,
) {
    let default_text = if default.is_empty() {
        "System default".to_string()
    } else {
        format!("System default ({default})")
    };
    let describe = |device: &devices::Device| match device.num_channels {
        Some(n) => format!("{} ({n} ch)", device.name),
        None => device.name.clone(),
    };
    let selected = match chosen.as_deref() {
        None => default_text.clone(),
        Some(name) => name.to_string(),
    };
    egui::ComboBox::from_id_salt(id)
        .selected_text(selected)
        .width(ui.available_width())
        .truncate()
        .show_ui(ui, |ui| {
            ui.selectable_value(chosen, None, default_text);
            for device in devices {
                ui.selectable_value(chosen, Some(device.name.clone()), describe(device));
            }
        });
}

/// A checkbox and a name per channel.
fn channel_choice(ui: &mut egui::Ui, id: &str, channels: &mut [measure::Channel]) {
    egui::Grid::new(id)
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            for (index, channel) in channels.iter_mut().enumerate() {
                ui.checkbox(&mut channel.enabled, format!("{}", index + 1));
                ui.add_enabled(
                    channel.enabled,
                    egui::TextEdit::singleline(&mut channel.name).desired_width(180.0),
                );
                ui.end_row();
            }
        });
}

fn sweep_settings(ui: &mut egui::Ui, settings: &mut Settings) {
    egui::Grid::new("sweep").num_columns(2).show(ui, |ui| {
        ui.label("Sample rate");
        egui::ComboBox::from_id_salt("sample_rate")
            .selected_text(format!("{} Hz", settings.sample_rate))
            .show_ui(ui, |ui| {
                for rate in [44_100, 48_000, 88_200, 96_000] {
                    ui.selectable_value(&mut settings.sample_rate, rate, format!("{rate} Hz"));
                }
            });
        ui.end_row();

        ui.label("Length");
        let seconds = |exponent: u32| (1u64 << exponent) as f64 / f64::from(settings.sample_rate);
        egui::ComboBox::from_id_salt("sweep_len")
            .selected_text(format!("{:.1} s", seconds(settings.sweep_exponent)))
            .show_ui(ui, |ui| {
                for exponent in 14..=20 {
                    let text = format!("{:.1} s (2^{exponent})", seconds(exponent));
                    ui.selectable_value(&mut settings.sweep_exponent, exponent, text);
                }
            })
            .response
            .on_hover_text("Longer sweeps give more signal-to-noise.");
        ui.end_row();

        ui.label("Level");
        ui.add(egui::Slider::new(&mut settings.level_db, -60.0..=0.0).suffix(" dBFS"))
            .on_hover_text("Peak level of the sweep. Start low and turn it up.");
        ui.end_row();

        let nyquist = f64::from(settings.sample_rate) / 2.0;
        ui.label("From");
        ui.add(
            egui::DragValue::new(&mut settings.freq_low)
                .range(1.0..=nyquist)
                .speed(1.0)
                .suffix(" Hz"),
        );
        ui.end_row();
        ui.label("To");
        ui.add(
            egui::DragValue::new(&mut settings.freq_high)
                .range(1.0..=nyquist)
                .speed(10.0)
                .suffix(" Hz"),
        );
        ui.end_row();

        ui.label("Tail");
        ui.add(
            egui::DragValue::new(&mut settings.tail_ms)
                .range(200.0..=10_000.0)
                .speed(10.0)
                .suffix(" ms"),
        )
        .on_hover_text(
            "Recorded after each sweep: must be longer than the device latency plus the reverb.",
        );
        ui.end_row();

        ui.label("IR window");
        ui.add(
            egui::DragValue::new(&mut settings.ir_length_ms)
                .range(10.0..=5_000.0)
                .speed(5.0)
                .suffix(" ms"),
        )
        .on_hover_text("How much of each impulse response is kept, after its start.");
        ui.end_row();
    });
}

/// Peak level, IR start and peak, shown when hovering a measurement.
fn details(response: &Response) -> String {
    let sample_rate = response.impulse_response.sample_rate();
    let ms = |samples: usize| samples as f64 / sample_rate * 1000.0;
    let peak = response.start + measure::peak_index(&response.impulse_response);
    format!(
        "Recording peak: {:.1} dBFS\nIR start: {:.2} ms\nIR peak: {:.2} ms\n\nTimes include the device latency.",
        response.peak_db,
        ms(response.start),
        ms(peak),
    )
}

/// `Measurement 3` becomes `Measurement 4`; a name without a number becomes `Measurement n`.
fn next_name(name: &str, next_id: u64) -> String {
    let trimmed = name.trim();
    match trimmed.rsplit_once(' ') {
        Some((stem, number)) if number.parse::<u64>().is_ok() => {
            format!("{stem} {}", number.parse::<u64>().unwrap() + 1)
        }
        _ => format!("Measurement {next_id}"),
    }
}

/// The `index`th of a series of well distinguishable colors.
fn color(index: usize) -> egui::Color32 {
    let golden_ratio = (5f32.sqrt() - 1.0) / 2.0;
    egui::epaint::Hsva::new(index as f32 * golden_ratio, 0.85, 0.6, 1.0).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_count_up() {
        assert_eq!(next_name("Measurement 3", 4), "Measurement 4");
        assert_eq!(next_name("Living room 9", 2), "Living room 10");
        assert_eq!(next_name("Living room", 2), "Measurement 2");
    }
}
