//! The window: measurements on the left, the plots in tabs on the right.

use std::rc::Rc;
use std::time::Duration;

use gpui_kit::component::button::{Button, ButtonVariant, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::color_picker::{ColorPicker, ColorPickerEvent, ColorPickerState};
use gpui_kit::component::menu::{ContextMenuExt as _, PopupMenuItem};
use gpui_kit::component::progress::Progress;
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::slider::{Slider, SliderEvent, SliderScale, SliderState};
use gpui_kit::component::{
    Disableable as _, Icon, IconName, Sizable as _, StyledExt as _, TitleBar, WindowExt as _,
    h_flex, h_resizable, resizable_panel, v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::curves::{Curves, impulse_points};
use crate::measure::{self, Job, Outcome, Response};
use crate::plot::{Plot, PlotKind, Series, Trace};
use crate::setup::{Setup, SetupEvent};
use crate::theme::{palette, trace_color};

actions!(
    measurement,
    [NewMeasurement, ShowMagnitude, ShowPhase, ShowImpulse]
);

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("ctrl-n", NewMeasurement, None),
        KeyBinding::new("alt-1", ShowMagnitude, None),
        KeyBinding::new("alt-2", ShowPhase, None),
        KeyBinding::new("alt-3", ShowImpulse, None),
    ]);
}

#[derive(Clone, Copy, PartialEq)]
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

impl Tab {
    const ALL: [Tab; 3] = [Tab::Magnitude, Tab::Phase, Tab::Impulse];

    fn title(self) -> &'static str {
        match self {
            Tab::Magnitude => "Magnitude",
            Tab::Phase => "Phase",
            Tab::Impulse => "Impulse response",
        }
    }
}

/// Measurements taken together: one per speaker and mic.
struct Group {
    id: u64,
    name: SharedString,
    /// The sweep's frequency range: outside of it there is nothing to show.
    range: (f64, f64),
    expanded: bool,
    measurements: Vec<Measurement>,
}

struct Measurement {
    id: u64,
    /// `speaker → mic`.
    name: SharedString,
    /// `group · speaker → mic`, for the plots.
    full_name: SharedString,
    response: Response,
    visible: bool,
    color: Hsla,
    color_picker: Entity<ColorPickerState>,
    /// The frequency curves for the current smoothing, computed when first shown.
    curves: Option<Rc<Curves>>,
    impulse: Rc<Vec<(f64, f64)>>,
    _subscription: Subscription,
}

/// Something to do to the list.
#[derive(Clone, Copy)]
enum Action {
    DeleteGroup(u64),
    Delete(u64),
    /// Show only this measurement.
    Solo(u64),
}

pub struct MeasurementApp {
    focus_handle: FocusHandle,
    groups: Vec<Group>,
    next_group_id: u64,
    next_measurement_id: u64,
    side_bar: SideBar,
    tab: Tab,
    smoothing_enabled: bool,
    /// 1/N octave.
    smoothing: f64,
    smoothing_slider: Entity<SliderState>,
    setup: Entity<Setup>,
    magnitude: Entity<Plot>,
    phase: Entity<Plot>,
    impulse: Entity<Plot>,
    job: Option<Job>,
    _job_task: Option<Task<()>>,
    _ticker: Option<Task<()>>,
    /// Problems with single responses of the last measurement.
    warnings: Vec<String>,
    _subscriptions: Vec<Subscription>,
}

impl MeasurementApp {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let setup = cx.new(|cx| Setup::new(window, cx));
        let smoothing = 6.0;
        let smoothing_slider = cx.new(|_| {
            SliderState::new()
                .min(1.0)
                .max(48.0)
                .step(1.0)
                .scale(SliderScale::Logarithmic)
                .default_value(smoothing as f32)
        });
        let subscriptions =
            vec![
                cx.subscribe_in(&setup, window, |this, _, event: &SetupEvent, window, cx| {
                    match event {
                        SetupEvent::Start(request) => this.start(request.clone(), window, cx),
                        SetupEvent::Close => {
                            this.side_bar = SideBar::Measurements;
                            cx.notify();
                        }
                    }
                }),
                cx.subscribe(&smoothing_slider, |this, _, event: &SliderEvent, cx| {
                    let (SliderEvent::Change(value) | SliderEvent::Release(value)) = event;
                    let value = f64::from(value.start()).round();
                    if value != this.smoothing {
                        this.smoothing = value;
                        this.refresh_plots(cx);
                    }
                }),
            ];
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        Self {
            focus_handle,
            groups: Vec::new(),
            next_group_id: 1,
            next_measurement_id: 1,
            side_bar: SideBar::Measurements,
            tab: Tab::Magnitude,
            smoothing_enabled: true,
            smoothing,
            smoothing_slider,
            setup,
            magnitude: cx.new(|_| Plot::new(PlotKind::Magnitude)),
            phase: cx.new(|_| Plot::new(PlotKind::Phase)),
            impulse: cx.new(|_| Plot::new(PlotKind::Impulse)),
            job: None,
            _job_task: None,
            _ticker: None,
            warnings: Vec::new(),
            _subscriptions: subscriptions,
        }
    }

    fn smoothing(&self) -> Option<f64> {
        self.smoothing_enabled.then_some(self.smoothing)
    }

    fn num_measurements(&self) -> usize {
        self.groups.iter().map(|g| g.measurements.len()).sum()
    }

    fn num_visible(&self) -> usize {
        self.groups
            .iter()
            .flat_map(|g| &g.measurements)
            .filter(|m| m.visible)
            .count()
    }

    /// Runs `request` in the background, and a timer for the progress bar.
    fn start(&mut self, request: measure::Request, window: &mut Window, cx: &mut Context<Self>) {
        if self.job.is_some() {
            return;
        }
        self.setup.update(cx, |setup, cx| setup.set_error(None, cx));
        self.job = Some(Job::new(request.clone()));
        let work = cx.background_spawn(async move { measure::run(&request) });
        self._job_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = work.await;
            this.update_in(cx, |this, window, cx| this.finish(result, window, cx))
                .ok();
        }));
        // redraws the progress until the measurement is done
        self._ticker = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(100))
                    .await;
                if this.update(cx, |_, cx| cx.notify()).is_err() {
                    break;
                }
            }
        }));
        cx.notify();
    }

    /// Picks up a finished measurement and goes back to the list.
    fn finish(
        &mut self,
        result: Result<Outcome, String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(job) = self.job.take() else { return };
        self._ticker = None;
        match result {
            Ok(outcome) => {
                self.add_group(outcome, &job, window, cx);
                self.side_bar = SideBar::Measurements;
                let next_id = self.next_group_id;
                self.setup
                    .update(cx, |setup, cx| setup.count_up(next_id, window, cx));
            }
            Err(error) => self
                .setup
                .update(cx, |setup, cx| setup.set_error(Some(error), cx)),
        }
        self.refresh_plots(cx);
    }

    fn add_group(
        &mut self,
        outcome: Outcome,
        job: &Job,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let name = self.setup.read(cx).name();
        let name = if name.is_empty() {
            format!("Measurement {}", self.next_group_id)
        } else {
            name
        };
        let mut measurements = Vec::new();
        for response in outcome.responses {
            let id = self.next_measurement_id;
            self.next_measurement_id += 1;
            let color = trace_color(id as usize - 1);
            let color_picker = cx.new(|cx| ColorPickerState::new(window, cx).default_value(color));
            let subscription = cx.subscribe(
                &color_picker,
                move |this, _, event: &ColorPickerEvent, cx| {
                    let ColorPickerEvent::Change(Some(color)) = event else {
                        return;
                    };
                    if let Some(m) = this.measurement_mut(id) {
                        m.color = *color;
                    }
                    this.refresh_plots(cx);
                },
            );
            let short = format!("{} → {}", response.speaker, response.mic);
            measurements.push(Measurement {
                id,
                full_name: format!("{name} · {short}").into(),
                name: short.into(),
                impulse: Rc::new(impulse_points(&response.impulse_response, response.start)),
                response,
                visible: true,
                color,
                color_picker,
                curves: None,
                _subscription: subscription,
            });
        }
        self.groups.push(Group {
            id: self.next_group_id,
            name: name.into(),
            range: (job.request.range.start, job.request.range.end),
            expanded: true,
            measurements,
        });
        self.next_group_id += 1;
        self.warnings = outcome.warnings;
    }

    fn measurement_mut(&mut self, id: u64) -> Option<&mut Measurement> {
        self.groups
            .iter_mut()
            .flat_map(|g| &mut g.measurements)
            .find(|m| m.id == id)
    }

    fn apply(&mut self, action: Action, cx: &mut Context<Self>) {
        match action {
            Action::DeleteGroup(id) => self.groups.retain(|group| group.id != id),
            Action::Delete(id) => {
                for group in &mut self.groups {
                    group.measurements.retain(|m| m.id != id);
                }
                self.groups.retain(|group| !group.measurements.is_empty());
            }
            Action::Solo(id) => {
                for m in self.groups.iter_mut().flat_map(|g| &mut g.measurements) {
                    m.visible = m.id == id;
                }
            }
        }
        self.refresh_plots(cx);
    }

    /// Asks before deleting a measurement or a folder, then deletes it.
    fn confirm_delete(&self, action: Action, window: &mut Window, cx: &mut Context<Self>) {
        let (title, description) = match action {
            Action::DeleteGroup(id) => {
                let Some(group) = self.groups.iter().find(|g| g.id == id) else {
                    return;
                };
                let count = group.measurements.len();
                (
                    format!("Delete “{}”?", group.name),
                    format!(
                        "Its {count} measurement{} will be deleted. This can't be undone.",
                        if count == 1 { "" } else { "s" }
                    ),
                )
            }
            Action::Delete(id) => {
                let Some(m) = self
                    .groups
                    .iter()
                    .flat_map(|g| &g.measurements)
                    .find(|m| m.id == id)
                else {
                    return;
                };
                (
                    format!("Delete “{}”?", m.full_name),
                    "This can't be undone.".to_string(),
                )
            }
            Action::Solo(_) => return,
        };
        let app = cx.entity();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let app = app.clone();
            alert
                .title(title.clone())
                .description(description.clone())
                .confirm()
                .ok_text("Delete")
                .ok_variant(ButtonVariant::Danger)
                .on_ok(move |_, _, cx| {
                    app.update(cx, |this, cx| this.apply(action, cx));
                    true
                })
        });
    }

    fn set_tab(&mut self, tab: Tab, cx: &mut Context<Self>) {
        self.tab = tab;
        self.refresh_plots(cx);
    }

    fn new_measurement(&mut self, cx: &mut Context<Self>) {
        self.side_bar = SideBar::Setup;
        cx.notify();
    }

    /// Hands the visible measurements to the plot of the current tab, computing the curves
    /// for the current smoothing where needed.
    fn refresh_plots(&mut self, cx: &mut Context<Self>) {
        let smoothing = self.smoothing();
        let frequency = self.tab != Tab::Impulse;
        let mut range = (f64::INFINITY, 0.0f64);
        let mut traces = Vec::new();
        for group in &mut self.groups {
            for m in group.measurements.iter_mut().filter(|m| m.visible) {
                range = (range.0.min(group.range.0), range.1.max(group.range.1));
                let series = if frequency {
                    if m.curves.as_ref().is_none_or(|c| c.smoothing != smoothing) {
                        m.curves = Some(Rc::new(Curves::new(
                            &m.response.frequency_response,
                            group.range,
                            smoothing,
                        )));
                    }
                    Series::Frequency(m.curves.clone().expect("just computed"))
                } else {
                    Series::Points(m.impulse.clone())
                };
                traces.push(Trace {
                    name: m.full_name.clone(),
                    color: m.color,
                    series,
                });
            }
        }
        if !range.0.is_finite() {
            range = (20.0, 20_000.0);
        }
        let plot = match self.tab {
            Tab::Magnitude => &self.magnitude,
            Tab::Phase => &self.phase,
            Tab::Impulse => &self.impulse,
        };
        plot.update(cx, |plot, cx| plot.set_traces(traces, range, cx));
        cx.notify();
    }

    fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        TitleBar::new().child(
            h_flex()
                .w_full()
                .pr_2()
                .justify_between()
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            Icon::new(gpui_kit::assets::IconName::AudioWaveform)
                                .small()
                                .text_color(palette::accent()),
                        )
                        .child(div().text_sm().child("audio-lab"))
                        .child(
                            div()
                                .text_sm()
                                .text_color(palette::text_muted())
                                .child("Measurements"),
                        ),
                )
                .child(
                    Button::new("title-new")
                        .ghost()
                        .xsmall()
                        .icon(IconName::Plus)
                        .label("New measurement")
                        .disabled(self.job.is_some())
                        .on_click(cx.listener(|this, _, _, cx| this.new_measurement(cx))),
                ),
        )
    }

    fn render_side_bar(&self, cx: &mut Context<Self>) -> AnyElement {
        if let Some(job) = &self.job {
            return self.render_progress(job).into_any_element();
        }
        match self.side_bar {
            SideBar::Setup => self.setup.clone().into_any_element(),
            SideBar::Measurements => self.render_list(cx).into_any_element(),
        }
    }

    fn render_progress(&self, job: &Job) -> impl IntoElement {
        let remaining = job.remaining().as_secs();
        v_flex().size_full().child(panel_header("Measuring")).child(
            v_flex()
                .p_3()
                .gap_3()
                .child(Progress::new("measure-progress").value(job.progress() * 100.0))
                .child(
                    div()
                        .text_sm()
                        .text_color(palette::text_muted())
                        .child(if remaining > 0 {
                            format!("About {remaining} s left. Keep it quiet.")
                        } else {
                            "Analyzing…".to_string()
                        }),
                ),
        )
    }

    fn render_list(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut list = v_flex().w_full().py_1();
        // newest on top
        for group in self.groups.iter().rev() {
            list = list.child(self.render_group_row(group, cx));
            if group.expanded {
                for m in &group.measurements {
                    list = list.child(self.render_measurement_row(m, cx));
                }
            }
        }
        v_flex()
            .size_full()
            .child(
                h_flex()
                    .h(px(36.0))
                    .flex_shrink_0()
                    .pl_3()
                    .pr_2()
                    .justify_between()
                    .border_b_1()
                    .border_color(palette::border_variant())
                    .child(
                        div()
                            .text_xs()
                            .font_medium()
                            .text_color(palette::text_muted())
                            .child("MEASUREMENTS"),
                    )
                    .child(
                        Button::new("new")
                            .ghost()
                            .xsmall()
                            .icon(IconName::Plus)
                            .tooltip("New measurement (Ctrl+N)")
                            .on_click(cx.listener(|this, _, _, cx| this.new_measurement(cx))),
                    ),
            )
            .when(!self.warnings.is_empty(), |this| {
                this.child(self.render_warnings(cx))
            })
            .child(
                div().flex_1().min_h_0().child(
                    v_flex()
                        .id("measurement-list")
                        .size_full()
                        .overflow_y_scrollbar()
                        .map(|this| {
                            if self.groups.is_empty() {
                                this.child(
                                    v_flex()
                                        .p_4()
                                        .gap_3()
                                        .items_center()
                                        .child(
                                            div()
                                                .text_sm()
                                                .text_color(palette::text_placeholder())
                                                .child("No measurements yet."),
                                        )
                                        .child(
                                            Button::new("first")
                                                .small()
                                                .icon(IconName::Plus)
                                                .label("New measurement")
                                                .on_click(cx.listener(|this, _, _, cx| {
                                                    this.new_measurement(cx)
                                                })),
                                        ),
                                )
                            } else {
                                this.child(list)
                            }
                        }),
                ),
            )
    }

    fn render_warnings(&self, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .m_2()
            .p_2()
            .gap_1()
            .rounded(px(4.0))
            .border_1()
            .border_color(palette::warning().opacity(0.4))
            .bg(palette::warning().opacity(0.08))
            .children(self.warnings.iter().map(|warning| {
                h_flex()
                    .gap_2()
                    .items_start()
                    .child(
                        Icon::new(IconName::TriangleAlert)
                            .xsmall()
                            .text_color(palette::warning()),
                    )
                    .child(
                        div()
                            .flex_1()
                            .text_xs()
                            .text_color(palette::text())
                            .child(warning.clone()),
                    )
            }))
            .child(
                h_flex().justify_end().child(
                    Button::new("dismiss")
                        .ghost()
                        .xsmall()
                        .label("Dismiss")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.warnings.clear();
                            cx.notify();
                        })),
                ),
            )
    }

    fn render_group_row(&self, group: &Group, cx: &mut Context<Self>) -> impl IntoElement {
        let id = group.id;
        let all = group.measurements.iter().all(|m| m.visible);
        let some = group.measurements.iter().any(|m| m.visible);
        let row_group = SharedString::from(format!("group-row-{id}"));
        h_flex()
            .id(("group", id))
            .group(row_group.clone())
            .h(px(26.0))
            .pl_2()
            .pr_1()
            .gap_1p5()
            .w_full()
            .cursor_pointer()
            .hover(|style| style.bg(palette::hover()))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(group) = this.groups.iter_mut().find(|g| g.id == id) {
                    group.expanded = !group.expanded;
                }
                cx.notify();
            }))
            .child(
                Icon::new(if group.expanded {
                    IconName::ChevronDown
                } else {
                    IconName::ChevronRight
                })
                .xsmall()
                .text_color(palette::text_muted()),
            )
            .child(
                Icon::new(if group.expanded {
                    IconName::FolderOpen
                } else {
                    IconName::Folder
                })
                .small()
                .text_color(palette::text_muted()),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .text_sm()
                    .text_color(if some {
                        palette::text()
                    } else {
                        palette::text_placeholder()
                    })
                    .child(group.name.clone()),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(palette::text_placeholder())
                    .child(format!("{}", group.measurements.len())),
            )
            .child(
                div()
                    .invisible()
                    .group_hover(row_group.clone(), |style| style.visible())
                    .child(
                        Button::new(("delete-group", id))
                            .ghost()
                            .xsmall()
                            .icon(IconName::Delete)
                            .tooltip("Delete the folder")
                            .on_click(cx.listener(move |this, _, window, cx| {
                                cx.stop_propagation();
                                this.confirm_delete(Action::DeleteGroup(id), window, cx);
                            })),
                    ),
            )
            .child(
                Checkbox::new(("group-visible", id))
                    .small()
                    .checked(all)
                    .when(some && !all, |this| this.tooltip("Some are shown"))
                    .when(!(some && !all), |this| this.tooltip("Show or hide all"))
                    .on_change(cx.listener(move |this, checked: &bool, _, cx| {
                        cx.stop_propagation();
                        if let Some(group) = this.groups.iter_mut().find(|g| g.id == id) {
                            group
                                .measurements
                                .iter_mut()
                                .for_each(|m| m.visible = *checked);
                        }
                        this.refresh_plots(cx);
                    })),
            )
    }

    fn render_measurement_row(&self, m: &Measurement, cx: &mut Context<Self>) -> impl IntoElement {
        let id = m.id;
        let details = details(&m.response);
        let app = cx.entity();
        h_flex()
            .id(("measurement", id))
            .h(px(26.0))
            .pl(px(34.0))
            .pr_1()
            .gap_2()
            .w_full()
            .cursor_pointer()
            .hover(|style| style.bg(palette::hover()))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(m) = this.measurement_mut(id) {
                    m.visible = !m.visible;
                }
                this.refresh_plots(cx);
            }))
            .child(
                div()
                    .id(("color", id))
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(ColorPicker::new(&m.color_picker).xsmall()),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .text_sm()
                    .text_color(if m.visible {
                        palette::text()
                    } else {
                        palette::text_placeholder()
                    })
                    .child(m.name.clone()),
            )
            .child(
                Checkbox::new(("visible", id))
                    .small()
                    .checked(m.visible)
                    .on_change(cx.listener(move |this, checked: &bool, _, cx| {
                        cx.stop_propagation();
                        if let Some(m) = this.measurement_mut(id) {
                            m.visible = *checked;
                        }
                        this.refresh_plots(cx);
                    })),
            )
            .context_menu(move |mut menu, _, _| {
                let solo = app.clone();
                let delete = app.clone();
                for line in &details {
                    menu = menu.label(line.clone());
                }
                menu.separator()
                    .item(
                        PopupMenuItem::new("Show only this").on_click(move |_, _, cx| {
                            solo.update(cx, |this, cx| this.apply(Action::Solo(id), cx));
                        }),
                    )
                    .separator()
                    .item(
                        PopupMenuItem::new("Delete…").on_click(move |_, window, cx| {
                            delete.update(cx, |this, cx| {
                                this.confirm_delete(Action::Delete(id), window, cx)
                            });
                        }),
                    )
            })
    }

    fn render_tab_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
            .h(px(34.0))
            .flex_shrink_0()
            .bg(palette::panel())
            .border_b_1()
            .border_color(palette::border_variant())
            .children(Tab::ALL.into_iter().map(|tab| {
                let active = tab == self.tab;
                h_flex()
                    .id(tab.title())
                    .h_full()
                    .px_4()
                    .border_r_1()
                    .border_color(palette::border_variant())
                    .text_sm()
                    .cursor_pointer()
                    .map(|this| {
                        if active {
                            // the active tab is part of the plot below it
                            this.bg(palette::editor())
                                .text_color(palette::text())
                                .mb(px(-1.0))
                                .h(px(35.0))
                        } else {
                            this.text_color(palette::text_muted())
                                .hover(|style| style.bg(palette::hover()))
                        }
                    })
                    .on_click(cx.listener(move |this, _, _, cx| this.set_tab(tab, cx)))
                    .child(tab.title())
            }))
    }

    fn render_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let frequency = self.tab != Tab::Impulse;
        h_flex()
            .h(px(36.0))
            .flex_shrink_0()
            .px_3()
            .gap_3()
            .bg(palette::editor())
            .border_b_1()
            .border_color(palette::border_variant())
            .when(frequency, |this| {
                this.child(
                    Checkbox::new("smoothing")
                        .small()
                        .label("Smoothing")
                        .checked(self.smoothing_enabled)
                        .on_change(cx.listener(|this, checked: &bool, _, cx| {
                            this.smoothing_enabled = *checked;
                            this.refresh_plots(cx);
                        })),
                )
                .child(
                    div()
                        .w(px(160.0))
                        .child(Slider::new(&self.smoothing_slider).disabled(!self.smoothing_enabled)),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(if self.smoothing_enabled {
                            palette::text_muted()
                        } else {
                            palette::text_placeholder()
                        })
                        .child(format!("1/{:.0} octave", self.smoothing)),
                )
            })
            .child(div().flex_1())
            .child(
                div()
                    .text_xs()
                    .text_color(palette::text_placeholder())
                    .child("Scroll to zoom · drag to pan · right-drag to zoom in · double-click to fit"),
            )
    }

    fn render_plot(&self, cx: &mut Context<Self>) -> AnyElement {
        if self.num_visible() == 0 {
            let message = if self.groups.is_empty() {
                "No measurements yet."
            } else {
                "Tick a measurement in the side bar to show it."
            };
            return v_flex()
                .size_full()
                .items_center()
                .justify_center()
                .gap_3()
                .bg(palette::editor())
                .child(
                    Icon::new(gpui_kit::assets::IconName::AudioWaveform)
                        .size(px(48.0))
                        .text_color(palette::border()),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(palette::text_placeholder())
                        .child(message),
                )
                .when(self.groups.is_empty() && self.job.is_none(), |this| {
                    this.child(
                        Button::new("empty-new")
                            .small()
                            .icon(IconName::Plus)
                            .label("New measurement")
                            .on_click(cx.listener(|this, _, _, cx| this.new_measurement(cx))),
                    )
                })
                .into_any_element();
        }
        match self.tab {
            Tab::Magnitude => self.magnitude.clone().into_any_element(),
            Tab::Phase => self.phase.clone().into_any_element(),
            Tab::Impulse => self.impulse.clone().into_any_element(),
        }
    }

    fn render_status_bar(&self) -> impl IntoElement {
        let total = self.num_measurements();
        let shown = self.num_visible();
        h_flex()
            .h(px(26.0))
            .flex_shrink_0()
            .px_3()
            .gap_3()
            .justify_between()
            .bg(palette::chrome())
            .border_t_1()
            .border_color(palette::border())
            .text_xs()
            .text_color(palette::text_muted())
            .child(match &self.job {
                Some(job) => h_flex()
                    .gap_2()
                    .child(
                        div().w(px(120.0)).child(
                            Progress::new("status-progress")
                                .small()
                                .value(job.progress() * 100.0),
                        ),
                    )
                    .child("Measuring…"),
                None => h_flex().child(if total == 0 {
                    "No measurements".to_string()
                } else {
                    format!(
                        "{total} measurement{} · {shown} shown",
                        if total == 1 { "" } else { "s" }
                    )
                }),
            })
            .when(!self.warnings.is_empty(), |this| {
                this.child(
                    h_flex()
                        .gap_1()
                        .text_color(palette::warning())
                        .child(Icon::new(IconName::TriangleAlert).xsmall())
                        .child(format!(
                            "{} warning{}",
                            self.warnings.len(),
                            if self.warnings.len() == 1 { "" } else { "s" }
                        )),
                )
            })
    }
}

impl Focusable for MeasurementApp {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for MeasurementApp {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("app")
            .key_context("MeasurementApp")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &NewMeasurement, _, cx| {
                if this.job.is_none() {
                    this.new_measurement(cx)
                }
            }))
            .on_action(
                cx.listener(|this, _: &ShowMagnitude, _, cx| this.set_tab(Tab::Magnitude, cx)),
            )
            .on_action(cx.listener(|this, _: &ShowPhase, _, cx| this.set_tab(Tab::Phase, cx)))
            .on_action(cx.listener(|this, _: &ShowImpulse, _, cx| this.set_tab(Tab::Impulse, cx)))
            .size_full()
            .bg(palette::editor())
            .text_color(palette::text())
            .child(self.render_title_bar(cx))
            .child(
                div().flex_1().min_h_0().child(
                    h_resizable("main")
                        .child(
                            resizable_panel()
                                .size(px(320.0))
                                .size_range(px(260.0)..px(560.0))
                                .child(
                                    div()
                                        .size_full()
                                        .bg(palette::panel())
                                        .border_r_1()
                                        .border_color(palette::border_variant())
                                        .child(self.render_side_bar(cx)),
                                ),
                        )
                        .child(
                            resizable_panel().child(
                                v_flex()
                                    .size_full()
                                    .child(self.render_tab_bar(cx))
                                    .child(self.render_toolbar(cx))
                                    .child(div().flex_1().min_h_0().child(self.render_plot(cx))),
                            ),
                        ),
                ),
            )
            .child(self.render_status_bar())
    }
}

fn panel_header(title: &'static str) -> impl IntoElement {
    h_flex()
        .h(px(36.0))
        .flex_shrink_0()
        .px_3()
        .border_b_1()
        .border_color(palette::border_variant())
        .text_xs()
        .font_medium()
        .text_color(palette::text_muted())
        .child(title.to_uppercase())
}

/// Peak level, IR start and peak, shown in a measurement's menu. The times include the device
/// latency.
fn details(response: &Response) -> Vec<SharedString> {
    let sample_rate = response.impulse_response.sample_rate();
    let ms = |samples: usize| samples as f64 / sample_rate * 1000.0;
    let peak = response.start + measure::peak_index(&response.impulse_response);
    vec![
        format!("Recording peak: {:.1} dBFS", response.peak_db).into(),
        format!("IR start: {:.2} ms (with latency)", ms(response.start)).into(),
        format!("IR peak: {:.2} ms (with latency)", ms(peak)).into(),
    ]
}
