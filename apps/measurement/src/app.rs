//! The window: measurements on the left, the plots in tabs on the right.

use std::rc::Rc;
use std::time::Duration;

use gpui_kit::base::actions::{SelectDown, SelectUp};
use gpui_kit::component::button::{Button, ButtonVariant, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::color_picker::{ColorPicker, ColorPickerEvent, ColorPickerState};
use gpui_kit::component::menu::{ContextMenuExt as _, PopupMenuItem};
use gpui_kit::component::progress::Progress;
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::select::{Select, SelectEvent, SelectItem as _, SelectState};
use gpui_kit::component::slider::{Slider, SliderEvent, SliderScale, SliderState};
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Icon, IconName, IndexPath, Sizable as _, StyledExt as _,
    TitleBar, WindowExt as _, h_flex, h_resizable, resizable_panel, v_flex, v_resizable,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::curves::{Curves, impulse_points};
use crate::measure::{self, Job, Outcome, Response};
use crate::panes::{Panes, Side, Target};
use crate::plot::{FrequencyChanged, Plot, PlotKind, Series, Trace};
use crate::setup::{Setup, SetupEvent, next_name};
use crate::theme::{self, ShownThemes, ThemeList, palette, trace_color};

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

#[derive(Clone, Copy, PartialEq, Debug)]
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

/// A tab being dragged.
struct DraggedTab(Tab);

/// What follows the mouse while a tab is dragged.
struct TabGhost(&'static str);

impl Render for TabGhost {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .px_4()
            .py_1p5()
            .text_sm()
            .bg(palette::tab_active(cx))
            .text_color(palette::text(cx))
            .border_1()
            .border_color(palette::border(cx))
            .rounded_sm()
            .shadow_md()
            .opacity(0.9)
            .child(self.0)
    }
}

/// Where a tab dragged over the plot of a pane would go.
#[derive(Clone, Copy)]
struct ContentDrop {
    pane: usize,
    target: Target<Tab>,
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
    panes: Panes<Tab>,
    /// Where a tab dragged over a plot would go, to show it there.
    content_drop: Option<ContentDrop>,
    /// Counts the splits, so each starts with panes of equal size.
    splits: u32,
    /// The measurements under the mouse in the list, highlighted in the plots.
    highlight: Rc<[u64]>,
    smoothing_enabled: bool,
    /// 1/N octave.
    smoothing: f64,
    smoothing_slider: Entity<SliderState>,
    theme_select: Entity<SelectState<ThemeList>>,
    shown_themes: ShownThemes,
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
        let themes = theme::choices(cx);
        let current = themes
            .iter()
            .position(|choice| *choice.value() == theme::current(cx));
        let (theme_list, shown_themes) = ThemeList::new(themes);
        let theme_select = cx.new(|cx| {
            SelectState::new(theme_list, current.map(IndexPath::new), window, cx).searchable(true)
        });
        let magnitude = cx.new(|_| Plot::new(PlotKind::Magnitude));
        let phase = cx.new(|_| Plot::new(PlotKind::Phase));
        let subscriptions =
            vec![
                // magnitude and phase always show the same frequencies
                cx.subscribe(&magnitude, |this, _, FrequencyChanged(x), cx| {
                    this.phase
                        .update(cx, |phase, cx| phase.link_frequency(*x, cx))
                }),
                cx.subscribe(&phase, |this, _, FrequencyChanged(x), cx| {
                    this.magnitude
                        .update(cx, |magnitude, cx| magnitude.link_frequency(*x, cx))
                }),
                cx.subscribe(&theme_select, |_, _, event: &SelectEvent<_>, cx| {
                    if let SelectEvent::Confirm(Some(name)) = event {
                        theme::select(name, cx);
                    }
                }),
                // closed without choosing: back to the theme from before the preview
                cx.subscribe(&theme_select, |_, _, _: &DismissEvent, cx| {
                    theme::end_preview(cx)
                }),
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
            panes: Panes::new(Tab::ALL.to_vec()),
            content_drop: None,
            splits: 0,
            highlight: Rc::new([]),
            smoothing_enabled: true,
            smoothing,
            smoothing_slider,
            theme_select,
            shown_themes,
            setup,
            magnitude,
            phase,
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

    /// Previews the theme under the theme select's keyboard cursor. Meant for an arrow key on
    /// its way to the select, so it waits until the select has moved the cursor.
    fn theme_cursor_preview(&self) -> impl Fn(&mut App) + 'static {
        let select = self.theme_select.downgrade();
        let shown = self.shown_themes.clone();
        move |cx| {
            let select = select.clone();
            let shown = shown.clone();
            cx.defer(move |cx| {
                let Ok(row) = select.read_with(cx, |select, cx| select.selected_index(cx)) else {
                    return;
                };
                shown.preview(row, cx);
            });
        }
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
                let taken = self.group_names();
                self.setup.update(cx, |setup, cx| {
                    setup.count_up(taken.iter().map(|n| n.as_ref()), window, cx)
                });
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
            let taken = self.group_names();
            next_name("", taken.iter().map(|n| n.as_ref()))
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

    fn group_names(&self) -> Vec<SharedString> {
        self.groups.iter().map(|group| group.name.clone()).collect()
    }

    fn apply(&mut self, action: Action, window: &mut Window, cx: &mut Context<Self>) {
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
        if matches!(action, Action::DeleteGroup(_) | Action::Delete(_)) {
            // the deleted numbers are free again
            let taken = self.group_names();
            self.setup.update(cx, |setup, cx| {
                setup.renumber(taken.iter().map(|n| n.as_ref()), window, cx)
            });
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
                .on_ok(move |_, window, cx| {
                    app.update(cx, |this, cx| this.apply(action, window, cx));
                    true
                })
        });
    }

    /// Highlights the measurements `ids` in the plots while the mouse is over their row,
    /// and stops when it leaves, unless it went on to another row meanwhile.
    fn hover_measurements(&mut self, ids: Rc<[u64]>, hovered: bool, cx: &mut Context<Self>) {
        let highlight: Rc<[u64]> = if hovered {
            ids
        } else if self.highlight == ids {
            Rc::new([])
        } else {
            return;
        };
        self.highlight = highlight.clone();
        for plot in [&self.magnitude, &self.phase, &self.impulse] {
            plot.update(cx, |plot, cx| plot.set_highlight(highlight.clone(), cx));
        }
    }

    /// Shows `tab` in its pane.
    fn set_tab(&mut self, tab: Tab, cx: &mut Context<Self>) {
        self.panes.activate(tab);
        self.refresh_plots(cx);
    }

    /// Moves a dragged tab to `target`.
    fn drop_tab(&mut self, tab: Tab, target: Target<Tab>, cx: &mut Context<Self>) {
        self.content_drop = None;
        if self.panes.drop_tab(tab, target) {
            if matches!(target, Target::Split { .. }) {
                self.splits += 1;
            }
            self.refresh_plots(cx);
        }
        cx.notify();
    }

    /// Follows a tab dragged over the plot of the `pane`th pane, to show where it would go.
    fn drag_over_content(
        &mut self,
        pane: usize,
        event: &DragMoveEvent<DraggedTab>,
        cx: &mut Context<Self>,
    ) {
        let bounds = event.bounds;
        let position = event.event.position;
        let drop = if bounds.contains(&position) {
            let offset = position - bounds.origin;
            let side = Side::near(offset.x / bounds.size.width, offset.y / bounds.size.height);
            let tab = event.drag(cx).0;
            self.panes
                .content_target(tab, pane, side)
                .map(|target| ContentDrop { pane, target })
        } else if self.content_drop.is_some_and(|drop| drop.pane == pane) {
            None
        } else {
            // over another pane, which keeps track of itself
            return;
        };
        let changed = match (self.content_drop, drop) {
            (Some(a), Some(b)) => a.pane != b.pane || a.target != b.target,
            (a, b) => a.is_some() != b.is_some(),
        };
        if changed {
            self.content_drop = drop;
            cx.notify();
        }
    }

    /// Drops a tab onto the plot of the `pane`th pane.
    fn drop_on_content(&mut self, pane: usize, tab: Tab, cx: &mut Context<Self>) {
        match self.content_drop.take() {
            Some(drop) if drop.pane == pane => self.drop_tab(tab, drop.target, cx),
            _ => cx.notify(),
        }
    }

    fn new_measurement(&mut self, cx: &mut Context<Self>) {
        self.side_bar = SideBar::Setup;
        cx.notify();
    }

    /// Hands the visible measurements to the plots on show, computing the curves for the
    /// current smoothing where needed.
    fn refresh_plots(&mut self, cx: &mut Context<Self>) {
        let tabs: Vec<Tab> = self.panes.visible().collect();
        for tab in tabs {
            self.refresh_plot(tab, cx);
        }
        cx.notify();
    }

    fn refresh_plot(&mut self, tab: Tab, cx: &mut Context<Self>) {
        let smoothing = self.smoothing();
        let frequency = tab != Tab::Impulse;
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
                    id: m.id,
                    name: m.full_name.clone(),
                    color: m.color,
                    series,
                });
            }
        }
        if !range.0.is_finite() {
            range = (20.0, 20_000.0);
        }
        self.plot(tab)
            .update(cx, |plot, cx| plot.set_traces(traces, range, cx));
    }

    fn plot(&self, tab: Tab) -> &Entity<Plot> {
        match tab {
            Tab::Magnitude => &self.magnitude,
            Tab::Phase => &self.phase,
            Tab::Impulse => &self.impulse,
        }
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
                                .text_color(palette::accent(cx)),
                        )
                        .child(div().text_sm().child("audio-lab"))
                        .child(
                            div()
                                .text_sm()
                                .text_color(palette::text_muted(cx))
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
            return self.render_progress(job, cx).into_any_element();
        }
        match self.side_bar {
            SideBar::Setup => self.setup.clone().into_any_element(),
            SideBar::Measurements => self.render_list(cx).into_any_element(),
        }
    }

    fn render_progress(&self, job: &Job, cx: &App) -> impl IntoElement {
        let remaining = job.remaining().as_secs();
        v_flex()
            .size_full()
            .child(panel_header("Measuring", cx))
            .child(
                v_flex()
                    .p_3()
                    .gap_3()
                    .child(Progress::new("measure-progress").value(job.progress() * 100.0))
                    .child(div().text_sm().text_color(palette::text_muted(cx)).child(
                        if remaining > 0 {
                            format!("About {remaining} s left. Keep it quiet.")
                        } else {
                            "Analyzing…".to_string()
                        },
                    )),
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
                    .border_color(palette::border(cx))
                    .child(
                        div()
                            .text_xs()
                            .font_medium()
                            .text_color(palette::text_muted(cx))
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
                                                .text_color(palette::text_placeholder(cx))
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
            .border_color(palette::warning(cx).opacity(0.4))
            .bg(palette::warning(cx).opacity(0.08))
            .children(self.warnings.iter().map(|warning| {
                h_flex()
                    .gap_2()
                    .items_start()
                    .child(
                        Icon::new(IconName::TriangleAlert)
                            .xsmall()
                            .text_color(palette::warning(cx)),
                    )
                    .child(
                        div()
                            .flex_1()
                            .text_xs()
                            .text_color(palette::text(cx))
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
        let ids: Rc<[u64]> = group.measurements.iter().map(|m| m.id).collect();
        h_flex()
            .id(("group", id))
            .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
                this.hover_measurements(ids.clone(), *hovered, cx)
            }))
            .group(row_group.clone())
            .h(px(26.0))
            .pl_2()
            .pr_1()
            .gap_1p5()
            .w_full()
            .cursor_pointer()
            .hover(|style| style.bg(palette::hover(cx)))
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
                .text_color(palette::text_muted(cx)),
            )
            .child(
                Icon::new(if group.expanded {
                    IconName::FolderOpen
                } else {
                    IconName::Folder
                })
                .small()
                .text_color(palette::text_muted(cx)),
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
                        palette::text(cx)
                    } else {
                        palette::text_placeholder(cx)
                    })
                    .child(group.name.clone()),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(palette::text_placeholder(cx))
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
                div()
                    .relative()
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
                    // some shown: a dash over the box; clicking it still shows all
                    .when(some && !all, |this| this.child(partly_checked_mark(cx))),
            )
    }

    fn render_measurement_row(&self, m: &Measurement, cx: &mut Context<Self>) -> impl IntoElement {
        let id = m.id;
        let details = details(&m.response);
        let app = cx.entity();
        let ids: Rc<[u64]> = Rc::new([id]);
        h_flex()
            .id(("measurement", id))
            .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
                this.hover_measurements(ids.clone(), *hovered, cx)
            }))
            .h(px(26.0))
            .pl(px(34.0))
            .pr_1()
            .gap_2()
            .w_full()
            .cursor_pointer()
            .hover(|style| style.bg(palette::hover(cx)))
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
                        palette::text(cx)
                    } else {
                        palette::text_placeholder(cx)
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
                        PopupMenuItem::new("Show only this").on_click(move |_, window, cx| {
                            solo.update(cx, |this, cx| this.apply(Action::Solo(id), window, cx));
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

    /// The tabs of the `ix`th pane. They can be dragged to reorder them, to the other pane,
    /// or onto a plot to split the view.
    fn render_tab_bar(&self, ix: usize, cx: &mut Context<Self>) -> impl IntoElement {
        let pane = &self.panes.panes[ix];
        h_flex()
            .h(px(34.0))
            .flex_shrink_0()
            .bg(palette::tab_bar(cx))
            .border_b_1()
            .border_color(palette::border(cx))
            .children(pane.tabs.iter().map(|&tab| {
                let active = tab == pane.active;
                h_flex()
                    .id(tab.title())
                    .h_full()
                    .px_4()
                    .border_r_1()
                    .border_color(palette::border(cx))
                    .text_sm()
                    .cursor_pointer()
                    .map(|this| {
                        if active {
                            // the active tab is part of the plot below it
                            this.bg(palette::tab_active(cx))
                                .text_color(palette::text(cx))
                                .mb(px(-1.0))
                                .h(px(35.0))
                        } else {
                            this.text_color(palette::text_muted(cx))
                                .hover(|style| style.bg(palette::hover(cx)))
                        }
                    })
                    .on_click(cx.listener(move |this, _, _, cx| this.set_tab(tab, cx)))
                    .on_drag(DraggedTab(tab), |dragged, _, _, cx| {
                        let title = dragged.0.title();
                        cx.new(|_| TabGhost(title))
                    })
                    .drag_over::<DraggedTab>(|style, _, _, cx| style.bg(palette::drop_target(cx)))
                    .on_drop(cx.listener(move |this, dragged: &DraggedTab, _, cx| {
                        let target = Target::Tabs {
                            pane: ix,
                            before: Some(tab),
                        };
                        this.drop_tab(dragged.0, target, cx)
                    }))
                    .child(tab.title())
            }))
            // the free space after the tabs: drops there go last
            .child(
                div()
                    .id(("tab-bar-end", ix))
                    .flex_1()
                    .h_full()
                    .drag_over::<DraggedTab>(|style, _, _, cx| style.bg(palette::drop_target(cx)))
                    .on_drop(cx.listener(move |this, dragged: &DraggedTab, _, cx| {
                        let target = Target::Tabs {
                            pane: ix,
                            before: None,
                        };
                        this.drop_tab(dragged.0, target, cx)
                    })),
            )
    }

    /// The panes: one, or two split side by side or one above the other.
    fn render_panes(&self, cx: &mut Context<Self>) -> AnyElement {
        if !self.panes.is_split() {
            return self.render_pane(0, cx).into_any_element();
        }
        let id = SharedString::from(format!("panes-{}", self.splits));
        let group = if self.panes.stacked {
            v_resizable(id)
        } else {
            h_resizable(id)
        };
        (0..self.panes.panes.len())
            .fold(group, |group, ix| {
                group.child(resizable_panel().child(self.render_pane(ix, cx)))
            })
            .into_any_element()
    }

    /// The `ix`th pane: its tabs and the plot of the active one.
    fn render_pane(&self, ix: usize, cx: &mut Context<Self>) -> impl IntoElement {
        let tab = self.panes.panes[ix].active;
        let drop = self
            .content_drop
            .filter(|drop| drop.pane == ix && cx.has_active_drag());
        v_flex()
            .size_full()
            .child(self.render_tab_bar(ix, cx))
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .on_drag_move(
                        cx.listener(move |this, event, _, cx| {
                            this.drag_over_content(ix, event, cx)
                        }),
                    )
                    .on_drop(cx.listener(move |this, dragged: &DraggedTab, _, cx| {
                        this.drop_on_content(ix, dragged.0, cx)
                    }))
                    .child(self.render_plot(tab, ix, cx))
                    .when_some(drop, |this, drop| this.child(drop_overlay(drop.target, cx))),
            )
    }

    fn render_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let frequency = self.panes.visible().any(|tab| tab != Tab::Impulse);
        h_flex()
            .h(px(36.0))
            .flex_shrink_0()
            .px_3()
            .gap_3()
            .bg(palette::editor(cx))
            .border_b_1()
            .border_color(palette::border(cx))
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
                            palette::text_muted(cx)
                        } else {
                            palette::text_placeholder(cx)
                        })
                        .child(format!("1/{:.0} octave", self.smoothing)),
                )
            })
            .child(div().flex_1())
            .child(
                div()
                    .text_xs()
                    .text_color(palette::text_placeholder(cx))
                    .child(
                        "Scroll to move · Ctrl+scroll to zoom (+Shift: horizontal, +Alt: vertical) · \
                         double-click to fit",
                    ),
            )
    }

    /// The plot of `tab`, shown in the `pane`th pane.
    fn render_plot(&self, tab: Tab, pane: usize, cx: &mut Context<Self>) -> AnyElement {
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
                .bg(palette::editor(cx))
                .child(
                    Icon::new(gpui_kit::assets::IconName::AudioWaveform)
                        .size(px(48.0))
                        .text_color(palette::border(cx)),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(palette::text_placeholder(cx))
                        .child(message),
                )
                .when(self.groups.is_empty() && self.job.is_none(), |this| {
                    this.child(
                        Button::new(("empty-new", pane))
                            .small()
                            .icon(IconName::Plus)
                            .label("New measurement")
                            .on_click(cx.listener(|this, _, _, cx| this.new_measurement(cx))),
                    )
                })
                .into_any_element();
        }
        self.plot(tab).clone().into_any_element()
    }

    fn render_status_bar(&self, cx: &App) -> impl IntoElement {
        let total = self.num_measurements();
        let shown = self.num_visible();
        h_flex()
            .h(px(26.0))
            .flex_shrink_0()
            .px_3()
            .gap_3()
            .justify_between()
            .bg(palette::status_bar(cx))
            .border_t_1()
            .border_color(palette::border(cx))
            .text_xs()
            .text_color(palette::text_muted(cx))
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
            .child(
                h_flex()
                    .gap_3()
                    .when(!self.warnings.is_empty(), |this| {
                        this.child(
                            h_flex()
                                .gap_1()
                                .text_color(palette::warning(cx))
                                .child(Icon::new(IconName::TriangleAlert).xsmall())
                                .child(format!(
                                    "{} warning{}",
                                    self.warnings.len(),
                                    if self.warnings.len() == 1 { "" } else { "s" }
                                )),
                        )
                    })
                    .child(
                        h_flex()
                            .gap_1()
                            .child(Icon::new(IconName::Palette).xsmall())
                            .child(
                                div()
                                    .w(px(170.0))
                                    // The arrow keys move the select's cursor; preview the
                                    // theme it lands on, once the list has moved it.
                                    .capture_action({
                                        let preview = self.theme_cursor_preview();
                                        move |_: &SelectUp, _, cx| preview(cx)
                                    })
                                    .capture_action({
                                        let preview = self.theme_cursor_preview();
                                        move |_: &SelectDown, _, cx| preview(cx)
                                    })
                                    .child(
                                        Select::new(&self.theme_select)
                                            .xsmall()
                                            .appearance(false)
                                            .search_placeholder("Search themes")
                                            .menu_width(px(240.0))
                                            .menu_max_h(px(420.0)),
                                    ),
                            ),
                    ),
            )
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
            .bg(palette::editor(cx))
            .text_color(palette::text(cx))
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
                                        .bg(palette::panel(cx))
                                        .border_r_1()
                                        .border_color(palette::border(cx))
                                        .child(self.render_side_bar(cx)),
                                ),
                        )
                        .child(
                            resizable_panel().child(
                                v_flex()
                                    .size_full()
                                    .child(self.render_toolbar(cx))
                                    .child(div().flex_1().min_h_0().child(self.render_panes(cx))),
                            ),
                        ),
                ),
            )
            .child(self.render_status_bar(cx))
    }
}

/// Covers a small checkbox with a filled box and a dash, for "some of them": GPUI Kit's
/// checkbox has no state for that. It catches no mouse, so the checkbox below still works.
fn partly_checked_mark(cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .absolute()
        .top_0()
        .left_0()
        // as big as the box of a small checkbox
        .size(rems(0.875))
        .flex()
        .items_center()
        .justify_center()
        .rounded(theme.radius.min(px(4.0)))
        .border_1()
        .border_color(theme.primary)
        .bg(theme.primary)
        .child(
            Icon::new(gpui_kit::assets::IconName::Minus)
                .size(rems(0.625))
                .text_color(theme.primary_foreground),
        )
}

/// Marks where a tab dragged over a plot would go: the half of the pane a split would take,
/// or the whole pane.
fn drop_overlay(target: Target<Tab>, cx: &App) -> impl IntoElement {
    let half = relative(0.5);
    div()
        .absolute()
        .top_0()
        .left_0()
        .size_full()
        .map(|this| match target {
            Target::Split { side: Side::Left } => this.w(half),
            Target::Split { side: Side::Right } => this.left(half).w(half),
            Target::Split { side: Side::Top } => this.h(half),
            Target::Split { side: Side::Bottom } => this.top(half).h(half),
            Target::Tabs { .. } => this,
        })
        .bg(palette::drop_target(cx))
        .border_2()
        .border_color(palette::accent(cx))
}

fn panel_header(title: &'static str, cx: &App) -> impl IntoElement {
    h_flex()
        .h(px(36.0))
        .flex_shrink_0()
        .px_3()
        .border_b_1()
        .border_color(palette::border(cx))
        .text_xs()
        .font_medium()
        .text_color(palette::text_muted(cx))
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
