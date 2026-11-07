//! The window: measurements on the left, the plots in tabs on the right.

use std::rc::Rc;
use std::time::Duration;

use chrono::{DateTime, Local};
use gpui_kit::base::actions::{SelectDown, SelectUp};
use gpui_kit::component::button::{Button, ButtonVariant, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::color_picker::{ColorPicker, ColorPickerEvent, ColorPickerState};
use gpui_kit::component::input::{InputEvent, Textarea, TextareaState};
use gpui_kit::component::menu::{ContextMenuExt as _, PopupMenuItem};
use gpui_kit::component::popover::Popover;
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
use crate::live::{self, Analysis, Live, LiveRequest, Status};
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

/// A folder of measurements: those taken together, one per speaker and mic, or whatever the
/// user moved into it. Only a name and notes: when and how its measurements were measured is
/// theirs, so they can move between folders.
struct Group {
    id: u64,
    name: SharedString,
    notes: Notes,
    expanded: bool,
    measurements: Vec<Measurement>,
}

/// What the user writes down about a measurement or a folder.
struct Notes {
    field: Entity<TextareaState>,
    _subscription: Subscription,
}

impl Notes {
    fn new(text: &str, window: &mut Window, cx: &mut Context<MeasurementApp>) -> Self {
        let field = cx.new(|cx| {
            TextareaState::new(window, cx)
                .placeholder("Add notes…")
                .default_value(text.to_string())
        });
        // the list shows which have notes
        let subscription = cx.subscribe(&field, |_, _, event: &InputEvent, cx| {
            if matches!(event, InputEvent::Change) {
                cx.notify();
            }
        });
        Self {
            field,
            _subscription: subscription,
        }
    }

    fn text(&self, cx: &App) -> SharedString {
        self.field.read(cx).value()
    }
}

struct Measurement {
    id: u64,
    /// `speaker → mic`.
    name: SharedString,
    /// When it was measured: the start of the sweeps or the snapshot.
    measured: DateTime<Local>,
    /// How it was measured, a line each.
    about: Rc<[SharedString]>,
    /// The frequency range it was measured in, in Hz: outside of it there is nothing to show.
    range: (f64, f64),
    notes: Notes,
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

impl Measurement {
    /// Names it after its response, in the folder `group`, and takes the impulse response to
    /// draw from it: after it was made, and whenever a live response changes.
    fn follow_response(&mut self, group: &str) {
        let short = format!("{} → {}", self.response.speaker, self.response.mic);
        self.full_name = format!("{group} · {short}").into();
        self.name = short.into();
        self.impulse = Rc::new(impulse_points(
            &self.response.impulse_response,
            self.response.start,
        ));
    }
}

/// Something to do to the list.
#[derive(Clone, Copy)]
enum Action {
    DeleteGroup(u64),
    Delete(u64),
    /// Show only this measurement.
    Solo(u64),
    /// Move the measurement `id` into the folder `to`, or a new one.
    Move {
        id: u64,
        to: Option<u64>,
    },
}

/// The id of the live folder; the others count from 1.
const LIVE_GROUP: u64 = 0;
/// How often a running live measurement is analyzed.
const LIVE_UPDATE: Duration = Duration::from_millis(100);

/// A live measurement that runs, and its folder at the top of the list.
/// When, how and within which frequency range something was measured.
#[derive(Clone)]
struct Measured {
    at: DateTime<Local>,
    about: Rc<[SharedString]>,
    /// In Hz.
    range: (f64, f64),
}

struct LiveRun {
    /// Tells this run's analyses from those of an earlier one that still finish.
    run: u64,
    live: Live,
    /// Since when, how and in which range it measures.
    how: Measured,
    /// One measurement per mic, updated with every analysis.
    group: Group,
    level: Entity<SliderState>,
    /// An analysis runs in the background; the next waits for it.
    analyzing: bool,
    status: Status,
    /// Why the last analysis failed.
    error: Option<String>,
    _ticker: Task<()>,
    _subscription: Subscription,
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
    live: Option<LiveRun>,
    /// Counts the live runs.
    live_runs: u64,
    /// Below this coherence, live measurements and their snapshots aren't drawn.
    min_coherence: f64,
    coherence_slider: Entity<SliderState>,
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
        let min_coherence = 0.5;
        let coherence_slider = cx.new(|_| {
            SliderState::new()
                .min(0.0)
                .max(1.0)
                .step(0.05)
                .default_value(min_coherence as f32)
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
        let subscriptions = vec![
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
            cx.subscribe_in(
                &setup,
                window,
                |this, _, event: &SetupEvent, window, cx| match event {
                    SetupEvent::Start(request) => this.start(request.clone(), window, cx),
                    SetupEvent::StartLive(request) => this.start_live(request.clone(), window, cx),
                    SetupEvent::Close => {
                        this.side_bar = SideBar::Measurements;
                        cx.notify();
                    }
                },
            ),
            cx.subscribe(&coherence_slider, |this, _, event: &SliderEvent, cx| {
                let (SliderEvent::Change(value) | SliderEvent::Release(value)) = event;
                let value = f64::from(value.start());
                if value != this.min_coherence {
                    this.min_coherence = value;
                    this.refresh_plots(cx);
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
            live: None,
            live_runs: 0,
            min_coherence,
            coherence_slider,
            _ticker: None,
            warnings: Vec::new(),
            _subscriptions: subscriptions,
        }
    }

    fn smoothing(&self) -> Option<f64> {
        self.smoothing_enabled.then_some(self.smoothing)
    }

    /// The live folder while live runs, then the others.
    fn all_groups(&self) -> impl Iterator<Item = &Group> {
        self.live.iter().map(|run| &run.group).chain(&self.groups)
    }

    fn all_groups_mut(&mut self) -> impl Iterator<Item = &mut Group> {
        self.live
            .iter_mut()
            .map(|run| &mut run.group)
            .chain(&mut self.groups)
    }

    fn num_measurements(&self) -> usize {
        self.groups.iter().map(|g| g.measurements.len()).sum()
    }

    /// Whether hidden measurements are shown because the mouse is over them in the list.
    fn previewing(&self) -> bool {
        self.all_groups()
            .flat_map(|g| &g.measurements)
            .any(|m| !m.visible && self.highlight.contains(&m.id))
    }

    fn num_visible(&self) -> usize {
        self.all_groups()
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
        // the sweep needs the device
        self.stop_live(cx);
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
                let name = self.setup.read(cx).name();
                let name = if name.is_empty() {
                    let taken = self.group_names();
                    next_name("", taken.iter().map(|n| n.as_ref()))
                } else {
                    name
                };
                let range = (job.request.range.start, job.request.range.end);
                let about = job.request.describe();
                // written while setting it up and while it ran
                let notes = self
                    .setup
                    .update(cx, |setup, cx| setup.take_notes(window, cx));
                self.add_group(
                    name,
                    range,
                    about,
                    job.started_at,
                    &notes,
                    outcome.responses,
                    window,
                    cx,
                );
                self.warnings = outcome.warnings;
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

    /// Adds a folder `name` with the `responses`, measured at `measured` within `range` (Hz)
    /// as `about` tells, with `notes`.
    #[allow(clippy::too_many_arguments)]
    fn add_group(
        &mut self,
        name: String,
        range: (f64, f64),
        about: Vec<String>,
        measured: DateTime<Local>,
        notes: &str,
        responses: Vec<Response>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let about: Rc<[SharedString]> = about.into_iter().map(Into::into).collect();
        let how = Measured {
            at: measured,
            about,
            range,
        };
        let measurements = responses
            .into_iter()
            .map(|response| self.new_measurement_of(response, &name, &how, window, cx))
            .collect();
        self.groups.push(Group {
            id: self.next_group_id,
            name: name.into(),
            notes: Notes::new(notes, window, cx),
            expanded: true,
            measurements,
        });
        self.next_group_id += 1;
    }

    /// A measurement of `response` in the folder `group`, measured as `how` tells, with a
    /// color of its own.
    fn new_measurement_of(
        &mut self,
        response: Response,
        group: &str,
        how: &Measured,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Measurement {
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
        let mut measurement = Measurement {
            id,
            measured: how.at,
            about: how.about.clone(),
            range: how.range,
            notes: Notes::new("", window, cx),
            full_name: SharedString::default(),
            name: SharedString::default(),
            impulse: Rc::default(),
            response,
            visible: true,
            color,
            color_picker,
            curves: None,
            _subscription: subscription,
        };
        measurement.follow_response(group);
        measurement
    }

    fn measurement_mut(&mut self, id: u64) -> Option<&mut Measurement> {
        self.all_groups_mut()
            .flat_map(|g| &mut g.measurements)
            .find(|m| m.id == id)
    }

    /// Plays noise and analyzes what the mics hear until stopped, in the live folder.
    fn start_live(&mut self, request: LiveRequest, window: &mut Window, cx: &mut Context<Self>) {
        if self.job.is_some() {
            return;
        }
        self.stop_live(cx);
        let range = request.range();
        let level_db = request.level_db;
        let about = request.describe(level_db);
        let live = match Live::start(request) {
            Ok(live) => live,
            Err(error) => {
                self.setup
                    .update(cx, |setup, cx| setup.set_error(Some(error), cx));
                return;
            }
        };
        // written while setting it up; editable in the live box while it runs
        let notes = self
            .setup
            .update(cx, |setup, cx| setup.take_notes(window, cx));
        self.live_runs += 1;
        let run = self.live_runs;
        let level = cx.new(|_| {
            SliderState::new()
                .min(-60.0)
                .max(0.0)
                .step(1.0)
                .default_value(level_db as f32)
        });
        let subscription = cx.subscribe(&level, |this, _, event: &SliderEvent, cx| {
            let (SliderEvent::Change(value) | SliderEvent::Release(value)) = event;
            if let Some(run) = &this.live {
                run.live.set_level(f64::from(value.start()));
            }
            cx.notify();
        });
        let ticker = cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(LIVE_UPDATE).await;
                if this
                    .update_in(cx, |this, window, cx| this.live_tick(window, cx))
                    .is_err()
                {
                    break;
                }
            }
        });
        self.live = Some(LiveRun {
            run,
            live,
            how: Measured {
                at: Local::now(),
                about: about.into_iter().map(Into::into).collect(),
                range,
            },
            group: Group {
                id: LIVE_GROUP,
                name: "Live".into(),
                notes: Notes::new(&notes, window, cx),
                expanded: true,
                measurements: Vec::new(),
            },
            level,
            analyzing: false,
            status: Status::default(),
            error: None,
            _ticker: ticker,
            _subscription: subscription,
        });
        self.side_bar = SideBar::Measurements;
        cx.notify();
    }

    fn stop_live(&mut self, cx: &mut Context<Self>) {
        let Some(run) = self.live.take() else { return };
        run.live.stop();
        for plot in [&self.magnitude, &self.phase, &self.impulse] {
            plot.update(cx, |plot, _| plot.set_steady(false));
        }
        self.refresh_plots(cx);
    }

    /// Starts analyzing the latest recording, unless the last analysis still runs.
    fn live_tick(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let smoothing = self.smoothing();
        let Some(run) = &mut self.live else { return };
        run.status = run.live.status();
        if let Some(error) = run.status.error.take() {
            self.warnings = vec![format!("Live stopped: {error}")];
            self.stop_live(cx);
            return;
        }
        if run.analyzing {
            return;
        }
        let Some(recording) = run.live.take(smoothing) else {
            return;
        };
        run.analyzing = true;
        let id = run.run;
        cx.spawn_in(window, async move |this, cx| {
            let analysis = cx
                .background_spawn(async move { live::analyze(recording) })
                .await;
            this.update_in(cx, |this, window, cx| {
                this.live_analyzed(id, analysis, window, cx)
            })
            .ok();
        })
        .detach();
    }

    /// Shows a live analysis in the live folder.
    fn live_analyzed(
        &mut self,
        id: u64,
        analysis: Result<Analysis, String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(run) = self.live.as_mut().filter(|run| run.run == id) else {
            return;
        };
        run.analyzing = false;
        let analysis = match analysis {
            Ok(analysis) => analysis,
            Err(error) => {
                run.error = Some(error);
                cx.notify();
                return;
            }
        };
        run.error = None;
        // an analysis that started before "Find delays" still has the old ones
        if run.live.delays_ms.is_none() && analysis.found_delays {
            run.live.delays_ms = Some(analysis.delays_ms);
        }
        if run.group.measurements.len() != analysis.responses.len() {
            let how = run.how.clone();
            let measurements = analysis
                .responses
                .into_iter()
                .map(|response| self.new_measurement_of(response, "Live", &how, window, cx))
                .collect();
            if let Some(run) = &mut self.live {
                run.group.measurements = measurements;
            }
            // from now on the curves change many times a second: the plots fit to them once,
            // then stay where they are
            for plot in [&self.magnitude, &self.phase, &self.impulse] {
                plot.update(cx, |plot, _| plot.set_steady(true));
            }
        } else {
            for ((m, response), curves) in run
                .group
                .measurements
                .iter_mut()
                .zip(analysis.responses)
                .zip(analysis.curves)
            {
                m.response = response;
                m.follow_response("Live");
                // made with the smoothing of the time; if that changed meanwhile, they're
                // made again
                m.curves = Some(Rc::new(curves));
            }
        }
        self.refresh_plots(cx);
    }

    /// Keeps the live curves as they are now, in a folder of their own.
    fn snapshot(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(run) = &self.live else { return };
        let responses: Vec<Response> = run
            .group
            .measurements
            .iter()
            .map(|m| m.response.clone())
            .collect();
        if responses.is_empty() {
            return;
        }
        let range = run.how.range;
        let level_db = f64::from(run.level.read(cx).value().start());
        let about = run.live.request.describe(level_db);
        // what was written about the live curves goes with them
        let folder_notes = run.group.notes.text(cx);
        let notes: Vec<SharedString> = run
            .group
            .measurements
            .iter()
            .map(|m| m.notes.text(cx))
            .collect();
        let taken = self.group_names();
        let name = next_name("Snapshot 1", taken.iter().map(|n| n.as_ref()));
        self.add_group(
            name,
            range,
            about,
            Local::now(),
            &folder_notes,
            responses,
            window,
            cx,
        );
        if let Some(group) = self.groups.last() {
            for (m, text) in group.measurements.iter().zip(notes) {
                m.notes
                    .field
                    .update(cx, |field, cx| field.set_value(text, window, cx));
            }
        }
        self.refresh_plots(cx);
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
                for m in self.all_groups_mut().flat_map(|g| &mut g.measurements) {
                    m.visible = m.id == id;
                }
            }
            Action::Move { id, to } => self.move_measurement(id, to, window, cx),
        }
        if matches!(
            action,
            Action::DeleteGroup(_) | Action::Delete(_) | Action::Move { .. }
        ) {
            // the deleted numbers are free again
            let taken = self.group_names();
            self.setup.update(cx, |setup, cx| {
                setup.renumber(taken.iter().map(|n| n.as_ref()), window, cx)
            });
        }
        self.refresh_plots(cx);
    }

    /// Moves the measurement `id` into the folder `to`, or into a new one. It keeps when and
    /// how it was measured. A folder it leaves empty is removed.
    fn move_measurement(
        &mut self,
        id: u64,
        to: Option<u64>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(from) = self
            .groups
            .iter()
            .position(|group| group.measurements.iter().any(|m| m.id == id))
        else {
            return;
        };
        if to == Some(self.groups[from].id) {
            return;
        }
        let dest = match to {
            Some(to) => match self.groups.iter().position(|group| group.id == to) {
                Some(dest) => dest,
                None => return,
            },
            None => {
                let taken = self.group_names();
                let name = next_name("Folder 1", taken.iter().map(|n| n.as_ref()));
                self.groups.push(Group {
                    id: self.next_group_id,
                    name: name.into(),
                    notes: Notes::new("", window, cx),
                    expanded: true,
                    measurements: Vec::new(),
                });
                self.next_group_id += 1;
                self.groups.len() - 1
            }
        };
        let source = &mut self.groups[from].measurements;
        let at = source.iter().position(|m| m.id == id).expect("found above");
        let mut m = source.remove(at);
        let dest = &mut self.groups[dest];
        m.follow_response(&dest.name);
        dest.measurements.push(m);
        dest.expanded = true;
        self.groups.retain(|group| !group.measurements.is_empty());
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
            Action::Solo(_) | Action::Move { .. } => return,
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
        // hidden ones among them are shown while the mouse is over them
        let hidden = self
            .all_groups()
            .flat_map(|g| &g.measurements)
            .any(|m| !m.visible);
        if hidden {
            self.refresh_plots(cx);
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
        let min_coherence = self.min_coherence;
        let frequency = tab != Tab::Impulse;
        let mut range = (f64::INFINITY, 0.0f64);
        let mut traces = Vec::new();
        // the live curves last, so on top
        let highlight = self.highlight.clone();
        let groups: Vec<&mut Group> = self.all_groups_mut().collect();
        for group in groups.into_iter().rev() {
            // hidden ones under the mouse in the list are shown muted, without fitting to them
            let shown = |m: &&mut Measurement| m.visible || highlight.contains(&m.id);
            for m in group.measurements.iter_mut().filter(shown) {
                let preview = !m.visible;
                if !preview {
                    range = (range.0.min(m.range.0), range.1.max(m.range.1));
                }
                let series = if frequency {
                    if m.curves.as_ref().is_none_or(|c| c.smoothing != smoothing) {
                        m.curves = Some(Rc::new(Curves::new(
                            &m.response.frequency_response,
                            m.response.coherence.as_ref(),
                            m.range,
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
                    min_coherence,
                    preview,
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

    /// While a sweep runs: how far along it is, and the notes, still to edit.
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
                    ))
                    .child(notes_field(self.setup.read(cx).notes(), cx)),
            )
    }

    fn render_list(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut list = v_flex().w_full().py_1();
        // live on top, then the newest
        let live = self.live.iter().map(|run| &run.group);
        for group in live.chain(self.groups.iter().rev()) {
            list = list.child(self.render_group_row(group, cx));
            if group.expanded {
                for m in &group.measurements {
                    list = list.child(self.render_measurement_row(m, group.id == LIVE_GROUP, cx));
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
            .when_some(self.live.as_ref(), |this, run| {
                this.child(self.render_live(run, cx))
            })
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
                            if self.groups.is_empty() && self.live.is_none() {
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

    /// The controls of a running live measurement: which speakers play, the level, the
    /// levels the mics hear, and a snapshot.
    fn render_live(&self, run: &LiveRun, cx: &mut Context<Self>) -> impl IntoElement {
        let live = &run.live;
        let level = run.level.read(cx).value().start();
        let delays = live.delays_ms.as_ref();
        v_flex()
            .m_2()
            .p_2()
            .gap_2()
            .rounded(px(4.0))
            .border_1()
            .border_color(palette::accent(cx).opacity(0.4))
            .bg(palette::accent(cx).opacity(0.06))
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Icon::new(gpui_kit::assets::IconName::Radio)
                            .small()
                            .text_color(palette::accent(cx)),
                    )
                    .child(div().flex_1().text_sm().font_medium().child("Live"))
                    .child(
                        Button::new("live-stop")
                            .ghost()
                            .xsmall()
                            .icon(IconName::Square)
                            .label("Stop")
                            .on_click(cx.listener(|this, _, _, cx| this.stop_live(cx))),
                    ),
            )
            // which speakers play
            .child(
                v_flex()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .text_color(palette::text_muted(cx))
                            .child("Plays on"),
                    )
                    .children(live.request.speakers.iter().enumerate().map(
                        |(index, (_, name))| {
                            Checkbox::new(("live-speaker", index))
                                .small()
                                .label(name.clone())
                                .checked(live.is_playing(index))
                                .on_change(cx.listener(move |this, checked: &bool, _, cx| {
                                    if let Some(run) = &mut this.live {
                                        run.live.set_playing(index, *checked);
                                    }
                                    cx.notify();
                                }))
                        },
                    )),
            )
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(palette::text_muted(cx))
                            .child("Level"),
                    )
                    .child(div().flex_1().child(Slider::new(&run.level)))
                    .child(
                        div()
                            .w(px(64.0))
                            .text_xs()
                            .text_color(palette::text_muted(cx))
                            .child(format!("{level:.0} dBFS")),
                    ),
            )
            // what the mics hear
            .children(run.group.measurements.iter().enumerate().map(|(index, m)| {
                let peak = m.response.peak_db;
                let clipping = peak >= -0.1;
                let delay = delays.and_then(|delays| delays.get(index));
                h_flex()
                    .gap_2()
                    .text_xs()
                    .text_color(if clipping {
                        palette::error(cx)
                    } else {
                        palette::text_muted(cx)
                    })
                    .child(div().flex_1().child(m.response.mic.clone()))
                    .when_some(delay, |this, delay| this.child(format!("{delay:.2} ms")))
                    .child(if clipping {
                        "clipping!".to_string()
                    } else {
                        format!("peak {peak:.1} dBFS")
                    })
            }))
            .when(
                run.status.dropouts > 0 || run.status.lost_frames > 0,
                |this| {
                    this.child(
                        div()
                            .text_xs()
                            .text_color(palette::warning(cx))
                            .child(format!(
                                "Dropouts: {} · lost frames: {}",
                                run.status.dropouts, run.status.lost_frames
                            )),
                    )
                },
            )
            .when_some(run.error.clone(), |this, error| {
                this.child(div().text_xs().text_color(palette::error(cx)).child(error))
            })
            // what is going on now; snapshots keep it
            .child(notes_field(&run.group.notes.field, cx))
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("live-snapshot")
                            .primary()
                            .xsmall()
                            .icon(gpui_kit::assets::IconName::Camera)
                            .label("Snapshot")
                            .tooltip("Keep the live curves as they are now, in a folder")
                            .disabled(run.group.measurements.is_empty())
                            .on_click(cx.listener(|this, _, window, cx| this.snapshot(window, cx))),
                    )
                    .child(
                        Button::new("live-find-delays")
                            .ghost()
                            .xsmall()
                            .icon(gpui_kit::assets::IconName::Crosshair)
                            .label("Find delays")
                            .tooltip(
                                "Take each mic's delay from the peak of its impulse response \
                                 and keep it, so changes show in the phase",
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                if let Some(run) = &mut this.live {
                                    run.live.find_delays();
                                }
                                cx.notify();
                            })),
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
                if let Some(group) = this.all_groups_mut().find(|g| g.id == id) {
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
            .child(if id == LIVE_GROUP {
                Icon::new(gpui_kit::assets::IconName::Radio)
                    .small()
                    .text_color(palette::accent(cx))
            } else {
                Icon::new(if group.expanded {
                    IconName::FolderOpen
                } else {
                    IconName::Folder
                })
                .small()
                .text_color(palette::text_muted(cx))
            })
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
            .child(info_button(
                ("group-info", id),
                Info {
                    title: group.name.clone(),
                    // live tells since when and how it measures; a folder what its
                    // measurements have in common
                    when: match &self.live {
                        Some(run) if id == LIVE_GROUP => when(run.how.at, true),
                        _ => span(group.measurements.iter().map(|m| m.measured)),
                    },
                    lines: match &self.live {
                        Some(run) if id == LIVE_GROUP => run.how.about.to_vec(),
                        _ => common_about(group.measurements.iter().map(|m| &m.about)),
                    },
                    notes: group.notes.field.clone(),
                    note_text: group.notes.text(cx),
                },
                cx,
            ))
            // the live folder goes with Stop
            .when(id != LIVE_GROUP, |this| {
                this.child(
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
            })
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
                                if let Some(group) = this.all_groups_mut().find(|g| g.id == id) {
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

    /// A measurement in the list; `live` ones change all the time and can't be deleted.
    fn render_measurement_row(
        &self,
        m: &Measurement,
        live: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let id = m.id;
        let details = details(&m.response);
        // where it can move: the other folders, newest first like the list
        let folders: Vec<(u64, SharedString)> = self
            .groups
            .iter()
            .rev()
            .filter(|group| group.measurements.iter().all(|other| other.id != id))
            .map(|group| (group.id, group.name.clone()))
            .collect();
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
            .child(info_button(
                ("measurement-info", id),
                Info {
                    title: m.full_name.clone(),
                    when: when(m.measured, live),
                    lines: m.about.iter().chain(&details).cloned().collect(),
                    notes: m.notes.field.clone(),
                    note_text: m.notes.text(cx),
                },
                cx,
            ))
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
            .context_menu(move |mut menu, window, cx| {
                let solo = app.clone();
                let delete = app.clone();
                for line in &details {
                    menu = menu.label(line.clone());
                }
                menu = menu
                    .separator()
                    .item(
                        PopupMenuItem::new("Show only this").on_click(move |_, window, cx| {
                            solo.update(cx, |this, cx| this.apply(Action::Solo(id), window, cx));
                        }),
                    );
                if live {
                    return menu;
                }
                let folders = folders.clone();
                let mover = app.clone();
                menu = menu.separator().submenu_with_icon(
                    Some(gpui_kit::assets::IconName::FolderInput.into()),
                    "Move to",
                    window,
                    cx,
                    move |mut menu, _, _| {
                        for (folder, name) in &folders {
                            let (mover, to) = (mover.clone(), *folder);
                            menu = menu.item(
                                PopupMenuItem::new(name.clone())
                                    .icon(IconName::Folder)
                                    .on_click(move |_, window, cx| {
                                        mover.update(cx, |this, cx| {
                                            this.apply(
                                                Action::Move { id, to: Some(to) },
                                                window,
                                                cx,
                                            )
                                        });
                                    }),
                            );
                        }
                        if !folders.is_empty() {
                            menu = menu.separator();
                        }
                        let mover = mover.clone();
                        menu.item(
                            PopupMenuItem::new("New folder")
                                .icon(gpui_kit::assets::IconName::FolderPlus)
                                .on_click(move |_, window, cx| {
                                    mover.update(cx, |this, cx| {
                                        this.apply(Action::Move { id, to: None }, window, cx)
                                    });
                                }),
                        )
                    },
                );
                menu.separator()
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
        // only live measurements and their snapshots know their coherence
        let coherence = frequency
            && self
                .all_groups()
                .flat_map(|g| &g.measurements)
                .any(|m| m.visible && m.response.coherence.is_some());
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
            .when(coherence, |this| {
                this.child(
                    div()
                        .id("min-coherence-label")
                        .text_sm()
                        .text_color(palette::text_muted(cx))
                        .child("Min. coherence")
                        .tooltip(|window, cx| {
                            gpui_kit::component::tooltip::Tooltip::new(
                                "Hides the live curves where the result changes from one noise \
                                 period to the next, e.g. from room noise. 0 shows everything.",
                            )
                            .build(window, cx)
                        }),
                )
                .child(div().w(px(100.0)).child(Slider::new(&self.coherence_slider)))
                .child(
                    div()
                        .text_sm()
                        .text_color(palette::text_muted(cx))
                        .child(format!("{:.2}", self.min_coherence)),
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
        if self.num_visible() == 0 && !self.previewing() {
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

/// A notes field with its label.
fn notes_field(notes: &Entity<TextareaState>, cx: &App) -> impl IntoElement {
    v_flex()
        .gap_1()
        .child(
            div()
                .text_xs()
                .font_medium()
                .text_color(palette::text_muted(cx))
                .child("Notes"),
        )
        .child(Textarea::new(notes).h(px(72.0)))
}

/// What the info card of a measurement or a folder shows.
struct Info {
    title: SharedString,
    when: String,
    /// How it was measured, or what came out.
    lines: Vec<SharedString>,
    notes: Entity<TextareaState>,
    note_text: SharedString,
}

/// When the measurements of a folder were measured, given their `dates`: the date if they
/// share it, else from the first to the last.
fn span(dates: impl Iterator<Item = DateTime<Local>> + Clone) -> String {
    let first = dates.clone().min();
    let last = dates.max();
    match (first, last) {
        (Some(first), Some(last)) if first == last => when(first, false),
        (Some(first), Some(last)) if first.date_naive() == last.date_naive() => format!(
            "{}–{}",
            first.format("%A, %-d %B %Y, %H:%M:%S"),
            last.format("%H:%M:%S")
        ),
        (Some(first), Some(last)) => format!(
            "{} – {}",
            first.format("%-d %b %Y, %H:%M"),
            last.format("%-d %b %Y, %H:%M")
        ),
        _ => String::new(),
    }
}

/// How the measurements of a folder were measured, given each one's `about`, if they all were
/// the same way.
fn common_about<'a>(mut abouts: impl Iterator<Item = &'a Rc<[SharedString]>>) -> Vec<SharedString> {
    let Some(first) = abouts.next() else {
        return Vec::new();
    };
    if abouts.all(|about| about == first) {
        first.to_vec()
    } else {
        vec!["Measured in different ways, see each measurement".into()]
    }
}

/// When something was measured: the date and time, or since when it runs live.
fn when(measured: DateTime<Local>, live: bool) -> String {
    if live {
        format!("Live since {}", measured.format("%H:%M:%S"))
    } else {
        measured.format("%A, %-d %B %Y, %H:%M:%S").to_string()
    }
}

/// A button that opens the info card: when it was measured, how, and notes to edit. Its
/// tooltip shows the notes, if there are any.
fn info_button(id: impl Into<ElementId>, info: Info, cx: &App) -> impl IntoElement {
    let id = id.into();
    let has_notes = !info.note_text.trim().is_empty();
    let tooltip: SharedString = if has_notes {
        let text = info.note_text.trim();
        match text.char_indices().nth(200) {
            Some((end, _)) => format!("{}…", &text[..end]).into(),
            None => text.to_string().into(),
        }
    } else {
        "Info and notes".into()
    };
    let trigger = Button::new(ElementId::Name(format!("{id}-button").into()))
        .ghost()
        .xsmall()
        .icon(Icon::new(IconName::Info).text_color(palette::text_placeholder(cx)))
        .tooltip(tooltip);
    // a click opens the card, it doesn't toggle the row
    div()
        .id(ElementId::Name(format!("{id}-wrapper").into()))
        .on_click(|_, _, cx| cx.stop_propagation())
        .child(Popover::new(id).trigger(trigger).content(move |_, _, cx| {
            v_flex()
                .w(px(300.0))
                .gap_2()
                .child(div().text_sm().font_medium().child(info.title.clone()))
                .child(
                    h_flex()
                        .gap_1p5()
                        .text_xs()
                        .text_color(palette::text_muted(cx))
                        .child(Icon::new(gpui_kit::assets::IconName::CalendarClock).xsmall())
                        .child(info.when.clone()),
                )
                .when(!info.lines.is_empty(), |this| {
                    this.child(
                        v_flex()
                            .gap_0p5()
                            .text_xs()
                            .text_color(palette::text_muted(cx))
                            .children(info.lines.iter().cloned()),
                    )
                })
                .child(notes_field(&info.notes, cx))
        }))
}

/// Peak level, IR start and peak, shown in a measurement's menu. The times include the device
/// latency.
fn details(response: &Response) -> Vec<SharedString> {
    let sample_rate = response.impulse_response.sample_rate();
    let ms = |samples: usize| samples as f64 / sample_rate * 1000.0;
    let peak = response.start + measure::peak_index(&response.impulse_response);
    let mut lines = Vec::new();
    if response.coherence.is_some() {
        lines.push("Measured live, with pink noise".into());
    }
    lines.push(format!("Recording peak: {:.1} dBFS", response.peak_db).into());
    if response.coherence.is_none() {
        lines.push(format!("IR start: {:.2} ms (with latency)", ms(response.start)).into());
    }
    lines.push(format!("IR peak: {:.2} ms (with latency)", ms(peak)).into());
    lines
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone as _;

    use super::*;
    // `gpui_kit::*` brings GPUI's test macro, which needs an app
    use core::prelude::v1::test;

    #[test]
    fn dates_are_spelled_out_and_live_tells_since_when() {
        let measured = Local.with_ymd_and_hms(2026, 11, 7, 14, 32, 5).unwrap();
        assert_eq!(when(measured, false), "Saturday, 7 November 2026, 14:32:05");
        assert_eq!(when(measured, true), "Live since 14:32:05");
    }

    #[test]
    fn folders_tell_what_their_measurements_share() {
        let at = |day, hour, minute| {
            Local
                .with_ymd_and_hms(2026, 11, day, hour, minute, 0)
                .unwrap()
        };
        assert_eq!(
            span([at(7, 14, 32), at(7, 14, 32)].into_iter()),
            "Saturday, 7 November 2026, 14:32:00"
        );
        assert_eq!(
            span([at(7, 15, 10), at(7, 14, 32)].into_iter()),
            "Saturday, 7 November 2026, 14:32:00–15:10:00"
        );
        assert_eq!(
            span([at(7, 14, 32), at(8, 9, 10)].into_iter()),
            "7 Nov 2026, 14:32 – 8 Nov 2026, 09:10"
        );
        assert_eq!(span(std::iter::empty()), "");

        let sweep: Rc<[SharedString]> = Rc::new(["Sweep of 1.4 s".into()]);
        let live: Rc<[SharedString]> = Rc::new(["Live, pink noise".into()]);
        assert_eq!(common_about([&sweep, &sweep].into_iter()), sweep.to_vec());
        assert_eq!(
            common_about([&sweep, &live].into_iter()),
            ["Measured in different ways, see each measurement"]
        );
    }
}
