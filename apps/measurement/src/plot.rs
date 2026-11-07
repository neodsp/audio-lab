//! The plots of the tabs: magnitude and phase over log frequency, impulse responses over time.
//!
//! Like in Figma or Excalidraw, scrolling (two fingers on a touchpad) moves the view and
//! Ctrl+scroll or a pinch zooms around the mouse; Ctrl+Alt+scroll zooms only the time or
//! frequency, Ctrl+Shift+scroll only the value. Drag to move the view, right-drag to zoom into a
//! box, double-click to go back to the default view. The view never follows the data by
//! itself: [`Plot::fit_to_data`] zooms to the curves on request.

use std::rc::Rc;

use gpui_kit::component::ActiveTheme as _;
use gpui_kit::*;

use crate::curves::{Curves, draw_positions, split_at_wraps};
use crate::theme::palette;

/// Room for the value labels left of the plot area.
const LEFT: f32 = 56.0;
/// Room for the frequency or time labels below the plot area.
const BOTTOM: f32 = 24.0;
const TOP: f32 = 10.0;
const RIGHT: f32 = 14.0;
const LABEL_SIZE: f32 = 11.0;
const LINE_WIDTH: f32 = 1.5;
/// The width of a highlighted trace.
const HIGHLIGHT_WIDTH: f32 = 3.0;
/// How much of their color the other traces keep while some are highlighted.
const DIMMED: f32 = 0.35;
/// A hidden trace under the mouse in the list is dashed: so long on, so long off, in pixels.
const DASH: f32 = 7.0;
const GAP: f32 = 5.0;
/// A path's vertex indices are 16 bit, so long curves are drawn in pieces.
const POINTS_PER_PATH: usize = 2000;
/// At most this many traces are listed under the mouse.
const MAX_READOUT: usize = 12;

#[derive(Clone, Copy, PartialEq)]
pub enum PlotKind {
    Magnitude,
    Phase,
    Impulse,
}

impl PlotKind {
    fn is_frequency(self) -> bool {
        self != Self::Impulse
    }

    /// The phase always shows the full turn: only the frequency zooms and pans.
    fn y_fixed(self) -> bool {
        self == Self::Phase
    }

    fn format_x(self, x: f64, precise: bool) -> String {
        match self {
            Self::Impulse if precise => format!("{x:.2} ms"),
            Self::Impulse => format!("{x} ms"),
            _ => format_hz(10f64.powf(x), if precise { 1 } else { 0 }),
        }
    }

    fn format_y(self, y: f64, step: f64) -> String {
        match self {
            Self::Magnitude => format!("{} dB", format_number(y, step)),
            Self::Phase => format!("{y:.0}°"),
            Self::Impulse => format_number(y, step),
        }
    }

    fn format_value(self, y: f64) -> String {
        match self {
            Self::Magnitude => format!("{y:.1} dB"),
            Self::Phase => format!("{y:.1}°"),
            Self::Impulse => format!("{y:.4}"),
        }
    }
}

/// What a trace draws.
#[derive(Clone)]
pub enum Series {
    /// The magnitude or phase, depending on the plot.
    Frequency(Rc<Curves>),
    /// Points sorted by x.
    Points(Rc<Vec<(f64, f64)>>),
}

#[derive(Clone)]
pub struct Trace {
    /// The measurement it shows.
    pub id: u64,
    pub name: SharedString,
    pub color: Hsla,
    pub series: Series,
    /// Where a live measurement's coherence is below this, its curve isn't drawn.
    pub min_coherence: f64,
    /// Hidden, but shown while the mouse is over it in the list. Fitting ignores it.
    pub preview: bool,
}

impl Trace {
    /// The value at `x`, if there is data there.
    fn value_at(&self, kind: PlotKind, x: f64) -> Option<f64> {
        match &self.series {
            Series::Frequency(curves) => {
                let (low, high) = curves.log_range;
                if !(low..=high).contains(&x) {
                    return None;
                }
                let hz = 10f64.powf(x);
                if !curves.trusted(hz, self.min_coherence) {
                    return None;
                }
                Some(match kind {
                    PlotKind::Phase => curves.phase_degrees(hz),
                    _ => curves.magnitude_db(hz),
                })
            }
            Series::Points(points) => {
                let first = points.first()?.0;
                let last = points.last()?.0;
                if !(first..=last).contains(&x) {
                    return None;
                }
                let index = points.partition_point(|p| p.0 < x).min(points.len() - 1);
                let nearest = match index.checked_sub(1) {
                    Some(before) if x - points[before].0 < points[index].0 - x => before,
                    _ => index,
                };
                Some(points[nearest].1)
            }
        }
    }
}

/// The visible part of the plot, in plot units: log10 Hz or ms, and dB, degrees or amplitude.
#[derive(Clone, Copy, PartialEq, Debug)]
struct View {
    x: (f64, f64),
    y: (f64, f64),
}

enum Drag {
    Pan {
        last: Point<Pixels>,
    },
    Zoom {
        start: Point<Pixels>,
        current: Point<Pixels>,
    },
}

/// The user moved or zoomed a frequency plot: the frequencies it shows now, in log10 Hz.
/// Magnitude and phase pass it on to each other, so they always show the same frequencies.
pub struct FrequencyChanged(pub (f64, f64));

pub struct Plot {
    kind: PlotKind,
    traces: Vec<Trace>,
    /// The measurements to draw on top and thicker, as the mouse is over them in the list.
    highlight: Rc<[u64]>,
    /// The frequency range of the data, in log10 Hz, for fitting a frequency plot to it.
    data_x: (f64, f64),
    view: View,
    /// Where the plot was painted last, in window coordinates.
    bounds: Bounds<Pixels>,
    hover: Option<Point<Pixels>>,
    drag: Option<Drag>,
}

impl Plot {
    pub fn new(kind: PlotKind) -> Self {
        Self {
            kind,
            traces: Vec::new(),
            highlight: Rc::new([]),
            data_x: AUDIO_RANGE,
            view: default_view(kind),
            bounds: Bounds::default(),
            hover: None,
            drag: None,
        }
    }

    /// Shows `traces`, whose data spans `freq_range` (Hz). The view stays where it is.
    pub fn set_traces(
        &mut self,
        traces: Vec<Trace>,
        freq_range: (f64, f64),
        cx: &mut Context<Self>,
    ) {
        self.traces = traces;
        self.data_x = (freq_range.0.log10(), freq_range.1.log10());
        cx.notify();
    }

    /// Zooms to the shown curves.
    pub fn fit_to_data(&mut self, cx: &mut Context<Self>) {
        let before = self.view.x;
        self.view = self.fitted_view();
        self.user_changed(before, cx);
    }

    /// Goes back to the default view.
    pub fn reset_view(&mut self, cx: &mut Context<Self>) {
        let before = self.view.x;
        self.view = default_view(self.kind);
        self.user_changed(before, cx);
    }

    /// Draws the traces of the measurements `ids` on top and thicker, and the others fainter.
    pub fn set_highlight(&mut self, ids: Rc<[u64]>, cx: &mut Context<Self>) {
        if self.highlight != ids {
            self.highlight = ids;
            cx.notify();
        }
    }

    /// Shows the frequencies `x` (log10 Hz) another plot was moved to.
    pub fn link_frequency(&mut self, x: (f64, f64), cx: &mut Context<Self>) {
        if self.view.x == x {
            return;
        }
        self.view.x = x;
        cx.notify();
    }

    /// Redraws after the user moved or zoomed, telling a change of the frequencies to the
    /// other frequency plot. `before` is the x range from before.
    fn user_changed(&mut self, before: (f64, f64), cx: &mut Context<Self>) {
        if self.kind.is_frequency() && self.view.x != before {
            cx.emit(FrequencyChanged(self.view.x));
        }
        cx.notify();
    }

    /// The traces the view fits to: the shown ones, not previews of hidden ones, unless there
    /// is nothing else.
    fn fitted(&self) -> impl Iterator<Item = &Trace> {
        let only_previews = self.traces.iter().all(|trace| trace.preview);
        self.traces
            .iter()
            .filter(move |trace| only_previews || !trace.preview)
    }

    /// The view that fits the data, or the default one without data.
    fn fitted_view(&self) -> View {
        match self.kind {
            PlotKind::Magnitude => {
                let (low, high) = self
                    .fitted()
                    .filter_map(|trace| match &trace.series {
                        Series::Frequency(curves) => {
                            Some(curves.magnitude_extent(trace.min_coherence))
                        }
                        Series::Points(_) => None,
                    })
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), (low, high)| {
                        (a.min(low), b.max(high))
                    });
                let y = if low.is_finite() && high.is_finite() {
                    // a deep notch shouldn't squeeze the rest
                    let low = low.max(high - 90.0);
                    (
                        (low / 5.0).floor() * 5.0 - 5.0,
                        (high / 5.0).ceil() * 5.0 + 5.0,
                    )
                } else {
                    default_view(self.kind).y
                };
                View { x: self.data_x, y }
            }
            PlotKind::Phase => View {
                x: self.data_x,
                y: default_view(self.kind).y,
            },
            PlotKind::Impulse => {
                let (mut first, mut last, mut peak) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
                for trace in self.fitted() {
                    if let Series::Points(points) = &trace.series {
                        if let (Some(a), Some(b)) = (points.first(), points.last()) {
                            first = first.min(a.0);
                            last = last.max(b.0);
                        }
                        peak = points.iter().fold(peak, |m, p| m.max(p.1.abs()));
                    }
                }
                if first < last {
                    let peak = if peak > 0.0 { peak * 1.1 } else { 1.0 };
                    View {
                        x: (first, last),
                        y: (-peak, peak),
                    }
                } else {
                    default_view(self.kind)
                }
            }
        }
    }

    fn area(&self) -> Bounds<Pixels> {
        plot_area(self.bounds)
    }

    /// The plot position at the window position `at`.
    fn to_plot(&self, at: Point<Pixels>) -> (f64, f64) {
        Transform::new(self.area(), self.view).to_plot(at)
    }

    fn zoom(&mut self, at: Point<Pixels>, factor_x: f64, factor_y: f64) {
        let (x, y) = self.to_plot(at);
        let scale = |(low, high): (f64, f64), center: f64, factor: f64| {
            let low = center - (center - low) * factor;
            let high = center + (high - center) * factor;
            if high - low > 1e-9 {
                (low, high)
            } else {
                (low, high.max(low + 1e-9))
            }
        };
        self.view.x = scale(self.view.x, x, factor_x);
        if !self.kind.y_fixed() {
            self.view.y = scale(self.view.y, y, factor_y);
        }
    }

    /// Like Figma or Excalidraw: scrolling (two fingers on a touchpad) moves the view, with Ctrl
    /// it zooms around the mouse, with Ctrl+Alt only horizontally (the keys sit side by side),
    /// with Ctrl+Shift only vertically (the keys sit on top of each other). Shift turns a mouse
    /// wheel's vertical scrolling into horizontal moving.
    fn on_scroll(&mut self, event: &ScrollEvent, window: &mut Window, cx: &mut Context<Self>) {
        let before = self.view.x;
        let delta = event.delta.pixel_delta(window.line_height());
        let (mut dx, mut dy) = (delta.x.as_f32() as f64, delta.y.as_f32() as f64);
        let modifiers = event.modifiers;
        if modifiers.control || modifiers.platform {
            let amount = if dy != 0.0 { dy } else { dx };
            let factor = (-amount * 0.002).exp();
            let (fx, fy) = match (modifiers.shift, modifiers.alt) {
                (false, true) => (factor, 1.0),
                (true, false) => (1.0, factor),
                _ => (factor, factor),
            };
            self.zoom(event.position, fx, fy);
        } else {
            if modifiers.shift && dx == 0.0 {
                (dx, dy) = (dy, 0.0);
            }
            self.pan_by_pixels(dx, dy);
        }
        self.user_changed(before, cx);
    }

    /// Zooms with a touchpad pinch.
    fn on_pinch(&mut self, event: &PinchEvent, cx: &mut Context<Self>) {
        let before = self.view.x;
        let factor = 1.0 / (1.0 + f64::from(event.delta)).max(0.1);
        self.zoom(event.position, factor, factor);
        self.user_changed(before, cx);
    }

    /// Moves what is shown along with content dragged by `dx`, `dy` pixels.
    fn pan_by_pixels(&mut self, dx: f64, dy: f64) {
        let area = self.area();
        let width = area.size.width.as_f32().max(1.0) as f64;
        let height = area.size.height.as_f32().max(1.0) as f64;
        let shift_x = -dx / width * (self.view.x.1 - self.view.x.0);
        self.view.x = (self.view.x.0 + shift_x, self.view.x.1 + shift_x);
        if !self.kind.y_fixed() {
            // the value axis points up, the screen down
            let shift_y = dy / height * (self.view.y.1 - self.view.y.0);
            self.view.y = (self.view.y.0 + shift_y, self.view.y.1 + shift_y);
        }
    }

    fn on_mouse_down(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        match event.button {
            MouseButton::Left if event.click_count >= 2 => {
                self.drag = None;
                self.reset_view(cx);
            }
            MouseButton::Left => {
                self.drag = Some(Drag::Pan {
                    last: event.position,
                })
            }
            MouseButton::Right => {
                self.drag = Some(Drag::Zoom {
                    start: event.position,
                    current: event.position,
                })
            }
            _ => {}
        }
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        self.hover = Some(event.position);
        let before = self.view.x;
        let transform = Transform::new(self.area(), self.view);
        match &mut self.drag {
            Some(Drag::Pan { last }) => {
                let (x0, y0) = transform.to_plot(*last);
                let (x1, y1) = transform.to_plot(event.position);
                *last = event.position;
                self.view.x = (self.view.x.0 - (x1 - x0), self.view.x.1 - (x1 - x0));
                if !self.kind.y_fixed() {
                    self.view.y = (self.view.y.0 - (y1 - y0), self.view.y.1 - (y1 - y0));
                }
            }
            Some(Drag::Zoom { current, .. }) => *current = event.position,
            None => {}
        }
        self.user_changed(before, cx);
    }

    fn on_mouse_up(&mut self, cx: &mut Context<Self>) {
        let before = self.view.x;
        if let Some(Drag::Zoom { start, current }) = self.drag.take() {
            let width = (current.x - start.x).abs().as_f32();
            let height = (current.y - start.y).abs().as_f32();
            if width > 6.0 && (height > 6.0 || self.kind.y_fixed()) {
                let (x0, y0) = self.to_plot(start);
                let (x1, y1) = self.to_plot(current);
                self.view.x = (x0.min(x1), x0.max(x1));
                if !self.kind.y_fixed() {
                    self.view.y = (y0.min(y1), y0.max(y1));
                }
            }
        }
        self.user_changed(before, cx);
    }
}

impl EventEmitter<FrequencyChanged> for Plot {}

impl Render for Plot {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let scene = Scene {
            kind: self.kind,
            traces: self.traces.clone(),
            highlight: self.highlight.clone(),
            view: self.view,
            hover: self
                .hover
                .filter(|_| !matches!(self.drag, Some(Drag::Pan { .. }))),
            zoom_box: match self.drag {
                Some(Drag::Zoom { start, current }) => Some((start, current)),
                _ => None,
            },
            font: font(cx.theme().font_family.clone()),
        };
        let this = cx.entity().downgrade();
        div()
            .id(match self.kind {
                PlotKind::Magnitude => "magnitude-plot",
                PlotKind::Phase => "phase-plot",
                PlotKind::Impulse => "impulse-plot",
            })
            .size_full()
            .bg(palette::editor(cx))
            .cursor_crosshair()
            .on_scroll(cx.listener(Self::on_scroll))
            .on_pinch(cx.listener(|this, event, _, cx| this.on_pinch(event, cx)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event, _, cx| this.on_mouse_down(event, cx)),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, event, _, cx| this.on_mouse_down(event, cx)),
            )
            .on_mouse_move(cx.listener(|this, event, _, cx| this.on_mouse_move(event, cx)))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| this.on_mouse_up(cx)),
            )
            .on_mouse_up(
                MouseButton::Right,
                cx.listener(|this, _, _, cx| this.on_mouse_up(cx)),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| this.on_mouse_up(cx)),
            )
            .on_mouse_up_out(
                MouseButton::Right,
                cx.listener(|this, _, _, cx| this.on_mouse_up(cx)),
            )
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !hovered {
                    this.hover = None;
                    cx.notify();
                }
            }))
            .child(
                canvas(
                    move |bounds, _, cx| {
                        this.update(cx, |plot, _| plot.bounds = bounds).ok();
                    },
                    move |bounds, _, window, cx| scene.paint(bounds, window, cx),
                )
                .size_full(),
            )
    }
}

fn plot_area(bounds: Bounds<Pixels>) -> Bounds<Pixels> {
    Bounds::new(
        point(bounds.origin.x + px(LEFT), bounds.origin.y + px(TOP)),
        size(
            (bounds.size.width - px(LEFT + RIGHT)).max(px(1.0)),
            (bounds.size.height - px(TOP + BOTTOM)).max(px(1.0)),
        ),
    )
}

/// Maps between plot units and window pixels.
#[derive(Clone, Copy)]
struct Transform {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
    view: View,
}

impl Transform {
    fn new(area: Bounds<Pixels>, view: View) -> Self {
        Self {
            left: area.origin.x.as_f32() as f64,
            top: area.origin.y.as_f32() as f64,
            width: area.size.width.as_f32() as f64,
            height: area.size.height.as_f32() as f64,
            view,
        }
    }

    fn x(&self, x: f64) -> f64 {
        self.left + (x - self.view.x.0) / (self.view.x.1 - self.view.x.0) * self.width
    }

    fn y(&self, y: f64) -> f64 {
        self.top + (self.view.y.1 - y) / (self.view.y.1 - self.view.y.0) * self.height
    }

    fn to_screen(self, (x, y): (f64, f64)) -> Point<Pixels> {
        // far outside values are clamped, so the tessellation stays sane
        let limit = |v: f64, start: f64, len: f64| v.clamp(start - 4.0 * len, start + 5.0 * len);
        point(
            px(limit(self.x(x), self.left, self.width) as f32),
            px(limit(self.y(y), self.top, self.height) as f32),
        )
    }

    fn to_plot(self, at: Point<Pixels>) -> (f64, f64) {
        let fx = (at.x.as_f32() as f64 - self.left) / self.width;
        let fy = (at.y.as_f32() as f64 - self.top) / self.height;
        (
            self.view.x.0 + fx * (self.view.x.1 - self.view.x.0),
            self.view.y.1 - fy * (self.view.y.1 - self.view.y.0),
        )
    }
}

/// Everything one frame paints.
struct Scene {
    kind: PlotKind,
    traces: Vec<Trace>,
    highlight: Rc<[u64]>,
    view: View,
    hover: Option<Point<Pixels>>,
    zoom_box: Option<(Point<Pixels>, Point<Pixels>)>,
    font: Font,
}

/// A grid line and maybe its label.
struct Mark {
    value: f64,
    strong: bool,
    label: Option<String>,
}

impl Scene {
    fn paint(self, bounds: Bounds<Pixels>, window: &mut Window, cx: &mut App) {
        let area = plot_area(bounds);
        let transform = Transform::new(area, self.view);

        let x_marks = self.x_marks(&transform);
        let (y_marks, y_step) = self.y_marks(&transform);

        // grid
        for mark in &x_marks {
            let x = px(transform.x(mark.value) as f32);
            let color = if mark.strong {
                palette::grid_strong(cx)
            } else {
                palette::grid(cx)
            };
            window.paint_quad(fill(
                Bounds::new(point(x, area.origin.y), size(px(1.0), area.size.height)),
                color,
            ));
        }
        for mark in &y_marks {
            let y = px(transform.y(mark.value) as f32);
            let color = if mark.strong {
                palette::grid_strong(cx)
            } else {
                palette::grid(cx)
            };
            window.paint_quad(fill(
                Bounds::new(point(area.origin.x, y), size(area.size.width, px(1.0))),
                color,
            ));
        }

        // labels
        let label_color = palette::text_muted(cx);
        let mut right_of_last = f32::NEG_INFINITY;
        for mark in &x_marks {
            let Some(label) = &mark.label else { continue };
            let line = self.shape(label, label_color, window);
            let right_edge = (bounds.origin.x + bounds.size.width).as_f32() - 2.0;
            let x = (transform.x(mark.value) as f32 - line.width.as_f32() / 2.0)
                .min(right_edge - line.width.as_f32());
            if x < right_of_last + 10.0 {
                continue;
            }
            right_of_last = x + line.width.as_f32();
            let y = area.origin.y + area.size.height + px(6.0);
            line.paint(
                point(px(x), y),
                px(LABEL_SIZE + 2.0),
                TextAlign::Left,
                None,
                window,
                cx,
            )
            .ok();
        }
        for mark in &y_marks {
            let label = mark
                .label
                .clone()
                .unwrap_or_else(|| self.kind.format_y(mark.value, y_step));
            let line = self.shape(&label, label_color, window);
            let x = area.origin.x - px(8.0) - line.width;
            let y = px(transform.y(mark.value) as f32 - (LABEL_SIZE + 2.0) / 2.0);
            line.paint(
                point(x, y),
                px(LABEL_SIZE + 2.0),
                TextAlign::Left,
                None,
                window,
                cx,
            )
            .ok();
        }
        // curves, clipped to the plot area
        window.with_content_mask(Some(ContentMask { bounds: area }), |window| {
            // the highlighted ones last, so on top
            let highlighted = |trace: &&Trace| self.highlight.contains(&trace.id);
            let others = self.traces.iter().filter(|t| !highlighted(t));
            // hidden or not, what is under the mouse in the list stands out
            let dim = self.traces.iter().any(|t| highlighted(&t));
            for trace in others {
                let color = if dim {
                    trace.color.opacity(DIMMED)
                } else {
                    trace.color
                };
                for piece in self.trace_points(trace, &transform) {
                    stroke(window, &piece, color, LINE_WIDTH, &transform);
                }
            }
            // hidden ones under the mouse in the list dashed, shown ones solid
            for trace in self.traces.iter().filter(highlighted) {
                for piece in self.trace_points(trace, &transform) {
                    if trace.preview {
                        stroke_dashed(window, &piece, trace.color, HIGHLIGHT_WIDTH, &transform);
                    } else {
                        stroke(window, &piece, trace.color, HIGHLIGHT_WIDTH, &transform);
                    }
                }
            }
            if let Some((start, current)) = self.zoom_box {
                let mut rect = Bounds::from_corners(
                    point(start.x.min(current.x), start.y.min(current.y)),
                    point(start.x.max(current.x), start.y.max(current.y)),
                );
                if self.kind.y_fixed() {
                    rect.origin.y = area.origin.y;
                    rect.size.height = area.size.height;
                }
                window.paint_quad(
                    fill(rect, palette::accent(cx).opacity(0.12))
                        .border_widths(px(1.0))
                        .border_color(palette::accent(cx).opacity(0.6)),
                );
            }
        });

        // the frame
        window.paint_quad(
            fill(area, transparent_black())
                .border_widths(px(1.0))
                .border_color(palette::border(cx)),
        );

        // where the mouse is now: a plot may have missed the mouse leaving it, e.g. while
        // something was dragged, and nothing is read out while a tab is dragged over it
        let at = window.mouse_position();
        if self.hover.is_some() && area.contains(&at) && !cx.has_active_drag() {
            self.paint_readout(at, area, &transform, window, cx);
        }
    }

    fn shape(&self, text: &str, color: Hsla, window: &mut Window) -> ShapedLine {
        let text = SharedString::from(text.to_string());
        let run = TextRun {
            len: text.len(),
            font: self.font.clone(),
            color,
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        window
            .text_system()
            .shape_line(text, px(LABEL_SIZE), &[run], None)
    }

    /// The points of `trace` to draw in the current view, in plot units, in pieces.
    fn trace_points(&self, trace: &Trace, transform: &Transform) -> Vec<Vec<(f64, f64)>> {
        match &trace.series {
            Series::Frequency(curves) => {
                let positions = draw_positions(self.view.x, transform.width, curves.log_range);
                // in pieces, with gaps where a live measurement can't be trusted
                let mut runs = vec![Vec::new()];
                for x in positions {
                    let hz = 10f64.powf(x);
                    if !curves.trusted(hz, trace.min_coherence) {
                        if !runs.last().is_some_and(Vec::is_empty) {
                            runs.push(Vec::new());
                        }
                        continue;
                    }
                    let y = match self.kind {
                        PlotKind::Phase => curves.phase_degrees(hz),
                        _ => curves.magnitude_db(hz),
                    };
                    runs.last_mut().expect("starts with one").push((x, y));
                }
                if self.kind == PlotKind::Phase {
                    runs.iter().flat_map(|run| split_at_wraps(run)).collect()
                } else {
                    runs
                }
            }
            Series::Points(points) => vec![decimate(points, self.view.x, transform.width)],
        }
    }

    fn x_marks(&self, transform: &Transform) -> Vec<Mark> {
        let (low, high) = self.view.x;
        if self.kind.is_frequency() {
            let decade_px = transform.width / (high - low);
            let mut marks = Vec::new();
            for exp in low.floor() as i32 - 1..=high.ceil() as i32 + 1 {
                for mult in 1..10 {
                    let value = (10f64.powi(exp) * mult as f64).log10();
                    if !(low..=high).contains(&value) {
                        continue;
                    }
                    let show_line = mult == 1 || decade_px > 60.0;
                    let show_label = match mult {
                        1 => true,
                        2 | 5 => decade_px > 160.0,
                        _ => decade_px > 480.0,
                    };
                    if show_line {
                        marks.push(Mark {
                            value,
                            strong: mult == 1,
                            label: show_label.then(|| self.kind.format_x(value, false)),
                        });
                    }
                }
            }
            marks
        } else {
            let step = nice_step(high - low, transform.width, 90.0);
            linear_marks(low, high, step)
                .map(|value| Mark {
                    value,
                    strong: value.abs() < step * 1e-6,
                    label: Some(match self.kind {
                        PlotKind::Impulse => format!("{} ms", format_number(value, step)),
                        _ => format_number(value, step),
                    }),
                })
                .collect()
        }
    }

    fn y_marks(&self, transform: &Transform) -> (Vec<Mark>, f64) {
        let (low, high) = self.view.y;
        let step = if self.kind == PlotKind::Phase {
            let per_degree = transform.height / (high - low);
            [15.0, 45.0, 90.0, 180.0]
                .into_iter()
                .find(|step| step * per_degree >= 22.0)
                .unwrap_or(180.0)
        } else {
            nice_step(high - low, transform.height, 36.0)
        };
        let marks = linear_marks(low, high, step)
            .map(|value| Mark {
                value,
                strong: value.abs() < step * 1e-6,
                label: None,
            })
            .collect();
        (marks, step)
    }

    /// A crosshair, a dot on each curve and a box listing the values under the mouse.
    fn paint_readout(
        &self,
        at: Point<Pixels>,
        area: Bounds<Pixels>,
        transform: &Transform,
        window: &mut Window,
        cx: &mut App,
    ) {
        let (x, y) = transform.to_plot(at);
        let crosshair = palette::text_muted(cx).opacity(0.35);
        window.paint_quad(fill(
            Bounds::new(point(at.x, area.origin.y), size(px(1.0), area.size.height)),
            crosshair,
        ));
        window.paint_quad(fill(
            Bounds::new(point(area.origin.x, at.y), size(area.size.width, px(1.0))),
            crosshair,
        ));

        let values: Vec<(&Trace, f64)> = self
            .traces
            .iter()
            .filter_map(|trace| Some((trace, trace.value_at(self.kind, x)?)))
            .collect();
        for (trace, value) in &values {
            let center = transform.to_screen((x, *value));
            if area.contains(&center) {
                let radius = px(3.5);
                window.paint_quad(
                    fill(
                        Bounds::new(
                            point(center.x - radius, center.y - radius),
                            size(radius * 2.0, radius * 2.0),
                        ),
                        trace.color,
                    )
                    .corner_radii(radius)
                    .border_widths(px(1.5))
                    .border_color(palette::editor(cx)),
                );
            }
        }

        // the box: the position, then one row per curve
        let title = if values.is_empty() {
            format!(
                "{}  {}",
                self.kind.format_x(x, true),
                self.kind.format_value(y)
            )
        } else {
            self.kind.format_x(x, true)
        };
        let title = self.shape(&title, palette::text(cx), window);
        let rows: Vec<(Hsla, ShapedLine, ShapedLine)> = values
            .iter()
            .take(MAX_READOUT)
            .map(|(trace, value)| {
                let name = if trace.preview {
                    format!("{} (hidden)", trace.name)
                } else {
                    trace.name.to_string()
                };
                (
                    trace.color,
                    self.shape(&name, palette::text_muted(cx), window),
                    self.shape(&self.kind.format_value(*value), palette::text(cx), window),
                )
            })
            .collect();
        let row_height = px(18.0);
        let padding = px(8.0);
        let swatch = px(8.0);
        let name_width = rows
            .iter()
            .map(|r| r.1.width)
            .fold(px(0.0), |a, b| a.max(b));
        let value_width = rows
            .iter()
            .map(|r| r.2.width)
            .fold(px(0.0), |a, b| a.max(b));
        let rows_width = if rows.is_empty() {
            px(0.0)
        } else {
            swatch + px(6.0) + name_width + px(16.0) + value_width
        };
        let more = values.len().saturating_sub(MAX_READOUT);
        let width = title.width.max(rows_width) + padding * 2.0;
        let height =
            row_height * (1 + rows.len() + usize::from(more > 0)) as f32 + padding * 2.0 - px(4.0);
        // beside the mouse, flipped away from the edges
        let mut origin = point(at.x + px(14.0), at.y + px(14.0));
        if origin.x + width > area.origin.x + area.size.width {
            origin.x = at.x - px(14.0) - width;
        }
        if origin.y + height > area.origin.y + area.size.height {
            origin.y = (at.y - px(14.0) - height).max(area.origin.y);
        }
        let rect = Bounds::new(origin, size(width, height));
        window.paint_quad(
            fill(rect, palette::elevated(cx))
                .corner_radii(px(6.0))
                .border_widths(px(1.0))
                .border_color(palette::border(cx)),
        );
        let mut y = origin.y + padding - px(2.0);
        let left = origin.x + padding;
        title
            .paint(
                point(left, y),
                row_height,
                TextAlign::Left,
                None,
                window,
                cx,
            )
            .ok();
        for (color, name, value) in rows {
            y += row_height;
            let swatch_top = y + (row_height - swatch) / 2.0;
            window.paint_quad(
                fill(
                    Bounds::new(point(left, swatch_top), size(swatch, swatch)),
                    color,
                )
                .corner_radii(px(2.0)),
            );
            name.paint(
                point(left + swatch + px(6.0), y),
                row_height,
                TextAlign::Left,
                None,
                window,
                cx,
            )
            .ok();
            let value_left = origin.x + width - padding - value.width;
            value
                .paint(
                    point(value_left, y),
                    row_height,
                    TextAlign::Left,
                    None,
                    window,
                    cx,
                )
                .ok();
        }
        if more > 0 {
            y += row_height;
            self.shape(
                &format!("and {more} more"),
                palette::text_placeholder(cx),
                window,
            )
            .paint(
                point(left, y),
                row_height,
                TextAlign::Left,
                None,
                window,
                cx,
            )
            .ok();
        }
    }
}

/// The audible range, 20 Hz to 20 kHz, in log10 Hz.
const AUDIO_RANGE: (f64, f64) = (1.301_029_995_663_981_2, 4.301_029_995_663_981);

/// What a plot shows until zoomed or moved, and again after a double-click.
fn default_view(kind: PlotKind) -> View {
    match kind {
        PlotKind::Magnitude => View {
            x: AUDIO_RANGE,
            y: (-60.0, 12.0),
        },
        PlotKind::Phase => View {
            x: AUDIO_RANGE,
            y: (-180.0, 180.0),
        },
        PlotKind::Impulse => View {
            x: (0.0, 100.0),
            y: (-1.0, 1.0),
        },
    }
}

/// Strokes the polyline through `points` (plot units).
fn stroke(
    window: &mut Window,
    points: &[(f64, f64)],
    color: Hsla,
    width: f32,
    transform: &Transform,
) {
    if points.len() < 2 {
        return;
    }
    let mut start = 0;
    while start + 1 < points.len() {
        let end = (start + POINTS_PER_PATH).min(points.len());
        let mut path = PathBuilder::stroke(px(width));
        path.move_to(transform.to_screen(points[start]));
        for &p in &points[start + 1..end] {
            path.line_to(transform.to_screen(p));
        }
        if let Ok(path) = path.build() {
            window.paint_path(path, color);
        }
        // the next piece starts where this one ends
        start = end - 1;
    }
}

/// Strokes the polyline through `points` (plot units) in dashes of [`DASH`] pixels with gaps
/// of [`GAP`], measured along the line on screen, so they look the same at any zoom.
fn stroke_dashed(
    window: &mut Window,
    points: &[(f64, f64)],
    color: Hsla,
    width: f32,
    transform: &Transform,
) {
    let screen: Vec<(f32, f32)> = points
        .iter()
        .map(|&p| {
            let p = transform.to_screen(p);
            (p.x.as_f32(), p.y.as_f32())
        })
        .collect();
    let mut path = PathBuilder::stroke(px(width));
    let mut vertices = 0;
    // whether the pen draws, how far until it switches, and whether a dash is open
    let (mut on, mut left, mut open) = (true, DASH, false);
    for segment in screen.windows(2) {
        let (mut from, to) = (segment[0], segment[1]);
        let mut length = ((to.0 - from.0).powi(2) + (to.1 - from.1).powi(2)).sqrt();
        while length > 0.0 {
            let step = left.min(length);
            let t = step / length;
            let end = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
            if on {
                if !open {
                    path.move_to(point(px(from.0), px(from.1)));
                    open = true;
                }
                path.line_to(point(px(end.0), px(end.1)));
                vertices += 1;
            }
            left -= step;
            length -= step;
            from = end;
            if left <= 0.0 {
                on = !on;
                left = if on { DASH } else { GAP };
                open = false;
            }
        }
        // a path's vertex indices are 16 bit: long curves in pieces
        if vertices >= POINTS_PER_PATH {
            if let Ok(full) = std::mem::replace(&mut path, PathBuilder::stroke(px(width))).build() {
                window.paint_path(full, color);
            }
            vertices = 0;
            open = false;
        }
    }
    if let Ok(path) = path.build() {
        window.paint_path(path, color);
    }
}

/// The points within `visible` (and one beyond each side). With more than two per pixel, the
/// lowest and highest of each pixel column are kept, so peaks don't get lost.
fn decimate(points: &[(f64, f64)], visible: (f64, f64), width: f64) -> Vec<(f64, f64)> {
    let first = points
        .partition_point(|p| p.0 < visible.0)
        .saturating_sub(1);
    let last = (points.partition_point(|p| p.0 <= visible.1) + 1).min(points.len());
    let shown = &points[first..last];
    if (shown.len() as f64) <= 2.0 * width || width < 1.0 {
        return shown.to_vec();
    }
    let column_width = (visible.1 - visible.0) / width;
    let mut out = Vec::with_capacity(2 * width as usize + 4);
    let mut column = f64::NAN;
    let (mut low, mut high) = ((0.0, f64::INFINITY), (0.0, f64::NEG_INFINITY));
    let flush = |low: (f64, f64), high: (f64, f64), out: &mut Vec<(f64, f64)>| {
        if low.1.is_finite() {
            // in the order they came
            if low.0 <= high.0 {
                out.extend([low, high]);
            } else {
                out.extend([high, low]);
            }
        }
    };
    for &p in shown {
        let c = ((p.0 - visible.0) / column_width).floor();
        if c != column {
            flush(low, high, &mut out);
            column = c;
            low = p;
            high = p;
        } else {
            if p.1 < low.1 {
                low = p;
            }
            if p.1 > high.1 {
                high = p;
            }
        }
    }
    flush(low, high, &mut out);
    out
}

/// A round step between grid lines, about `target` pixels apart.
fn nice_step(span: f64, pixels: f64, target: f64) -> f64 {
    let raw = span * target / pixels.max(1.0);
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    let magnitude = 10f64.powf(raw.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * magnitude)
        .find(|&step| step >= raw)
        .unwrap_or(10.0 * magnitude)
}

fn linear_marks(low: f64, high: f64, step: f64) -> impl Iterator<Item = f64> {
    let first = (low / step).ceil() as i64;
    let last = (high / step).floor() as i64;
    (first..=last.max(first - 1)).map(move |i| i as f64 * step)
}

/// `value` with as many decimals as the grid `step` needs.
fn format_number(value: f64, step: f64) -> String {
    let decimals = if step > 0.0 {
        (-step.log10().floor()).max(0.0) as usize
    } else {
        0
    };
    let text = format!("{value:.decimals$}");
    // no "-0"
    if text
        .trim_start_matches('-')
        .chars()
        .all(|c| c == '0' || c == '.')
    {
        text.trim_start_matches('-').to_string()
    } else {
        text
    }
}

pub fn format_hz(hz: f64, decimals: usize) -> String {
    if hz >= 1000.0 {
        format!("{:.*} kHz", decimals, hz / 1000.0)
    } else {
        format!("{hz:.decimals$} Hz")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_frequencies_are_the_audible_range() {
        assert!((AUDIO_RANGE.0 - 20f64.log10()).abs() < 1e-12);
        assert!((AUDIO_RANGE.1 - 20_000f64.log10()).abs() < 1e-12);
    }

    // `gpui_kit::*` brings GPUI's test macro, which needs an app
    use core::prelude::v1::test;

    #[test]
    fn steps_are_round() {
        assert_eq!(nice_step(100.0, 400.0, 40.0), 10.0);
        assert_eq!(nice_step(100.0, 400.0, 60.0), 20.0);
        assert_eq!(nice_step(1.0, 400.0, 90.0), 0.5);
        let marks: Vec<f64> = linear_marks(-7.0, 12.0, 5.0).collect();
        assert_eq!(marks, [-5.0, 0.0, 5.0, 10.0]);
    }

    #[test]
    fn numbers_get_the_decimals_of_the_step() {
        assert_eq!(format_number(-0.0001, 0.5), "0.0");
        assert_eq!(format_number(12.0, 5.0), "12");
        assert_eq!(format_number(0.25, 0.05), "0.25");
        assert_eq!(format_hz(2000.0, 0), "2 kHz");
        assert_eq!(format_hz(31.5, 1), "31.5 Hz");
    }

    #[test]
    fn decimation_keeps_the_peaks() {
        let points: Vec<(f64, f64)> = (0..10_000)
            .map(|i| (i as f64, if i == 5_003 { 1.0 } else { 0.0 }))
            .collect();
        let drawn = decimate(&points, (0.0, 10_000.0), 100.0);
        assert!(drawn.len() <= 202);
        assert!(drawn.iter().any(|p| p.1 == 1.0));
        // zoomed in: every point, and one beyond each side
        let drawn = decimate(&points, (10.5, 20.5), 100.0);
        assert_eq!(drawn.first().unwrap().0, 10.0);
        assert_eq!(drawn.last().unwrap().0, 21.0);
    }

    #[test]
    fn transform_round_trips() {
        let area = Bounds::new(point(px(10.0), px(20.0)), size(px(200.0), px(100.0)));
        let view = View {
            x: (1.0, 3.0),
            y: (-10.0, 10.0),
        };
        let transform = Transform::new(area, view);
        let at = transform.to_screen((2.0, 5.0));
        assert_eq!((at.x, at.y), (px(110.0), px(45.0)));
        let (x, y) = transform.to_plot(at);
        assert!((x - 2.0).abs() < 1e-6 && (y - 5.0).abs() < 1e-6);
    }
}
