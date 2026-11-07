//! The curves of a response and the plots of the tabs.

use audio_lab::dsp::{
    FractionalOctaveSmoothingConfig, FractionalOctaveSmoothingMode, smooth_fractional_octave,
};
use audio_lab::signal::{FreqSignal, TimeSignal};
use eframe::egui;
use egui_plot::{GridInput, GridMark, HoverPosition, Line, Plot, PlotPoint, PlotPoints};

const DB_FLOOR: f64 = -150.0;
/// Points drawn per screen point along the frequency axis. The interpolation between the bins
/// is smooth, so more wouldn't look different.
const POINTS_PER_SCREEN_POINT: f64 = 1.0;

/// A response's smoothed magnitude and phase per bin, from which the curves are drawn at the
/// resolution of the screen.
pub struct Curves {
    /// What they were computed with: the smoothing (1/N octave) or `None`.
    pub smoothing: Option<f64>,
    /// The frequency range that is drawn, on the log axis (log10 of Hz).
    log_range: (f64, f64),
    first_bin: f64,
    bin_width: f64,
    magnitude_db: Vec<f64>,
    /// The phase as complex values, which interpolate across its wraps.
    re: Vec<f64>,
    im: Vec<f64>,
}

impl Curves {
    fn position(&self, hz: f64) -> f64 {
        (hz - self.first_bin) / self.bin_width
    }

    fn magnitude_db(&self, hz: f64) -> f64 {
        interpolate(&self.magnitude_db, self.position(hz))
    }

    fn phase_degrees(&self, hz: f64) -> f64 {
        let position = self.position(hz);
        interpolate(&self.im, position)
            .atan2(interpolate(&self.re, position))
            .to_degrees()
    }
}

/// The curves of `response` within `range`. The magnitude is smoothed on its own, the phase as
/// complex values, as in `plot::show_magnitude_and_phase`.
pub fn curves(response: &FreqSignal, range: (f64, f64), smoothing: Option<f64>) -> Curves {
    let smoothed = |mode| match smoothing {
        Some(num_fractions) => {
            let config = FractionalOctaveSmoothingConfig::new(num_fractions).with_mode(mode);
            smooth_fractional_octave(response, &config)
                .map(|(smoothed, _)| smoothed)
                .unwrap_or_else(|_| response.clone())
        }
        None => response.clone(),
    };
    let magnitude = smoothed(FractionalOctaveSmoothingMode::MagnitudeZeroPhase);
    let phase = smoothed(FractionalOctaveSmoothingMode::Complex);

    let freq_bins = response.freq_bins();
    let first_bin = freq_bins.first().copied().unwrap_or(0.0);
    let highest_bin = freq_bins.last().copied().unwrap_or(0.0);
    // the lowest bin above 0 Hz: 0 Hz has no place on a log axis
    let lowest_bin = freq_bins.iter().copied().find(|&f| f > 0.0).unwrap_or(1.0);
    let low = range.0.max(lowest_bin);
    let high = range.1.min(highest_bin).max(low);
    Curves {
        smoothing,
        log_range: (low.log10(), high.log10()),
        first_bin,
        bin_width: freq_bins.get(1).map_or(1.0, |f| f - first_bin),
        magnitude_db: magnitude
            .channel(0)
            .iter()
            .map(|h| (20.0 * h.norm().log10()).max(DB_FLOOR))
            .collect(),
        re: phase.channel(0).iter().map(|h| h.re).collect(),
        im: phase.channel(0).iter().map(|h| h.im).collect(),
    }
}

/// The impulse response over time in ms since the start of the recording, so the delays
/// between speakers stay visible.
pub fn impulse_points(impulse_response: &TimeSignal, start: usize) -> Vec<PlotPoint> {
    let ms_per_sample = 1000.0 / impulse_response.sample_rate();
    impulse_response
        .channel(0)
        .iter()
        .enumerate()
        .map(|(i, &s)| PlotPoint::new((start + i) as f64 * ms_per_sample, s))
        .collect()
}

/// Where to draw the curves on the log axis: evenly spaced, one per screen point of the
/// visible range, and beyond it by half the visible width on each side, so the curves don't
/// end at the old edges for the frame in which the view is panned or zoomed. Only within
/// `log_range`, where there is data.
fn draw_positions(visible: Option<(f64, f64)>, width: f64, log_range: (f64, f64)) -> Vec<f64> {
    let (low, high) = log_range;
    if low >= high {
        return Vec::new();
    }
    let (start, end, step) = match visible {
        Some((left, right)) if left < right && left < high && low < right && width >= 1.0 => {
            let margin = 0.5 * (right - left);
            let step = (right - left) / (width * POINTS_PER_SCREEN_POINT);
            (low.max(left - margin), high.min(right + margin), step)
        }
        // nothing to go by yet, e.g. in the first frame
        _ => (low, high, (high - low) / 2000.0),
    };
    let num_points = ((end - start) / step).ceil() as usize + 1;
    (0..num_points)
        .map(|i| (start + i as f64 * step).min(end))
        .collect()
}

/// [`draw_positions`] for the plot's current view.
fn positions_in_view(plot_ui: &egui_plot::PlotUi, log_range: (f64, f64)) -> Vec<f64> {
    let bounds = plot_ui.plot_bounds();
    let visible = bounds
        .is_valid_x()
        .then(|| (bounds.min()[0], bounds.max()[0]));
    let width = f64::from(plot_ui.transform().frame().width());
    draw_positions(visible, width, log_range)
}

/// `values` at the fractional index `position`, by monotone cubic interpolation: a smooth curve
/// that never goes above or below its neighbors, so bins that lie far apart don't turn into
/// lobes and notches that weren't measured.
fn interpolate(values: &[f64], position: f64) -> f64 {
    let Some(last) = values.len().checked_sub(1) else {
        return f64::NAN;
    };
    let index = (position.max(0.0).floor() as usize).min(last);
    let t = (position - index as f64).clamp(0.0, 1.0);
    let at = |i: isize| values[i.clamp(0, last as isize) as usize];
    let i = index as isize;
    let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
    // slopes at both ends: flat at a peak or dip, else the harmonic mean of the neighbors
    let slope = |before: f64, after: f64| {
        if before * after <= 0.0 {
            0.0
        } else {
            2.0 / (1.0 / before + 1.0 / after)
        }
    };
    let (m1, m2) = (slope(p1 - p0, p2 - p1), slope(p2 - p1, p3 - p2));
    let (t2, t3) = (t * t, t * t * t);
    (2.0 * t3 - 3.0 * t2 + 1.0) * p1
        + (t3 - 2.0 * t2 + t) * m1
        + (3.0 * t2 - 2.0 * t3) * p2
        + (t3 - t2) * m2
}

/// Splits a wrapped phase in degrees into pieces that each run to the edge where it wraps and
/// continue from the other edge, so a wrap isn't drawn as a line across the plot.
fn split_at_wraps(points: &[PlotPoint]) -> Vec<Vec<PlotPoint>> {
    let mut pieces = vec![Vec::new()];
    for (index, &point) in points.iter().enumerate() {
        if let Some(&last) = index.checked_sub(1).map(|i| &points[i]) {
            let step = point.y - last.y;
            // a jump of more than half a turn is a wrap: the phase went the short way round
            if step.abs() > 180.0 {
                let unwrapped = point.y - 360.0 * step.signum();
                let edge = if unwrapped > last.y { 180.0 } else { -180.0 };
                let x = last.x + (point.x - last.x) * (edge - last.y) / (unwrapped - last.y);
                pieces.last_mut().unwrap().push(PlotPoint::new(x, edge));
                pieces.push(vec![PlotPoint::new(x, -edge)]);
            }
        }
        pieces.last_mut().unwrap().push(point);
    }
    pieces
}

/// A frequency response to draw.
pub struct FreqTrace<'a> {
    pub name: &'a str,
    pub color: egui::Color32,
    pub curves: &'a Curves,
}

/// An impulse response to draw.
pub struct TimeTrace<'a> {
    pub name: &'a str,
    pub color: egui::Color32,
    pub points: &'a [PlotPoint],
}

/// The magnitude in dB over log frequency.
pub fn magnitude_plot(ui: &mut egui::Ui, traces: &[FreqTrace], range: (f64, f64)) {
    frequency_plot("magnitude", range, "dB")
        .y_axis_label("Magnitude (dB)")
        .show(ui, |plot_ui| {
            for trace in traces {
                let points = positions_in_view(plot_ui, trace.curves.log_range)
                    .into_iter()
                    .map(|x| PlotPoint::new(x, trace.curves.magnitude_db(10f64.powf(x))))
                    .collect();
                draw(plot_ui, trace.name, trace.color, points);
            }
        });
}

/// The phase in degrees, wrapped to ±180°, over log frequency.
pub fn phase_plot(ui: &mut egui::Ui, traces: &[FreqTrace], range: (f64, f64)) {
    frequency_plot("phase", range, "°")
        .y_axis_label("Phase (°)")
        .default_y_bounds(-180.0, 180.0)
        // the phase always shows the full turn; only the frequency zooms and pans
        .allow_zoom([true, false])
        .allow_drag([true, false])
        .allow_scroll([true, false])
        .allow_axis_zoom_drag([true, false])
        .allow_boxed_zoom(false)
        .y_grid_spacer(egui_plot::uniform_grid_spacer(|_| [15.0, 45.0, 90.0]))
        .y_axis_formatter(|mark, _| format!("{:.0}°", mark.value))
        .show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_y(-180.0..=180.0);
            for trace in traces {
                let points: Vec<PlotPoint> = positions_in_view(plot_ui, trace.curves.log_range)
                    .into_iter()
                    .map(|x| PlotPoint::new(x, trace.curves.phase_degrees(10f64.powf(x))))
                    .collect();
                for piece in split_at_wraps(&points) {
                    draw(plot_ui, trace.name, trace.color, piece);
                }
            }
        });
}

/// The impulse responses over time.
pub fn impulse_plot<'a>(ui: &mut egui::Ui, traces: &'a [TimeTrace<'a>]) {
    Plot::new("impulse")
        .x_axis_label("Time (ms)")
        .y_axis_label("Amplitude")
        .label_formatter(|pos| hover_label(pos, |x| format!("{x:.2} ms"), |y| format!("{y:.4}")))
        .show(ui, move |plot_ui| {
            for trace in traces {
                let line = Line::new(trace.name, PlotPoints::Borrowed(trace.points));
                plot_ui.line(line.color(trace.color));
            }
        });
}

fn draw(plot_ui: &mut egui_plot::PlotUi, name: &str, color: egui::Color32, points: Vec<PlotPoint>) {
    if points.len() > 1 {
        plot_ui.line(Line::new(name, PlotPoints::Owned(points)).color(color));
    }
}

/// A plot over log frequency, showing `range` at the start and after a double-click.
fn frequency_plot(id: &str, (low, high): (f64, f64), unit: &'static str) -> Plot<'static> {
    Plot::new(id)
        .x_axis_label("Frequency")
        .default_x_bounds(low.log10(), high.log10())
        .x_grid_spacer(spectrum_grid_spacer)
        .x_axis_formatter(|mark, _| format_hz(10f64.powf(mark.value), 0))
        .label_formatter(move |pos| {
            hover_label(
                pos,
                |x| format_hz(10f64.powf(x), 1),
                |y| format!("{y:.1} {unit}"),
            )
        })
}

/// Grid marks in log10 space: decades, then ×2 and ×5, then the other integers.
fn spectrum_grid_spacer(input: GridInput) -> Vec<GridMark> {
    let (lo, hi) = input.bounds;
    let mut marks = Vec::new();
    for exp in lo.floor() as i32 - 1..=hi.ceil() as i32 + 1 {
        for mult in 1..10 {
            let value = (10f64.powi(exp) * mult as f64).log10();
            if (lo..=hi).contains(&value) {
                let step_size = match mult {
                    1 => 1.0,
                    2 | 5 => 0.3,
                    _ => 0.1,
                };
                marks.push(GridMark { value, step_size });
            }
        }
    }
    marks
}

fn format_hz(hz: f64, decimals: usize) -> String {
    if hz >= 1000.0 {
        format!("{:.*} kHz", decimals, hz / 1000.0)
    } else {
        format!("{hz:.decimals$} Hz")
    }
}

/// The curve, position and value under the mouse.
fn hover_label(
    pos: &HoverPosition,
    format_x: impl Fn(f64) -> String,
    format_y: impl Fn(f64) -> String,
) -> Option<String> {
    let (name, point) = match pos {
        HoverPosition::NearDataPoint {
            plot_name,
            position,
            ..
        } => (*plot_name, position),
        HoverPosition::Elsewhere { position } => ("", position),
    };
    let text = format!("{}\n{}", format_x(point.x), format_y(point.y));
    Some(if name.is_empty() {
        text
    } else {
        format!("{name}\n{text}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_pieces_meet_at_the_edges() {
        let points: Vec<PlotPoint> = (0..200)
            .map(|i| {
                let wrapped = (-7.0 * i as f64 + 180.0f64).rem_euclid(360.0) - 180.0;
                PlotPoint::new(i as f64, wrapped)
            })
            .collect();
        let pieces = split_at_wraps(&points);
        for pair in pieces.windows(2) {
            let (end, start) = (pair[0].last().unwrap(), pair[1][0]);
            assert_eq!((end.x, end.y), (start.x, -start.y));
        }
    }

    fn impulse_response() -> FreqSignal {
        let data = audio_lab::signal::ndarray::Array2::from_shape_fn((1, 4800), |(_, i)| {
            if i == 10 { 1.0 } else { 0.0 }
        });
        TimeSignal::new(data, 48_000.0).unwrap().to_freq()
    }

    #[test]
    fn curves_are_smooth_and_cover_the_range() {
        let curves = curves(&impulse_response(), (20.0, 20_000.0), Some(3.0));
        assert!((curves.log_range.0 - 20f64.log10()).abs() < 1e-9);
        assert!((curves.log_range.1 - 20_000f64.log10()).abs() < 1e-9);
        // an impulse is flat at 0 dB
        for hz in [20.0, 31.4, 1000.0, 19_999.0] {
            assert!(curves.magnitude_db(hz).abs() < 0.1, "{hz}");
        }
    }

    #[test]
    fn positions_follow_the_view_at_screen_resolution() {
        let range = (20f64.log10(), 20_000f64.log10());
        // zoomed in to 200..400 Hz on 600 screen points
        let visible = (200f64.log10(), 400f64.log10());
        let positions = draw_positions(Some(visible), 600.0, range);
        let step = positions[1] - positions[0];
        assert!((step - (visible.1 - visible.0) / 600.0).abs() < 1e-12);
        // half a view beyond each side
        let margin = 0.5 * (visible.1 - visible.0);
        assert!((positions[0] - (visible.0 - margin)).abs() < 1e-9);
        assert!((positions.last().unwrap() - (visible.1 + margin)).abs() < 1e-9);

        // zoomed out beyond the data: only where there is data
        let positions = draw_positions(Some((0.0, 6.0)), 600.0, range);
        assert_eq!(positions[0], range.0);
        assert_eq!(*positions.last().unwrap(), range.1);
        assert!(positions.len() < 400);

        // no view yet: the whole range
        let positions = draw_positions(None, 0.0, range);
        assert_eq!((positions[0], *positions.last().unwrap()), range);
    }
}
