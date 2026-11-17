//! A response's magnitude, phase and group delay, ready to be drawn at any resolution.

use audio_lab::dsp::{
    FractionalOctaveSmoothingConfig, FractionalOctaveSmoothingMode, smooth_fractional_octave,
};
use audio_lab::signal::{FreqSignal, TimeSignal};

use crate::plot::PlotKind;

const DB_FLOOR: f64 = -150.0;

/// A response's smoothed magnitude, phase and group delay per bin, from which the curves are
/// drawn at the resolution of the screen.
pub struct Curves {
    /// What they were computed with: the smoothing (1/N octave) or `None`.
    pub smoothing: Option<f64>,
    /// The frequency range that is drawn, on the log axis (log10 of Hz).
    pub log_range: (f64, f64),
    first_bin: f64,
    bin_width: f64,
    magnitude_db: Vec<f64>,
    /// The phase as complex values, which interpolate across its wraps.
    re: Vec<f64>,
    im: Vec<f64>,
    /// The group delay in ms, from the smoothed phase, relative to the delay taken out of it.
    group_delay_ms: Vec<f64>,
    /// The coherence of a live measurement, smoothed like the magnitude. Where it is below the
    /// chosen limit, the curves aren't drawn.
    coherence: Option<Vec<f64>>,
}

impl Curves {
    /// The curves of `response` within `range`. The magnitude is smoothed on its own, the phase
    /// as complex values, the `coherence` like the magnitude.
    pub fn new(
        response: &FreqSignal,
        coherence: Option<&FreqSignal>,
        range: (f64, f64),
        smoothing: Option<f64>,
    ) -> Self {
        let smoothed = |signal: &FreqSignal, mode| match smoothing {
            Some(num_fractions) => {
                let config = FractionalOctaveSmoothingConfig::new(num_fractions).with_mode(mode);
                smooth_fractional_octave(signal, &config)
                    .map(|(smoothed, _)| smoothed)
                    .unwrap_or_else(|_| signal.clone())
            }
            None => signal.clone(),
        };
        let magnitude = smoothed(response, FractionalOctaveSmoothingMode::MagnitudeZeroPhase);
        let phase = smoothed(response, FractionalOctaveSmoothingMode::Complex);
        let coherence = coherence.map(|coherence| {
            smoothed(coherence, FractionalOctaveSmoothingMode::MagnitudeZeroPhase)
                .channel(0)
                .iter()
                .map(|c| c.norm())
                .collect()
        });

        let freq_bins = response.freq_bins();
        let first_bin = freq_bins.first().copied().unwrap_or(0.0);
        let highest_bin = freq_bins.last().copied().unwrap_or(0.0);
        // the lowest bin above 0 Hz: 0 Hz has no place on a log axis
        let lowest_bin = freq_bins.iter().copied().find(|&f| f > 0.0).unwrap_or(1.0);
        let low = range.0.max(lowest_bin);
        let high = range.1.min(highest_bin).max(low);
        let bin_width = freq_bins.get(1).map_or(1.0, |f| f - first_bin);
        let re: Vec<f64> = phase.channel(0).iter().map(|h| h.re).collect();
        let im: Vec<f64> = phase.channel(0).iter().map(|h| h.im).collect();
        Self {
            smoothing,
            log_range: (low.log10(), high.log10()),
            first_bin,
            bin_width,
            group_delay_ms: group_delay_ms(&re, &im, bin_width),
            magnitude_db: magnitude
                .channel(0)
                .iter()
                .map(|h| (20.0 * h.norm().log10()).max(DB_FLOOR))
                .collect(),
            re,
            im,
            coherence,
        }
    }

    /// Whether the curve is drawn at `hz`: always for a sweep, for a live measurement where
    /// the coherence reaches `min_coherence`.
    pub fn trusted(&self, hz: f64, min_coherence: f64) -> bool {
        match &self.coherence {
            Some(coherence) if min_coherence > 0.0 => {
                interpolate(coherence, self.position(hz)) >= min_coherence
            }
            _ => true,
        }
    }

    fn position(&self, hz: f64) -> f64 {
        (hz - self.first_bin) / self.bin_width
    }

    pub fn magnitude_db(&self, hz: f64) -> f64 {
        interpolate(&self.magnitude_db, self.position(hz))
    }

    /// Wrapped to ±180°.
    pub fn phase_degrees(&self, hz: f64) -> f64 {
        let position = self.position(hz);
        interpolate(&self.im, position)
            .atan2(interpolate(&self.re, position))
            .to_degrees()
    }

    /// What the plot `kind` shows at `hz`: the phase, the group delay, else the magnitude.
    pub fn value(&self, kind: PlotKind, hz: f64) -> f64 {
        match kind {
            PlotKind::Phase => self.phase_degrees(hz),
            PlotKind::GroupDelay => self.group_delay_ms(hz),
            _ => self.magnitude_db(hz),
        }
    }

    /// In ms, relative to the delay taken out of the phase: up to the impulse response's peak.
    pub fn group_delay_ms(&self, hz: f64) -> f64 {
        interpolate(&self.group_delay_ms, self.position(hz))
    }

    /// The lowest and highest magnitude within the drawn range where it is
    /// [`trusted`](Self::trusted), in dB.
    pub fn magnitude_extent(&self, min_coherence: f64) -> (f64, f64) {
        self.sampled(min_coherence, |hz| self.magnitude_db(hz))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), db| {
                (min.min(db), max.max(db))
            })
    }

    /// Where most of the group delay lies within the drawn range where it is
    /// [`trusted`](Self::trusted), in ms: without the few spikes at deep notches, which would
    /// squeeze the rest.
    pub fn group_delay_extent(&self, min_coherence: f64) -> (f64, f64) {
        let mut values: Vec<f64> = self
            .sampled(min_coherence, |hz| self.group_delay_ms(hz))
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            return (f64::INFINITY, f64::NEG_INFINITY);
        }
        values.sort_by(f64::total_cmp);
        let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
        (at(0.02), at(0.98))
    }

    /// `value` at points evenly spaced on the log axis of the drawn range, where it is trusted.
    fn sampled(&self, min_coherence: f64, value: impl Fn(f64) -> f64) -> impl Iterator<Item = f64> {
        let (low, high) = self.log_range;
        (0..=200)
            .map(move |i| 10f64.powf(low + (high - low) * i as f64 / 200.0))
            .filter(move |&hz| self.trusted(hz, min_coherence))
            .map(value)
    }
}

/// The group delay per bin in ms, −dφ/dω, from the response as complex values `re` and `im`
/// on bins `bin_width` Hz apart. The phase step between two bins is the angle of one times the
/// conjugate of the other, so it needs no unwrapping. Central differences keep it on the bins.
fn group_delay_ms(re: &[f64], im: &[f64], bin_width: f64) -> Vec<f64> {
    let len = re.len().min(im.len());
    if len < 2 {
        return vec![0.0; len];
    }
    // the phase turn from bin a to bin b, in radians
    let step = |a: usize, b: usize| {
        // H_b · conj(H_a)
        let (x, y) = (re[b] * re[a] + im[b] * im[a], im[b] * re[a] - re[b] * im[a]);
        y.atan2(x)
    };
    let ms_per_radian = 1000.0 / (std::f64::consts::TAU * bin_width);
    (0..len)
        .map(|k| {
            let (a, b) = (k.saturating_sub(1), (k + 1).min(len - 1));
            -step(a, b) / (b - a) as f64 * ms_per_radian
        })
        .collect()
}

/// The impulse response over time in ms since the start of the recording, so the delays
/// between speakers stay visible.
pub fn impulse_points(impulse_response: &TimeSignal, start: usize) -> Vec<(f64, f64)> {
    let ms_per_sample = 1000.0 / impulse_response.sample_rate();
    impulse_response
        .channel(0)
        .iter()
        .enumerate()
        .map(|(i, &s)| ((start + i) as f64 * ms_per_sample, s))
        .collect()
}

/// Where to draw the curves on the log axis: evenly spaced, `per_pixel` per pixel of the
/// visible range `visible` that is `width` pixels wide. Only within `log_range`, where there is
/// data.
pub fn draw_positions(visible: (f64, f64), width: f64, log_range: (f64, f64)) -> Vec<f64> {
    let (low, high) = log_range;
    let (left, right) = visible;
    if low >= high || left >= right || width < 1.0 {
        return Vec::new();
    }
    let start = low.max(left);
    let end = high.min(right);
    if start >= end {
        return Vec::new();
    }
    let step = (right - left) / width;
    let num_points = ((end - start) / step).ceil() as usize + 1;
    (0..num_points)
        .map(|i| (start + i as f64 * step).min(end))
        .collect()
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
pub fn split_at_wraps(points: &[(f64, f64)]) -> Vec<Vec<(f64, f64)>> {
    let mut pieces = vec![Vec::new()];
    for (index, &(x, y)) in points.iter().enumerate() {
        if let Some(&(last_x, last_y)) = index.checked_sub(1).map(|i| &points[i]) {
            let step = y - last_y;
            // a jump of more than half a turn is a wrap: the phase went the short way round
            if step.abs() > 180.0 {
                let unwrapped = y - 360.0 * step.signum();
                let edge = if unwrapped > last_y { 180.0 } else { -180.0 };
                let at = last_x + (x - last_x) * (edge - last_y) / (unwrapped - last_y);
                pieces.last_mut().unwrap().push((at, edge));
                pieces.push(vec![(at, -edge)]);
            }
        }
        pieces.last_mut().unwrap().push((x, y));
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_pieces_meet_at_the_edges() {
        let points: Vec<(f64, f64)> = (0..200)
            .map(|i| {
                let wrapped = (-7.0 * i as f64 + 180.0f64).rem_euclid(360.0) - 180.0;
                (i as f64, wrapped)
            })
            .collect();
        let pieces = split_at_wraps(&points);
        assert!(pieces.len() > 2);
        for pair in pieces.windows(2) {
            let (end, start) = (pair[0].last().unwrap(), pair[1][0]);
            assert_eq!((end.0, end.1), (start.0, -start.1));
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
        let curves = Curves::new(&impulse_response(), None, (20.0, 20_000.0), Some(3.0));
        assert!((curves.log_range.0 - 20f64.log10()).abs() < 1e-9);
        assert!((curves.log_range.1 - 20_000f64.log10()).abs() < 1e-9);
        // an impulse is flat at 0 dB
        for hz in [20.0, 31.4, 1000.0, 19_999.0] {
            assert!(curves.magnitude_db(hz).abs() < 0.1, "{hz}");
        }
        let (low, high) = curves.magnitude_extent(0.5);
        assert!(low > -0.1 && high < 0.1);
    }

    #[test]
    fn group_delay_of_a_delay_is_flat() {
        // the impulse lies 10 samples in
        let expected = 10.0 / 48.0;
        // smoothing bends it a little at the lowest bins, where the window holds only a few
        for (smoothing, tolerance) in [(None, 1e-9), (Some(6.0), 1e-3)] {
            let curves = Curves::new(&impulse_response(), None, (20.0, 20_000.0), smoothing);
            for hz in [20.0, 31.4, 1000.0, 12_345.0, 19_999.0] {
                let ms = curves.group_delay_ms(hz);
                assert!(
                    (ms - expected).abs() < tolerance,
                    "{smoothing:?} {hz}: {ms}"
                );
            }
            let (low, high) = curves.group_delay_extent(0.5);
            assert!((low - expected).abs() < tolerance && (high - expected).abs() < tolerance);
        }
    }

    #[test]
    fn group_delay_steps_need_no_unwrapping() {
        // 0.2 ms on bins 1 kHz apart: a fifth of a turn per bin, wrapping every few bins
        let (bins, delay) = (100, 0.2);
        // −2π · f · τ, with f = k · 1000 Hz and τ in s
        let angle = |k: usize| -std::f64::consts::TAU * (k as f64 * 1000.0) * (delay / 1000.0);
        let re: Vec<f64> = (0..bins).map(|k| angle(k).cos()).collect();
        let im: Vec<f64> = (0..bins).map(|k| angle(k).sin()).collect();
        for ms in group_delay_ms(&re, &im, 1000.0) {
            assert!((ms - delay).abs() < 1e-9, "{ms}");
        }
    }

    #[test]
    fn low_coherence_is_left_out() {
        let response = impulse_response();
        // coherent up to 1 kHz, noise above
        let freqs = response.freq_bins().to_owned();
        let mut coherence = response.clone();
        for (c, f) in coherence.channel_mut(0).iter_mut().zip(freqs) {
            *c = if f < 1000.0 { 1.0 } else { 0.1 }.into();
        }
        let curves = Curves::new(&response, Some(&coherence), (20.0, 20_000.0), None);
        assert!(curves.trusted(500.0, 0.5));
        assert!(!curves.trusted(5000.0, 0.5));
        // a limit of 0 shows everything
        assert!(curves.trusted(5000.0, 0.0));
        // and a sweep has no coherence: all of it is trusted
        let sweep = Curves::new(&response, None, (20.0, 20_000.0), None);
        assert!(sweep.trusted(5000.0, 0.5));
    }

    #[test]
    fn positions_follow_the_view_at_screen_resolution() {
        let range = (20f64.log10(), 20_000f64.log10());
        // zoomed in to 200..400 Hz on 600 pixels
        let visible = (200f64.log10(), 400f64.log10());
        let positions = draw_positions(visible, 600.0, range);
        let step = positions[1] - positions[0];
        assert!((step - (visible.1 - visible.0) / 600.0).abs() < 1e-12);
        assert_eq!(positions.len(), 601);

        // zoomed out beyond the data: only where there is data
        let positions = draw_positions((0.0, 6.0), 600.0, range);
        assert_eq!(positions[0], range.0);
        assert_eq!(*positions.last().unwrap(), range.1);
        assert!(positions.len() < 400);

        // nothing visible
        assert!(draw_positions((5.0, 6.0), 600.0, range).is_empty());
    }
}
