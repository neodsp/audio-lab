# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `signal::measure::Sequence`: plays signals one after another (or several outputs at once)
  while recording any number of inputs, driven block by block from any audio host. It is the
  core for sweep measurements with several speakers and microphones.
- `signal::playback::play_and_record`: runs a `Sequence` on an audio device in one duplex
  stream, e.g. a sweep per speaker while several mics record. Repeats the measurement on
  dropouts and ignores those while the device starts up, but never returns a recording with
  gaps. `DeviceConfig` selects the devices, stream settings, settle time and attempts.
- Live measurements: `signal::measure::live_channel` records inputs next to a reference signal
  without locks on the audio thread, and `signal::playback::LiveSession` runs it on an audio
  device. `signal::measure::NoiseSource` plays looped pink (or white) noise on several outputs
  as the source, with gain and on/off that can change while it plays.
- `dsp::transfer_function`: the transfer function from a reference to each channel, averaged
  over segments. Exact for periodic excitation such as `NoiseSource`. `dsp::coherence` tells
  per frequency how reliable it is.
- Examples `measure_impulse_response` (sweeps, several speakers and mics) and `live_noise`
  (live transfer function with pink noise and an adjustable delay). The README's impulse
  response example now records a real measurement.
- A measurement app in `apps/measurement`, written with GPUI (`cargo run --release -p
  measurement-app`): a side bar of all measurements, grouped by run, a setup panel for devices,
  speakers, mics and the sweep, and tabs for magnitude, phase and impulse responses. A simulated
  device lets you try it without hardware.
- `dsp::energy_decay_curve`: the Schroeder energy decay curve of an impulse response. It stops
  where the decay meets the noise floor, so the noise doesn't flatten the end of the curve.
- `FractionalOctaveSmoothingMode::Complex`: smooths the real and imaginary parts, so the phase
  is smoothed too.
- `signal::audio_blocks`: re-export of `audio-blocks`, whose block types `Sequence` processes.
- `plot::freq::FreqPlotOptions::freq_range` limits a frequency plot to a range.
- `plot::show_magnitude_and_phase`: magnitude and phase in one window, phase on top, zooming
  together, the phase wrapped to ±180°. `measure_impulse_response` uses it.
- `plot::freq::FreqPlotOptions::smoothing` adds adjustable fractional-octave smoothing to a
  frequency plot window.

### Changed

- Frequency plots on a log axis draw magnitudes at points evenly spaced on the axis, with an
  interpolation that doesn't overshoot, so the low end looks as smooth as the top. They start
  zoomed to the data instead of adding margins.
- Frequency plots on a log axis draw their curves for the part in view, at the resolution of
  the screen, so they stay smooth however far you zoom in. The phase is interpolated too, and
  `show_freq_with_options` draws a wrapped phase without lines across the plot at its wraps.

## [0.1.0] - 2026-11-04

### Added

- Initial release.

[Unreleased]: https://github.com/neodsp/audio-lab/compare/0.1.0...HEAD
[0.1.0]: https://github.com/neodsp/audio-lab/releases/tag/0.1.0
