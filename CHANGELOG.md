# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `XAxis`, the x axis of `RealData` / `ComplexData`. Uniform axes store only
  start, step and length. `TimeSignal` and `FreqSignal` use uniform axes, so
  they no longer store an array of time steps or frequency bins.
- `RealData::from_axis` / `ComplexData::from_axis`, `x_axis()`, `set_x_axis()`,
  `TimeSignal::time_axis()` and `FreqSignal::freq_axis()`.

### Changed

- `x_data()`, `TimeSignal::time_steps()` and `FreqSignal::freq_bins()` return
  `CowArray<f64, Ix1>` instead of `ArrayView1<f64>`. They build the array for
  uniform axes; use `x_axis()` / `time_axis()` / `freq_axis()` to avoid that.
- `TimeSignal` / `FreqSignal` reject non-finite sample rates.

### Removed

- `RealData::x_data_mut()` / `ComplexData::x_data_mut()`. Use `set_x_axis()`.
- `FrequencyError::NonIncreasingFrequencyBins`, which could no longer occur.

[Unreleased]: https://github.com/neodsp/audio-lab/commits/main
