# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `signal::measure::Sequence`: plays signals one after another (or several outputs at once)
  while recording any number of inputs, driven block by block from any audio host. It is the
  core for sweep measurements with several speakers and microphones.
- `signal::audio_blocks`: re-export of `audio-blocks`, whose block types `Sequence` processes.

## [0.1.0] - 2026-11-04

### Added

- Initial release.

[Unreleased]: https://github.com/neodsp/audio-lab/compare/0.1.0...HEAD
[0.1.0]: https://github.com/neodsp/audio-lab/releases/tag/0.1.0
