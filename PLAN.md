# Plan

Planned features for `audio-lab`, in order. This file isn't published to crates.io: the
`include` list in `Cargo.toml` only packages `src/`, `examples/`, `README.md`,
`CHANGELOG.md` and `LICENSE`.

## 1. Play and record (next)

**Goal:** measure a real system response from inside the crate. Everything around the
measurement exists (sweeps, `deconvolve`, `find_impulse_response_start`,
`window_and_trim`, smoothing, plots), but the README's "Estimating an impulse response"
example still fakes the recording with `excitation.clone()`.

### API

```rust
// `playback` feature
pub fn play_and_record(
    excitation: &TimeSignal,
    config: &RecordConfig,
) -> Result<TimeSignal, RecordError>;

pub struct RecordConfig {
    pub num_input_channels: u16,
    /// Extra recording time after the excitation ends, for reverb tails and device latency.
    pub tail: Duration,
    pub input_device: Option<String>,   // `None` = system default
    pub output_device: Option<String>,
    /// Record an output channel wired back to an input, to remove device latency exactly.
    pub loopback: Option<Loopback>,
}
```

### Design

- **Testable core.** A `Measurement` type holds the excitation and a preallocated recording
  buffer, and is driven block by block: `process(input, output)` writes the next excitation
  block and appends the input; `is_done()`; `into_recording()`. It does no I/O and doesn't
  allocate while processing. Tests drive it with a simulated device (a delay plus a known
  filter), so CI covers it without audio hardware.
- **Device wrapper.** `play_and_record` runs a `Measurement` inside an `audio-host` duplex
  stream (`Config::with_input_channels`, `set_input` / `set_output` for device names), waits
  until it's done, stops the stream and returns the recording. Like `play`, its hardware
  test is `#[ignore]`d.
- **Dropouts.** If `status().xruns` is non-zero after the run, return an error instead of a
  recording with gaps.
- **Sample rate.** The stream runs at the excitation's sample rate. If the device can't, return
  the backend's error; don't resample silently.
- **Latency.** The cpal backend adds `duplex_latency_ms` (100 ms by default) of input latency.
  The tail must be longer than that, so `RecordConfig::tail` defaults to 1 s, and the docs say
  why. With a loopback channel, deconvolving against the recorded reference instead of the
  excitation removes the latency exactly.

### Steps (one stacked PR each)

1. `Measurement` core, plus simulated-device tests (delay, filter, block sizes that don't divide
   the signal length, multichannel).
2. `play_and_record` on `audio-host`: device selection, xrun check, ignored hardware test, an
   example `measure_impulse_response.rs`.
3. Loopback reference: record it as an extra channel and offer
   `deconvolve` against it; test that a simulated latency cancels out.
4. README: replace the faked recording in "Estimating an impulse response" with a real
   measurement (doc test compiled but not run).

### Open questions

- Should `play_and_record` report the recorded peak level, or warn when it clips (|x| ≥ 1)?
- Device lookup by exact name only, or also by substring?

## 2. Room acoustics from an impulse response

**Goal:** the standard ISO 3382-1 parameters from a measured impulse response, overall and per
octave or third-octave band (via `OctaveBands`).

- Schroeder backward integration → energy decay curve.
- Reverberation time T20 / T30, early decay time (EDT), clarity C50 / C80, definition D50.
- Noise floor: truncate the integration where the decay meets the noise floor (start with a fixed
  margin above the estimated floor; Lundeby's iterative method as a follow-up).
- Stays measurement only, in line with the `dsp` module boundary (no correction or target
  curves).
- Tests: synthetic exponentially decaying noise with a known T60, with and without added noise
  floor.

## 3. Inverse STFT

**Goal:** turn a `Spectrogram` back into a `TimeSignal`, for spectral editing and STFT round-trip
tests.

- Weighted overlap-add, using the `window_fn`, `window_size` and `hop_size` a `Spectrogram`
  already stores.
- Periodic windows (used by `stft` since #23) make Hann at 50 % / 75 % overlap sum to a constant;
  divide by the summed window² so other overlaps work too.
- Tests: `istft(stft(x)) ≈ x` (away from the edges) for every window and several hops.

## Conventions

- Dependent work goes in GitHub stacked PRs (`gh stack`); independent changes get separate PRs.
- `CHANGELOG.md`: 0.1.0 is released, so every user-visible change gets an entry under
  `[Unreleased]`.
- Before every PR: `cargo fmt --all`, `cargo clippy --all-targets -D warnings` with
  `--all-features` and `--no-default-features`, `cargo test --all-features`,
  `cargo test --no-default-features --all-targets`, `taplo fmt --check`, `cargo shear`.
