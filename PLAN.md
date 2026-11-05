# Plan

Planned features for `audio-lab`, in order. This file isn't published to crates.io: the
`include` list in `Cargo.toml` only packages `src/`, `examples/`, `README.md`,
`CHANGELOG.md` and `LICENSE`.

## 1. Play and record (next)

**Goal:** measure real systems from inside the crate, with several microphones at once:

- **Sweep measurements:** play a sweep on each speaker, one after another, while every mic
  records. The result is one impulse response per speaker and mic.
- **Realtime noise measurements:** play noise on several speakers at the same time and read the
  mics continuously, e.g. for a live spectrum while tuning a crossover or a sub delay.

Everything around the measurement exists (sweeps, noise, `deconvolve`,
`find_impulse_response_start`, `window_and_trim`, smoothing, plots), but the README's
"Estimating an impulse response" example still fakes the recording with `excitation.clone()`.
The design follows neodrc's `measurement.rs` and `realtime_spectrum.rs`, without their sample
loss (`try_lock` on the audio thread) and with tests that don't need audio hardware.

### API

```rust
// `signal::measure`, always available (no audio device needed)
pub struct Step {
    /// Output channels to play on, one per channel of `signal`.
    pub outputs: Vec<u16>,
    pub signal: TimeSignal,
}

pub struct SequenceConfig {
    /// Input channels to record, one per mic. All of them record during every step.
    pub inputs: Vec<u16>,
    /// Silence before the first step, not part of any recording. Lets the device settle.
    pub warmup: Duration,
    /// Recorded after each step (silence on the outputs), for reverb tails and device latency.
    pub tail: Duration,
}

/// Plays steps one after another on one timeline and records the inputs. Driven block by
/// block, does no I/O and doesn't allocate while processing.
pub struct Sequence { .. }
impl Sequence {
    pub fn new(steps: Vec<Step>, config: &SequenceConfig) -> Result<Self, RecordError>;
    pub fn process(&mut self, input: &impl AudioBlock<f32>, output: &mut impl AudioBlockMut<f32>);
    pub fn is_done(&self) -> bool;
    /// One `TimeSignal` per step, one channel per input, `step length + tail` long.
    pub fn into_recordings(self) -> Vec<TimeSignal>;
}

// `signal::playback`, `playback` feature
pub fn play_and_record(
    steps: Vec<Step>,
    config: &SequenceConfig,
    device: &DeviceConfig,
) -> Result<Vec<TimeSignal>, RecordError>;

pub struct DeviceConfig {
    pub input_device: Option<String>,  // `None` = system default
    pub output_device: Option<String>,
    pub num_frames: usize,
    pub duplex_latency_ms: u32,
}

/// Non-blocking: plays `source` and records the inputs until stopped.
pub struct LiveSession { .. }
impl LiveSession {
    pub fn start(
        source: impl FnMut(&mut InterleavedViewMut<f32>) + Send + 'static,
        inputs: Vec<u16>,
        sample_rate: f64,
        device: &DeviceConfig,
    ) -> Result<Self, RecordError>;
    /// The most recent `num_samples` of every input, one channel per input.
    pub fn latest(&mut self, num_samples: usize) -> TimeSignal;
    pub fn status(&self) -> StreamStatus;
    pub fn stop(self) -> Result<(), RecordError>;
}

/// A ready-made `source`: white or pink noise on some outputs, with gain and on/off that can
/// change while it plays. Wrap it in your own closure to add a crossover or delay.
pub struct NoiseSource { .. }
```

- Several speakers one after another: one `Step` per speaker. Several speakers at once: one
  `Step` with several outputs.
- `Sequence` is public, so hosts other than `audio-host` (e.g. a PipeWire app) can drive it.

### Design

- **One stream.** All steps run in one duplex stream, so the device latency is the same for
  every speaker. Restarting the stream per speaker lets the audio server pick a new latency,
  which breaks the delay alignment between speakers.
- **Testable core.** Tests drive `Sequence` with a simulated device (a delay plus a known
  filter), so CI covers it without audio hardware. The device wrappers' hardware tests are
  `#[ignore]`d, like `play`'s.
- **Dropouts.** `play_and_record` returns an error if `status().xruns` is non-zero after the run,
  instead of a recording with gaps. `LiveSession` reports them through `status()`.
- **No lost samples in live mode.** The audio thread pushes the inputs into a lock-free queue;
  `latest` drains it into a ring buffer on the caller's thread.
- **Sample rate.** All steps must have the same sample rate, and the stream runs at it. If the
  device can't, return the backend's error; don't resample silently.
- **Latency.** The cpal backend adds `duplex_latency_ms` (100 ms by default) of input latency.
  `SequenceConfig::tail` defaults to 1 s so it covers that, and the docs say why. For exact
  latency, record a loopback (an output wired back to an input) as one more input and deconvolve
  against it.

### Steps (one stacked PR each)

1. `Sequence` core, plus simulated-device tests (delay, filter, block sizes that don't divide
   the signal length, several mics, several speakers in one step).
2. `play_and_record` on `audio-host`: device selection, xrun check, ignored hardware test.
3. `LiveSession` and `NoiseSource`, plus tests for the input ring buffer.
4. Examples `measure_impulse_response.rs` and `live_noise.rs`. README: replace the faked recording
   in "Estimating an impulse response" with a real measurement (doc test compiled but not run).

Moving neodrc to this API is separate work there: it still depends on the old split crates.

### Open questions

- Should `play_and_record` report the recorded peak level, or warn when it clips (|x| ≥ 1)?

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
