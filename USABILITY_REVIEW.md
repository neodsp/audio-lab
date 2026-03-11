# Usability Review

Review of the `audio-lab` crate APIs and structure from the perspective of an audio researcher.

## What works well

- **Time/Freq domain toggling** via `signal.into_freq()` / `signal.into_time()` is very intuitive — feels like MATLAB.
- **Channel-oriented iteration** (`channel_iter`, `channel_iter_mut`) is the right abstraction for multi-channel audio.
- **The 3-crate split** (signal / dsp / plot) is clean — you don't pull in egui just to do DSP.
- **Test signal generators** are a nice inclusion — saves researchers from writing boilerplate every time.
- **`StftConfig` with sensible defaults** (Hann, 75% overlap) is researcher-friendly.

## Issues, roughly by severity

### 1. The `ops` module is the biggest usability problem

An audio researcher would naturally expect to write:

```rust
let mag = freq_signal.to_magnitude();
let peak = signal.max_abs_overall();
let db = data.to_decibels();
```

Instead they must write:

```rust
use audio_signal::ops::complex;
use audio_signal::ops::real;
use audio_signal::ops::freq;

let mag = complex::to_magnitude(freq_signal.data());
let peak = real::max_abs_overall(signal.data());  // wait, this takes RealData not TimeSignal
let nyquist = freq::nyquist_frequency(&freq_signal);
```

Problems:
- **Free functions that conceptually belong on their type.** `nyquist_frequency(&signal)` and `freq_spacing(&signal)` are just `sample_rate / 2.0` and `sample_rate / num_time_steps` — these should be methods on `FreqSignal`.
- **`ops::real` operates on `RealData`, not `TimeSignal`.** So you have to go through `signal.data()` to use them. But a researcher thinks in terms of signals, not data containers.
- **`ops::complex` operates on `ComplexData`, not `FreqSignal`.** Same issue — you need `freq_signal.data()`.
- The `analysis::spectrogram` module has the same pattern: all free functions taking `&Spectrogram`.

**Suggestion:** Make these methods. `freq_signal.nyquist()`, `freq_signal.to_magnitude()`, `signal.peak()`, `spectrogram.amplitude_spectrum()`. The free functions can stay as the implementation, but put method wrappers on the types where researchers will look for them.

### 2. `RealData` / `ComplexData` — unclear value for the end user

As a researcher, I'd never want to construct or think about `RealData` directly. I'd think about `TimeSignal` (samples + sample rate) or `FreqSignal` (spectrum + sample rate). The `data` module feels like an internal implementation detail that leaked into the public API.

`TimeSignal` is basically `RealData` + `sample_rate`. The `x_data` / `y_data` naming on `RealData` is very generic — it doesn't tell me what the axes mean. Compare with `TimeSignal` which has `time_steps()` and `time_data()` — much clearer.

**Suggestion:** Consider making `data` module `pub(crate)` or at least de-emphasizing it in documentation. Most researchers should only interact with `TimeSignal`, `FreqSignal`, and `Spectrogram`.

### 3. test_signal generators have too many positional parameters

```rust
generate_sine(num_time_steps, frequency, amplitude, sample_rate, num_channels)
generate_noise(num_time_steps, spectrum, amplitude, sample_rate, num_channels, seed)
generate_sweep(num_time_steps, freq_range, amplitude, sample_rate, num_channels, fade_out, sweep_type)
generate_pulsed_noise(pulse_length, pause_length, fade_length, repetitions, spectrum, amplitude, frozen, sample_rate, num_channels, seed)
```

`generate_pulsed_noise` has **10 positional arguments**. At the call site you can't tell what's what. A builder pattern or config struct (like you already did with `StftConfig`) would be much better:

```rust
Sine::new(frequency, sample_rate)
    .amplitude(0.5)
    .channels(2)
    .samples(48000)
    .build()?
```

### 4. `audio-dsp::time` has normalize functions that duplicate each other

You have 6 functions that are really 2:
- `normalize_peak_linear` / `normalize_peak_in_place_linear`
- `normalize_peak_db` / `normalize_peak_in_place_db`
- `normalize` / `normalize_in_place` (aliases for the linear versions)

Plus `audio-signal::test_signal::noise` has its own `normalize_peak()`. A researcher will be confused about which to use. This should be one method: `signal.normalize(peak_level)` with the dB variant being `signal.normalize_db(peak_db)`.

### 5. Module path ergonomics — too much nesting for common operations

A typical research script has to juggle:
```rust
use audio_signal::signal::{TimeSignal, FreqSignal};
use audio_signal::test_signal::sine::generate_sine;
use audio_signal::ops::complex;
use audio_signal::analysis::spectrogram as spec_analysis;
use audio_dsp::stft::{stft, StftConfig};
use audio_dsp::time::normalize;
use audio_plot::show_time_signal;
```

That's 7 use-statements before writing any actual code. Consider flattening re-exports in `lib.rs`:

```rust
// audio_signal prelude
pub use signal::{TimeSignal, FreqSignal, Spectrogram};
pub use test_signal::sine::generate_sine;
// etc.

// or provide a prelude module
pub mod prelude { ... }
```

### 6. `into_time()` / `into_freq()` are identity functions on the same type

`TimeSignal::into_time()` returns `self`. `FreqSignal::into_freq()` returns `self`. These exist so you can call `.into_time()` on either type, but they consume the value for no reason on the "same domain" case. A researcher calling `time_signal.into_time()` is likely confused — it looks like it should do something. If the intent is a uniform interface, a trait would be cleaner.

### 7. Inconsistent ownership patterns in DSP functions

- `audio_dsp::time::normalize_peak_linear` takes `&TimeSignal` and returns a new one (clone + modify).
- `audio_dsp::time::normalize_peak_in_place_linear` takes `&mut TimeSignal`.
- `audio_dsp::stft::stft` takes `&TimeSignal`.
- `audio_dsp::pad::pad_zeros` takes `&TimeSignal` and returns a new one.

But `TimeSignal::into_freq()` *consumes* self. So some operations clone, some consume. From a researcher's perspective this is confusing — they just want to process data and not think about ownership. Consistency would help: either everything borrows, or everything consumes and clones internally. Borrowing + returning new is the most researcher-friendly (matches NumPy/MATLAB mental model).

### 8. The `analysis::spectrogram` module lives in `audio-signal`, but `stft` lives in `audio-dsp`

So the thing that *creates* spectrograms is in a different crate than the thing that *analyzes* them. A researcher looking at `Spectrogram` will expect to find analysis methods nearby. Consider either:
- Moving analysis into `audio-dsp` next to `stft`, or
- Making them methods on `Spectrogram` directly

### 9. Minor: `filter_bank.rs` has its own `hanning_window()` while `window.rs` exists

`audio-dsp` has a full `window` module with `generate_window(WindowFn::Hann, n)`, but `filter_bank.rs` defines a private `hanning_window()`. This should just call the existing one.

## Summary of recommendations

| Priority | Change |
|----------|--------|
| High | Make `ops::*` and `analysis::spectrogram` functions into methods on their respective types |
| High | Add a prelude or flatten re-exports to reduce use-statement ceremony |
| Medium | Builder pattern for test signal generators (especially `pulsed_noise`) |
| Medium | Consolidate normalize variants — one method, not six free functions |
| Medium | Move `analysis::spectrogram` next to `stft` or make them methods on `Spectrogram` |
| Low | De-emphasize `RealData`/`ComplexData` from public API |
| Low | Remove identity `into_time()`/`into_freq()` or replace with a trait |
| Low | Deduplicate `hanning_window` in filter_bank |

The crate has solid foundations — the domain modeling (time/freq signals, spectrograms, filter banks) is correct and the DSP math is sound. The main friction is that the API is organized around Rust module boundaries rather than around how an audio researcher thinks about their workflow.
