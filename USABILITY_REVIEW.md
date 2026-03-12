# Usability Review

Review of the `audio-lab` crate APIs and structure from the perspective of an audio researcher.

## What works well

- **Time/Freq domain toggling** via `signal.into_freq()` / `signal.into_time()` is very intuitive — feels like MATLAB.
- **Channel-oriented iteration** (`channel_iter`, `channel_iter_mut`) is the right abstraction for multi-channel audio.
- **The 3-crate split** (signal / dsp / plot) is clean — you don't pull in egui just to do DSP.
- **Test signal generators** are a nice inclusion — saves researchers from writing boilerplate every time.
- **`StftConfig` with sensible defaults** (Hann, 75% overlap) is researcher-friendly.

## Issues, roughly by severity

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
