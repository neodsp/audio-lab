//! Generic DSP and measurement primitives built on top of `audio-signal`.
//!
//! Boundary:
//! - `audio-signal` owns signal containers, I/O, and synthetic test fixtures.
//! - `audio-dsp` owns reusable DSP building blocks such as padding, windowing,
//!   transforms, and filter-bank design.
//! - Room-correction-specific policy, target-curve logic, and correction filter
//!   design workflows do not belong in this crate.

pub mod convolve;
pub mod deconvolution;
pub mod filter_bank;
pub mod fractional_octave;
pub mod frequency;
pub mod pad;
pub mod stft;
pub mod time;
pub mod window;

pub use convolve::{ConvolveError, ConvolveMode, convolve};
pub use deconvolution::{DeconvolutionConfig, DeconvolutionError, deconvolve};
pub use fractional_octave::{
    FractionalOctaveSmoothingConfig, FractionalOctaveSmoothingError, FractionalOctaveSmoothingMode,
    FractionalOctaveSmoothingStats, smooth_fractional_octave,
};
pub use frequency::{FrequencyError, group_delay};
pub use pad::{PadMode, pad_zeros};
pub use stft::{StftConfig, StftError, stft};
pub use time::{
    TimeError, TimeShiftMode, apply_gain, apply_gain_db, find_impulse_response_start, resample,
    time_shift, time_shift_per_channel, trim_duration, trim_samples, window_and_trim,
};
pub use window::{WindowFn, apply_hann, apply_hann_left, apply_hann_right, generate_window};
