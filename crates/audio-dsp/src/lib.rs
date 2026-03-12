//! Generic DSP and measurement primitives built on top of `audio-signal`.
//!
//! Boundary:
//! - `audio-signal` owns signal containers, I/O, and synthetic test fixtures.
//! - `audio-dsp` owns reusable DSP building blocks such as padding, windowing,
//!   transforms, and filter-bank design.
//! - Room-correction-specific policy, target-curve logic, and correction filter
//!   design workflows do not belong in this crate.

pub mod convolve;
pub mod filter_bank;
pub mod pad;
pub mod stft;
pub mod time;
pub mod window;

pub use convolve::{ConvolveError, ConvolveMode, convolve};
pub use pad::{PadMode, pad_zeros};
pub use stft::{StftConfig, StftError, stft};
pub use time::{TimeError, trim_duration, trim_samples};
pub use window::{WindowFn, apply_hann, apply_hann_left, apply_hann_right, generate_window};
