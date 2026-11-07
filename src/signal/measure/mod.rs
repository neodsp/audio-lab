//! Play signals and record inputs at the same time, for measurements with several microphones.
//!
//! - [`Sequence`] plays [`Step`]s one after another on one timeline and records every input
//!   during each of them, e.g. a sweep per speaker while all mics record.
//! - [`live_channel`] records inputs continuously next to a reference signal, e.g. the
//!   [`NoiseSource`] playing on several speakers at once, for live transfer functions with
//!   [`transfer_function`](fn@crate::dsp::transfer_function).
//!
//! Both are driven block by block from an audio callback, do no I/O and don't allocate while
//! processing, so any audio host can run them. `signal::playback` runs them on `audio-host`.

pub use live::{LiveConfig, LiveReader, LiveSnapshot, LiveWriter, live_channel};
pub use noise::{NoiseControls, NoiseSource, NoiseSourceConfig, NoiseSourceError};
pub use sequence::{RecordError, Sequence, SequenceConfig, Step};

mod live;
mod noise;
mod sequence;
#[cfg(test)]
mod sim;
