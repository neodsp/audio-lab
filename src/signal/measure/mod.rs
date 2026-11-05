//! Play signals and record inputs at the same time, for measurements with several microphones.
//!
//! [`Sequence`] plays [`Step`]s one after another on one timeline and records every input
//! during each of them, e.g. a sweep per speaker while all mics record. It is driven block by
//! block from an audio callback, does no I/O and doesn't allocate while processing, so any
//! audio host can run it.

pub use sequence::{RecordError, Sequence, SequenceConfig, Step};

mod sequence;
#[cfg(test)]
mod sim;
