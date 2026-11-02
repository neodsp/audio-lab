//! Signal types, test signal generators, file I/O and playback.

pub use ndarray;

pub use blocks::{
    BlockAdapterError, signal_from_block, signal_from_blocks, signal_to_block, signal_to_blocks,
};
pub use test_signal::impulse::{ImpulseConfig, ImpulseError, generate_impulse};
pub use test_signal::noise::{NoiseConfig, NoiseError, Spectrum, generate_noise};
pub use test_signal::pulsed_noise::{PulsedNoiseConfig, PulsedNoiseError, generate_pulsed_noise};
pub use test_signal::sine::{SineConfig, generate_sine};
pub use test_signal::sweep::{SweepConfig, SweepError, SweepType, generate_sweep};
pub use types::{
    FreqSignal, SignalError, Spectrogram, SpectrogramNormalization, TimeSignal, join_freq_signals,
    join_time_signals, mix_freq_signals, mix_time_signals, utils,
};

#[macro_export]
macro_rules! join_time {
    ($($signal:expr),+ $(,)?) => {
        $crate::signal::join_time_signals(&[$($signal),+])
    };
}

#[macro_export]
macro_rules! mix_time {
    ($($signal:expr),+ $(,)?) => {
        $crate::signal::mix_time_signals(&[$($signal),+])
    };
}

#[macro_export]
macro_rules! join_freq {
    ($($signal:expr),+ $(,)?) => {
        $crate::signal::join_freq_signals(&[$($signal),+])
    };
}

#[macro_export]
macro_rules! mix_freq {
    ($($signal:expr),+ $(,)?) => {
        $crate::signal::mix_freq_signals(&[$($signal),+])
    };
}

// `#[macro_export]` puts the macros at the crate root, this keeps them available here too
pub use crate::{join_freq, join_time, mix_freq, mix_time};

mod blocks;
pub mod data;
pub mod fs;
pub mod math;
mod ops;
#[cfg(feature = "playback")]
pub mod playback;
mod test_signal;
mod types;
