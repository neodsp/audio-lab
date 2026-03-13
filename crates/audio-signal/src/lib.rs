pub use ndarray;

pub use blocks::{
    BlockAdapterError, signal_from_block, signal_from_blocks, signal_to_block, signal_to_blocks,
};
pub use signal::{
    FreqSignal, Spectrogram, SpectrogramNormalization, TimeSignal, join_freq_signals,
    join_time_signals, mix_freq_signals, mix_time_signals,
};
pub use test_signal::impulse::{ImpulseConfig, generate_impulse};
pub use test_signal::noise::{NoiseConfig, Spectrum, generate_noise};
pub use test_signal::pulsed_noise::{PulsedNoiseConfig, generate_pulsed_noise};
pub use test_signal::sine::{SineConfig, generate_sine};
pub use test_signal::sweep::{SweepConfig, SweepType, generate_sweep};

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

pub mod blocks;
pub mod data;
pub mod io;
pub mod math;
mod ops;
#[cfg(feature = "audio-playback")]
pub mod playback;
pub mod signal;
pub mod test_signal;
