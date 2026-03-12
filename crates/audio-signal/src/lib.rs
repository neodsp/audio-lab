pub use ndarray;

pub use signal::{FreqSignal, Spectrogram, SpectrogramNormalization, TimeSignal};
pub use test_signal::impulse::{ImpulseConfig, generate_impulse};
pub use test_signal::noise::{NoiseConfig, Spectrum, generate_noise};
pub use test_signal::pulsed_noise::{PulsedNoiseConfig, generate_pulsed_noise};
pub use test_signal::sine::{SineConfig, generate_sine};
pub use test_signal::sweep::{SweepConfig, SweepType, generate_sweep};

#[macro_export]
macro_rules! join {
    ($($signal:expr),+ $(,)?) => {
        $crate::signal::join_signals(&[$($signal.into_time()),+])
    };
}

pub mod data;
pub mod io;
pub mod math;
mod ops;
#[cfg(feature = "audio-playback")]
pub mod playback;
pub mod signal;
pub mod test_signal;
