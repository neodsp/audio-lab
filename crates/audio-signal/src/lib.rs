pub use ndarray;

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
#[cfg(feature = "audio-io")]
pub mod playback;
pub mod signal;
pub mod test_signal;
