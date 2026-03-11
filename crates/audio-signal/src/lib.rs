pub use ndarray;

#[macro_export]
macro_rules! join {
    ($($signal:expr),+ $(,)?) => {
        $crate::signal::join_channels(&[$(&$signal),+])
    };
}

pub mod data;
pub mod io;
#[cfg(feature = "audio-io")]
pub mod playback;
pub mod signal;
pub mod test_signal;
