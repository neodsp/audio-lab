#[cfg(feature = "audio-file")]
pub mod audio;
#[cfg(feature = "numpy")]
pub mod npy;

#[cfg(feature = "audio-file")]
#[deprecated(note = "use io::audio")]
pub mod audio_file {
    pub use super::audio::*;
}

#[cfg(feature = "numpy")]
#[deprecated(note = "use io::npy")]
pub mod numpy {
    pub use super::npy::*;
}
