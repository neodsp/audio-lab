//! Audio signal plotting and visualization.
//!
//! # Features
//!
//! The default `save-plot` feature enables the **Save Plot** button and PNG export
//! using the optional `image` and `rfd` dependencies. Disable it with:
//!
//! ```toml
//! audio-plot = { version = "0.1", default-features = false }
//! ```
//!
//! Plot titles remain visible when saving is disabled. Other dependencies may
//! still pull in `image` transitively. Cargo features are additive, so saving
//! remains enabled if another dependency enables `audio-plot/save-plot`.

pub mod data;
pub mod freq;
mod legend;
mod save;
pub mod spectrogram;
pub mod time;

pub use data::show_real_data;
pub use freq::show_freq;
pub use spectrogram::{SpectrogramPlotOptions, show_spectrogram};
pub use time::{TimePlotOptions, TimeValue, show_time};

/// Error returned by all `show_*` functions.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct Error(#[from] eframe::Error);

pub(crate) fn native_options_any_thread() -> eframe::NativeOptions {
    eframe::NativeOptions {
        event_loop_builder: Some(Box::new(|builder| {
            #[cfg(target_os = "linux")]
            if std::env::var("WAYLAND_DISPLAY").is_ok() {
                use winit::platform::wayland::EventLoopBuilderExtWayland;
                builder.with_any_thread(true);
            } else {
                use winit::platform::x11::EventLoopBuilderExtX11;
                builder.with_any_thread(true);
            }
            #[cfg(target_os = "windows")]
            {
                use winit::platform::windows::EventLoopBuilderExtWindows;
                builder.with_any_thread(true);
            }
        })),
        ..Default::default()
    }
}
