pub mod data;
pub mod freq;
pub mod time;

pub use data::show_real_data;
pub use freq::show_freq_signal;
pub use time::show_time_signal;

/// Return type of all `show_*` functions.
pub type Result = eframe::Result;

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
