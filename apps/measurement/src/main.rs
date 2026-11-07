//! A measurement app: measure impulse and frequency responses with sweeps and compare them.
//!
//! The side bar lists all measurements. Those taken together, one per speaker and mic, are
//! grouped in a folder that can be expanded, hidden or deleted as a whole. Tick a measurement
//! to show it, click its color to change it, right-click it for more.
//!
//! "New measurement" (Ctrl+N) opens the setup in the side bar: choose the devices, the speakers
//! (outputs) and microphones (inputs), the sweep, and start. Each speaker plays a sweep, one
//! after another, while all mics record.
//!
//! The tabs (Alt+1, 2, 3) show the magnitude, the phase and the impulse responses. The phase is
//! relative to each impulse response's peak, so the delays are taken out; the impulse
//! responses keep them, on a time axis from the start of the recording.
//!
//! "Simulated device" in the setup measures a made-up room instead, to try the app without
//! audio hardware.
//!
//! ```sh
//! cargo run --release -p measurement-app
//! ```

mod app;
mod curves;
mod devices;
mod live;
mod measure;
mod panes;
mod plot;
mod setup;
mod theme;

use std::borrow::Cow;

use gpui_kit::component::TitleBar;
use gpui_kit::*;

gpui_kit::assets::icon_assets!(
    ExtraIcons,
    [
        AudioWaveform,
        Camera,
        Radio,
        Crosshair,
        FolderInput,
        FolderPlus,
        CalendarClock
    ]
);

/// The component icons and the few extra ones the app uses.
struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        if let Some(bytes) = ExtraIcons.load(path)? {
            return Ok(Some(bytes));
        }
        gpui_kit::assets::Assets.load(path)
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        let mut paths = gpui_kit::assets::Assets.list(path)?;
        paths.extend(ExtraIcons.list(path)?);
        paths.sort();
        paths.dedup();
        Ok(paths)
    }
}

fn main() {
    gpui_kit::application().with_assets(Assets).run(|cx| {
        gpui_kit::init(cx);
        theme::init(cx);
        app::bind_keys(cx);
        cx.on_window_closed(|cx, _| cx.quit()).detach();

        let bounds = Bounds::centered(None, size(px(1280.0), px(800.0)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            window_min_size: Some(size(px(800.0), px(500.0))),
            app_id: Some("audio-lab-measurements".into()),
            ..TitleBar::window_options()
        };
        gpui_kit::open_window(options, cx, |window, cx| {
            cx.new(|cx| app::MeasurementApp::new(window, cx))
        })
        .expect("failed to open the window");
        cx.activate(true);
    });
}
