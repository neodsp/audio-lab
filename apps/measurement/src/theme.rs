//! A dark theme in the colors of Zed's One Dark, for the components and for what the app
//! draws itself.

use gpui_kit::component::{Theme, ThemeRegistry};
use gpui_kit::{App, Hsla, SharedString, hsla, rgb, rgba};

const THEME_NAME: &str = "Measurement Dark";

/// The colors the app draws with directly.
pub mod palette {
    use super::*;

    /// Behind the plots, like Zed's editor.
    pub fn editor() -> Hsla {
        rgb(0x282c33).into()
    }
    /// Side bar and tab bar.
    pub fn panel() -> Hsla {
        rgb(0x2f343e).into()
    }
    /// Title bar and status bar.
    pub fn chrome() -> Hsla {
        rgb(0x3b414d).into()
    }
    /// Popups and the readout under the mouse.
    pub fn elevated() -> Hsla {
        rgb(0x2f343e).into()
    }
    pub fn hover() -> Hsla {
        rgb(0x363c46).into()
    }
    pub fn border() -> Hsla {
        rgb(0x464b57).into()
    }
    pub fn border_variant() -> Hsla {
        rgb(0x363c46).into()
    }
    pub fn text() -> Hsla {
        rgb(0xdce0e5).into()
    }
    pub fn text_muted() -> Hsla {
        rgb(0xa9afbc).into()
    }
    pub fn text_placeholder() -> Hsla {
        rgb(0x878a98).into()
    }
    pub fn accent() -> Hsla {
        rgb(0x74ade8).into()
    }
    pub fn warning() -> Hsla {
        rgb(0xdec184).into()
    }
    pub fn error() -> Hsla {
        rgb(0xd07277).into()
    }
    pub fn grid() -> Hsla {
        rgba(0xc8ccd40d).into()
    }
    pub fn grid_strong() -> Hsla {
        rgba(0xc8ccd424).into()
    }
}

/// The `index`th of a series of well distinguishable curve colors: Zed's player colors, then
/// hues around the color wheel.
pub fn trace_color(index: usize) -> Hsla {
    const PLAYERS: [u32; 8] = [
        0x74ade8, 0xbf956a, 0xb477cf, 0x6eb4bf, 0xd07277, 0xdec184, 0xa1c181, 0xbe5046,
    ];
    match PLAYERS.get(index) {
        Some(&color) => rgb(color).into(),
        None => {
            let golden_ratio = (5f32.sqrt() - 1.0) / 2.0;
            hsla((index as f32 * golden_ratio).fract(), 0.6, 0.66, 1.0)
        }
    }
}

/// Installs the theme for the components.
pub fn init(cx: &mut App) {
    let registry = ThemeRegistry::global_mut(cx);
    registry
        .load_themes_from_str(THEME)
        .expect("the theme is valid");
    let config = registry
        .themes()
        .get(&SharedString::from(THEME_NAME))
        .cloned()
        .expect("the theme was just loaded");
    Theme::update(cx, |theme| {
        theme.apply_config(&config);
        theme.shadow = false;
    });
}

const THEME: &str = r##"{
  "name": "Measurement",
  "author": "audio-lab",
  "themes": [
    {
      "name": "Measurement Dark",
      "mode": "dark",
      "font.size": 14,
      "radius": 4,
      "radius.lg": 6,
      "shadow": false,
      "colors": {
        "accent.background": "#363c46",
        "accent.foreground": "#dce0e5",
        "accordion.background": "#2f343e",
        "background": "#2f343e",
        "base.blue": "#74ade8",
        "base.blue.light": "#a3c8f0",
        "base.cyan": "#6eb4bf",
        "base.cyan.light": "#9ccdd5",
        "base.green": "#a1c181",
        "base.green.light": "#bfd5a8",
        "base.magenta": "#b477cf",
        "base.magenta.light": "#cba2de",
        "base.red": "#d07277",
        "base.red.light": "#e09fa2",
        "base.yellow": "#dec184",
        "base.yellow.light": "#e9d4a9",
        "border": "#363c46",
        "caret": "#74ade8",
        "danger.background": "#d07277",
        "danger.foreground": "#d07277",
        "drag_border": "#74ade8",
        "drop_target.background": "#74ade833",
        "foreground": "#dce0e5",
        "group_box.background": "#2e343e",
        "group_box.foreground": "#dce0e5",
        "info.background": "#6eb4bf",
        "info.foreground": "#6eb4bf",
        "input.border": "#464b57",
        "link.active.foreground": "#a3c8f0",
        "link.foreground": "#74ade8",
        "link.hover.foreground": "#a3c8f0",
        "list.active.background": "#293b5b",
        "list.active.border": "#47679e",
        "list.background": "#2f343e",
        "list.even.background": "#2f343e",
        "list.head.background": "#2f343e",
        "list.hover.background": "#363c46",
        "muted.background": "#2e343e",
        "muted.foreground": "#a9afbc",
        "overlay": "#00000040",
        "popover.background": "#2f343e",
        "popover.foreground": "#dce0e5",
        "primary.active.background": "#5d9ee3",
        "primary.background": "#74ade8",
        "primary.foreground": "#282c33",
        "primary.hover.background": "#8cbcec",
        "progress_bar.background": "#74ade8",
        "ring": "#47679e",
        "scrollbar.background": "#00000000",
        "scrollbar.thumb.background": "#c8ccd44c",
        "scrollbar.thumb.hover.background": "#c8ccd480",
        "secondary.active.background": "#454a56",
        "secondary.background": "#2e343e",
        "secondary.foreground": "#dce0e5",
        "secondary.hover.background": "#363c46",
        "selection.background": "#74ade83d",
        "sidebar.accent.background": "#363c46",
        "sidebar.accent.foreground": "#dce0e5",
        "sidebar.background": "#2f343e",
        "sidebar.border": "#363c46",
        "sidebar.foreground": "#dce0e5",
        "sidebar.primary.background": "#74ade8",
        "sidebar.primary.foreground": "#282c33",
        "skeleton.background": "#2e343e",
        "slider.bar.background": "#74ade8",
        "slider.thumb.background": "#dce0e5",
        "status_bar.background": "#3b414d",
        "status_bar.border": "#464b57",
        "success.background": "#a1c181",
        "success.foreground": "#a1c181",
        "switch.background": "#454a56",
        "tab.active.background": "#282c33",
        "tab.active.foreground": "#dce0e5",
        "tab.background": "#00000000",
        "tab.foreground": "#a9afbc",
        "tab_bar.background": "#2f343e",
        "tab_bar.segmented.background": "#2e343e",
        "title_bar.background": "#3b414d",
        "title_bar.border": "#464b57",
        "warning.background": "#dec184",
        "warning.foreground": "#dec184",
        "window.border": "#464b57"
      }
    }
  ]
}"##;
