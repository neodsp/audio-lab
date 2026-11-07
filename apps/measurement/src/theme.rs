//! The look: one of gpui-kit's themes, for the components and for what the app draws itself.
//! The chosen theme is remembered in the user's config directory.

use std::path::PathBuf;

use gpui_kit::component::select::SelectItem;
use gpui_kit::component::{ActiveTheme as _, Theme, ThemeRegistry};
use gpui_kit::{
    App, Global, Hsla, InteractiveElement as _, IntoElement, ParentElement as _, SharedString,
    StatefulInteractiveElement as _, Styled as _, Window, div, rgb,
};

const DEFAULT_THEME: &str = "Default Dark";

/// gpui-kit's themes besides the default light and dark one, see `themes/README.md`.
const THEMES: [&str; 21] = [
    include_str!("../themes/adventure.json"),
    include_str!("../themes/alduin.json"),
    include_str!("../themes/asciinema.json"),
    include_str!("../themes/aurora.json"),
    include_str!("../themes/ayu.json"),
    include_str!("../themes/catppuccin.json"),
    include_str!("../themes/everforest.json"),
    include_str!("../themes/fahrenheit.json"),
    include_str!("../themes/flexoki.json"),
    include_str!("../themes/gruvbox.json"),
    include_str!("../themes/harper.json"),
    include_str!("../themes/hybrid.json"),
    include_str!("../themes/jellybeans.json"),
    include_str!("../themes/kibble.json"),
    include_str!("../themes/macos-classic.json"),
    include_str!("../themes/mellifluous.json"),
    include_str!("../themes/molokai.json"),
    include_str!("../themes/solarized.json"),
    include_str!("../themes/spaceduck.json"),
    include_str!("../themes/tokyonight.json"),
    include_str!("../themes/twilight.json"),
];

/// The colors the app draws with directly, taken from the theme, so they match the components.
pub mod palette {
    use super::*;

    /// Behind the plots.
    pub fn editor(cx: &App) -> Hsla {
        cx.theme().background
    }
    /// The side bar.
    pub fn panel(cx: &App) -> Hsla {
        cx.theme().sidebar
    }
    pub fn tab_bar(cx: &App) -> Hsla {
        cx.theme().tab_bar
    }
    pub fn tab_active(cx: &App) -> Hsla {
        cx.theme().tab_active
    }
    pub fn status_bar(cx: &App) -> Hsla {
        cx.theme().status_bar
    }
    /// Popups and the readout under the mouse.
    pub fn elevated(cx: &App) -> Hsla {
        cx.theme().popover
    }
    pub fn hover(cx: &App) -> Hsla {
        cx.theme().accent
    }
    pub fn border(cx: &App) -> Hsla {
        cx.theme().border
    }
    pub fn text(cx: &App) -> Hsla {
        cx.theme().foreground
    }
    pub fn text_muted(cx: &App) -> Hsla {
        cx.theme().muted_foreground
    }
    pub fn text_placeholder(cx: &App) -> Hsla {
        cx.theme().muted_foreground.opacity(0.65)
    }
    /// Marks the user's own drawing, such as the box of a box zoom.
    pub fn accent(cx: &App) -> Hsla {
        cx.theme().primary
    }
    pub fn warning(cx: &App) -> Hsla {
        cx.theme().warning
    }
    pub fn error(cx: &App) -> Hsla {
        cx.theme().danger
    }
    pub fn grid(cx: &App) -> Hsla {
        cx.theme().chart_grid.opacity(0.45)
    }
    pub fn grid_strong(cx: &App) -> Hsla {
        cx.theme().chart_grid
    }
}

/// The `index`th of a series of well distinguishable curve colors: medium shades that read on
/// light and dark themes, then hues around the color wheel.
pub fn trace_color(index: usize) -> Hsla {
    const COLORS: [u32; 8] = [
        0x3b82f6, // blue
        0xf59e0b, // amber
        0x8b5cf6, // violet
        0x10b981, // emerald
        0xf43f5e, // rose
        0x06b6d4, // cyan
        0xf97316, // orange
        0x84cc16, // lime
    ];
    match COLORS.get(index) {
        Some(&color) => rgb(color).into(),
        None => {
            let golden_ratio = (5f32.sqrt() - 1.0) / 2.0;
            gpui_kit::hsla((index as f32 * golden_ratio).fract(), 0.75, 0.55, 1.0)
        }
    }
}

/// Loads the themes and applies the one chosen last time, or the default dark one.
pub fn init(cx: &mut App) {
    let registry = ThemeRegistry::global_mut(cx);
    for theme in THEMES {
        registry
            .load_themes_from_str(theme)
            .expect("the bundled themes are valid");
    }
    let saved = settings_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|name| SharedString::from(name.trim().to_string()));
    let applied = saved.is_some_and(|name| apply(&name, cx));
    if !applied {
        apply(DEFAULT_THEME, cx);
    }
}

/// A theme in the theme selector. Hovering it previews it.
#[derive(Clone)]
pub struct ThemeChoice(SharedString);

impl SelectItem for ThemeChoice {
    type Value = SharedString;

    fn title(&self) -> SharedString {
        self.0.clone()
    }

    fn value(&self) -> &SharedString {
        &self.0
    }

    fn render(&self, _: &mut Window, _: &mut App) -> impl IntoElement {
        let name = self.0.clone();
        div()
            .id(SharedString::from(format!("theme-{name}")))
            .w_full()
            .child(name.clone())
            .on_hover(move |hovered, _, cx| {
                if *hovered {
                    preview(&name, cx);
                }
            })
    }
}

/// All themes: the default ones first, then the others by name.
pub fn choices(cx: &App) -> Vec<ThemeChoice> {
    let mut names: Vec<SharedString> = ThemeRegistry::global(cx).themes().keys().cloned().collect();
    names.sort_by_key(|name| (!name.starts_with("Default "), name.to_lowercase()));
    names.into_iter().map(ThemeChoice).collect()
}

/// The name of the theme in use.
pub fn current(cx: &App) -> SharedString {
    cx.theme().theme_name().clone()
}

/// The theme in use before the preview started, while one is shown.
#[derive(Default)]
struct Preview {
    before: Option<SharedString>,
}

impl Global for Preview {}

/// Shows the theme `name` until [`select`] keeps it or [`end_preview`] goes back.
fn preview(name: &SharedString, cx: &mut App) {
    if current(cx) == *name {
        return;
    }
    let before = current(cx);
    let preview = cx.default_global::<Preview>();
    preview.before.get_or_insert(before);
    apply(name, cx);
}

/// Goes back to the theme from before the preview, unless one was selected meanwhile.
pub fn end_preview(cx: &mut App) {
    if let Some(before) = cx.default_global::<Preview>().before.take() {
        apply(&before, cx);
    }
}

/// Switches to the theme `name` and remembers it for the next start.
pub fn select(name: &str, cx: &mut App) {
    cx.default_global::<Preview>().before = None;
    if apply(name, cx)
        && let Some(path) = settings_path()
    {
        // not being able to remember it is no reason to bother anyone
        let _ = std::fs::create_dir_all(path.parent().expect("the path has a directory"));
        let _ = std::fs::write(path, name);
    }
}

/// Switches to the theme `name`, if there is one.
fn apply(name: &str, cx: &mut App) -> bool {
    let Some(config) = ThemeRegistry::global(cx)
        .themes()
        .get(&SharedString::from(name.to_string()))
        .cloned()
    else {
        return false;
    };
    Theme::update(cx, |theme| theme.apply_config(&config));
    true
}

/// Where the chosen theme is stored: the platform's config directory.
fn settings_path() -> Option<PathBuf> {
    let env = |name| std::env::var_os(name).map(PathBuf::from);
    let base = if cfg!(target_os = "windows") {
        env("APPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|home| home.join("Library/Application Support"))
    } else {
        env("XDG_CONFIG_HOME").or_else(|| env("HOME").map(|home| home.join(".config")))
    }?;
    Some(base.join("audio-lab").join("measurement-theme"))
}
