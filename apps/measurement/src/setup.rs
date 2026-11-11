//! The setup of a new measurement, in the side bar: sweep or live, devices, speakers and
//! microphones, the sweep or the noise.

use gpui_kit::base::FocusableExt as _;
use gpui_kit::base::Selectable as _;
use gpui_kit::component::button::{Button, ButtonGroup, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::{Input, InputEvent, InputState, Textarea, TextareaState};
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::select::{Select, SelectEvent, SelectItem, SelectState};
use gpui_kit::component::slider::{Slider, SliderEvent, SliderState};
use gpui_kit::component::spinner::Spinner;
use gpui_kit::component::{
    Disableable as _, IconName, IndexPath, Sizable as _, StyledExt as _, h_flex, v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::devices::{self, DeviceList};
use crate::live::{LiveRequest, period_len};
use crate::measure::{self, Channel, Request, Settings, is_marked, same_name};
use crate::theme::palette;

/// Width of the labels in front of the controls.
const LABEL_WIDTH: f32 = 84.0;

/// An entry of a drop-down: what it shows and what it stands for.
#[derive(Clone)]
pub struct Choice<T> {
    title: SharedString,
    value: T,
}

impl<T: Clone + PartialEq + 'static> SelectItem for Choice<T> {
    type Value = T;

    fn title(&self) -> SharedString {
        self.title.clone()
    }

    fn value(&self) -> &T {
        &self.value
    }
}

type ChoiceState<T> = Entity<SelectState<Vec<Choice<T>>>>;

pub enum SetupEvent {
    Start(Request),
    StartLive(LiveRequest),
    Close,
}

/// A speaker or microphone in the setup: its name field.
struct ChannelRow {
    name: Entity<InputState>,
    _subscription: Subscription,
}

pub struct Setup {
    settings: Settings,
    devices: DeviceList,
    /// Why the last measurement failed.
    error: Option<String>,
    name: Entity<InputState>,
    /// What the user writes about the next measurement while setting it up. It goes with
    /// the measurement.
    notes: Entity<TextareaState>,
    /// The speaker and the microphone whose name or check box was changed last: of two
    /// with the same name, that one is marked.
    edited: [Option<usize>; 2],
    /// The names of the folders there are: a new one needs another.
    taken: Vec<String>,
    /// The name last filled in for the user, to tell whether they typed their own since.
    suggested_name: String,
    output_device: ChoiceState<Option<String>>,
    input_device: ChoiceState<Option<String>>,
    outputs: Vec<ChannelRow>,
    inputs: Vec<ChannelRow>,
    sample_rate: ChoiceState<u32>,
    sweep_len: ChoiceState<u32>,
    /// Noise periods averaged per live update.
    averaging: ChoiceState<usize>,
    level: Entity<SliderState>,
    freq_low: Entity<InputState>,
    freq_high: Entity<InputState>,
    tail: Entity<InputState>,
    ir_length: Entity<InputState>,
    _subscriptions: Vec<Subscription>,
    _scan: Option<Task<()>>,
}

impl EventEmitter<SetupEvent> for Setup {}

impl Setup {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let settings = Settings::default();
        let mut subscriptions = Vec::new();

        let name = cx.new(|cx| InputState::new(window, cx).default_value(settings.name.clone()));
        subscriptions.push(cx.subscribe(&name, |this, input, event: &InputEvent, cx| {
            if matches!(event, InputEvent::Change) {
                this.settings.name = input.read(cx).value().to_string();
                cx.notify();
            }
        }));

        let notes = cx.new(|cx| {
            TextareaState::new(window, cx).placeholder("What is measured,\nwhat changed…")
        });

        let device_select = |window: &mut Window, cx: &mut Context<Self>| {
            let items = vec![Choice {
                title: "System default".into(),
                value: None,
            }];
            cx.new(|cx| SelectState::new(items, Some(IndexPath::new(0)), window, cx))
        };
        let output_device = device_select(window, cx);
        let input_device = device_select(window, cx);
        subscriptions.push(cx.subscribe_in(
            &output_device,
            window,
            |this, _, event: &SelectEvent<_>, window, cx| {
                let SelectEvent::Confirm(Some(device)) = event else {
                    return;
                };
                this.settings.output_device = device.clone();
                this.sync_channels(window, cx);
            },
        ));
        subscriptions.push(cx.subscribe_in(
            &input_device,
            window,
            |this, _, event: &SelectEvent<_>, window, cx| {
                let SelectEvent::Confirm(Some(device)) = event else {
                    return;
                };
                this.settings.input_device = device.clone();
                this.sync_channels(window, cx);
            },
        ));

        let rates: Vec<Choice<u32>> = [44_100, 48_000, 88_200, 96_000]
            .into_iter()
            .map(|rate| Choice {
                title: format!("{rate} Hz").into(),
                value: rate,
            })
            .collect();
        let rate_index = rates.iter().position(|c| c.value == settings.sample_rate);
        let sample_rate =
            cx.new(|cx| SelectState::new(rates, rate_index.map(IndexPath::new), window, cx));
        subscriptions.push(cx.subscribe_in(
            &sample_rate,
            window,
            |this, _, event: &SelectEvent<_>, window, cx| {
                let SelectEvent::Confirm(Some(rate)) = event else {
                    return;
                };
                this.settings.sample_rate = *rate;
                this.update_sweep_lengths(window, cx);
                this.update_averaging(window, cx);
                cx.notify();
            },
        ));

        let sweep_len =
            cx.new(|cx| SelectState::new(sweep_lengths(settings.sample_rate), None, window, cx));
        subscriptions.push(
            cx.subscribe(&sweep_len, |this, _, event: &SelectEvent<_>, cx| {
                let SelectEvent::Confirm(Some(exponent)) = event else {
                    return;
                };
                this.settings.sweep_exponent = *exponent;
                cx.notify();
            }),
        );

        let averaging = cx
            .new(|cx| SelectState::new(averaging_choices(settings.sample_rate), None, window, cx));
        subscriptions.push(
            cx.subscribe(&averaging, |this, _, event: &SelectEvent<_>, cx| {
                let SelectEvent::Confirm(Some(periods)) = event else {
                    return;
                };
                this.settings.periods = *periods;
                cx.notify();
            }),
        );

        let level = cx.new(|_| {
            SliderState::new()
                .min(-60.0)
                .max(0.0)
                .step(1.0)
                .default_value(settings.level_db as f32)
        });
        subscriptions.push(cx.subscribe(&level, |this, _, event: &SliderEvent, cx| {
            let (SliderEvent::Change(value) | SliderEvent::Release(value)) = event;
            this.settings.level_db = f64::from(value.start());
            cx.notify();
        }));

        let mut number = |value: f64,
                          set: fn(&mut Settings, f64),
                          window: &mut Window,
                          cx: &mut Context<Self>| {
            let input = cx.new(|cx| InputState::new(window, cx).default_value(format!("{value}")));
            subscriptions.push(
                cx.subscribe(&input, move |this, input, event: &InputEvent, cx| {
                    if matches!(event, InputEvent::Change)
                        && let Ok(value) = input.read(cx).value().trim().parse::<f64>()
                    {
                        set(&mut this.settings, value);
                        cx.notify();
                    }
                }),
            );
            input
        };
        let freq_low = number(settings.freq_low, |s, v| s.freq_low = v, window, cx);
        let freq_high = number(settings.freq_high, |s, v| s.freq_high = v, window, cx);
        let tail = number(
            settings.tail_ms,
            |s, v| s.tail_ms = v.clamp(200.0, 10_000.0),
            window,
            cx,
        );
        let ir_length = number(
            settings.ir_length_ms,
            |s, v| s.ir_length_ms = v.clamp(10.0, 5_000.0),
            window,
            cx,
        );

        let mut setup = Self {
            suggested_name: settings.name.clone(),
            taken: Vec::new(),
            edited: [None; 2],
            settings,
            devices: DeviceList::Scanning,
            error: None,
            name,
            notes,
            output_device,
            input_device,
            outputs: Vec::new(),
            inputs: Vec::new(),
            sample_rate,
            sweep_len,
            averaging,
            level,
            freq_low,
            freq_high,
            tail,
            ir_length,
            _subscriptions: subscriptions,
            _scan: None,
        };
        setup.update_sweep_lengths(window, cx);
        setup.update_averaging(window, cx);
        setup.scan(window, cx);
        setup
    }

    /// The name the next measurement gets.
    pub fn name(&self) -> String {
        self.settings.name.trim().to_string()
    }

    /// The notes for the measurement that just started or finished; the field is cleared for
    /// the next one.
    pub fn take_notes(&mut self, window: &mut Window, cx: &mut Context<Self>) -> String {
        let text = self.notes.read(cx).value().to_string();
        self.notes
            .update(cx, |notes, cx| notes.set_value("", window, cx));
        text
    }

    pub fn set_error(&mut self, error: Option<String>, cx: &mut Context<Self>) {
        self.error = error;
        cx.notify();
    }

    /// After a measurement: fills in the next free number for the next one. `taken` are the
    /// names of the measurements there are.
    pub fn count_up<'a>(
        &mut self,
        taken: impl IntoIterator<Item = &'a str>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.taken = taken.into_iter().map(str::to_string).collect();
        let name = next_name(&self.settings.name, self.taken.iter().map(String::as_str));
        self.suggest_name(name, window, cx);
    }

    /// After deleting measurements: their numbers are free again, so the suggested name may
    /// go back down, to 1 once all are gone. A name the user typed stays.
    pub fn renumber<'a>(
        &mut self,
        taken: impl IntoIterator<Item = &'a str>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.taken = taken.into_iter().map(str::to_string).collect();
        if self.settings.name != self.suggested_name {
            // a name the user typed stays, but may now be taken
            cx.notify();
            return;
        }
        let name = next_name(&self.settings.name, self.taken.iter().map(String::as_str));
        self.suggest_name(name, window, cx);
    }

    /// Whether a folder already has the name the next measurement would get. An empty name
    /// gets a free one.
    fn name_taken(&self) -> bool {
        !self.settings.live
            && self
                .taken
                .iter()
                .any(|taken| same_name(taken, &self.settings.name))
    }

    /// Why the measurement can't start, if it can't.
    fn problem(&self) -> Option<String> {
        if self.name_taken() {
            return Some(format!(
                "There is a folder called “{}” already.",
                self.settings.name.trim()
            ));
        }
        self.settings.problem()
    }

    fn suggest_name(&mut self, name: String, window: &mut Window, cx: &mut Context<Self>) {
        if name != self.settings.name {
            self.name
                .update(cx, |input, cx| input.set_value(name.clone(), window, cx));
        }
        self.suggested_name = name.clone();
        self.settings.name = name;
        cx.notify();
    }

    /// Starts listing the devices in the background.
    fn scan(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.devices = DeviceList::Scanning;
        cx.notify();
        self._scan = Some(cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async { devices::list() }).await;
            this.update_in(cx, |this, window, cx| {
                this.devices = match result {
                    Ok(devices) => DeviceList::Ready(devices),
                    Err(error) => DeviceList::Failed(error),
                };
                this.update_device_choices(window, cx);
            })
            .ok();
        }));
    }

    fn update_device_choices(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let devices = self.devices.devices().cloned().unwrap_or_default();
        let choices = |list: &[devices::Device], default: &str| {
            let mut choices = vec![Choice {
                title: if default.is_empty() {
                    "System default".into()
                } else {
                    format!("System default ({default})").into()
                },
                value: None,
            }];
            choices.extend(list.iter().map(|device| Choice {
                title: match device.num_channels {
                    Some(n) => format!("{} ({n} ch)", device.name).into(),
                    None => device.name.clone().into(),
                },
                value: Some(device.name.clone()),
            }));
            choices
        };
        // a device that is gone falls back to the default
        let known = |chosen: &Option<String>, list: &[devices::Device]| {
            chosen
                .clone()
                .filter(|name| list.iter().any(|device| &device.name == name))
        };
        self.settings.output_device = known(&self.settings.output_device, &devices.outputs);
        self.settings.input_device = known(&self.settings.input_device, &devices.inputs);
        let output = self.settings.output_device.clone();
        let input = self.settings.input_device.clone();
        self.output_device.update(cx, |select, cx| {
            select.set_items(
                choices(&devices.outputs, &devices.default_output),
                window,
                cx,
            );
            select.set_selected_value(&output, window, cx);
        });
        self.input_device.update(cx, |select, cx| {
            select.set_items(choices(&devices.inputs, &devices.default_input), window, cx);
            select.set_selected_value(&input, window, cx);
        });
        self.sync_channels(window, cx);
    }

    fn update_sweep_lengths(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let exponent = self.settings.sweep_exponent;
        let lengths = sweep_lengths(self.settings.sample_rate);
        self.sweep_len.update(cx, |select, cx| {
            select.set_items(lengths, window, cx);
            select.set_selected_value(&exponent, window, cx);
        });
    }

    fn update_averaging(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let periods = self.settings.periods;
        let choices = averaging_choices(self.settings.sample_rate);
        self.averaging.update(cx, |select, cx| {
            select.set_items(choices, window, cx);
            select.set_selected_value(&periods, window, cx);
        });
    }

    /// Matches the speakers and microphones to the chosen devices.
    fn sync_channels(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (num_outputs, num_inputs) = if self.settings.simulated {
            (measure::SIMULATED_CHANNELS, measure::SIMULATED_CHANNELS)
        } else {
            let devices = self.devices.devices().cloned().unwrap_or_default();
            (
                devices.num_outputs(self.settings.output_device.as_deref()),
                devices.num_inputs(self.settings.input_device.as_deref()),
            )
        };
        self.settings.sync_channels(num_outputs, num_inputs);
        sync_rows(&mut self.outputs, &self.settings.outputs, true, window, cx);
        sync_rows(&mut self.inputs, &self.settings.inputs, false, window, cx);
        cx.notify();
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        if self.problem().is_none() {
            self.error = None;
            cx.emit(if self.settings.live {
                SetupEvent::StartLive(self.settings.live_request())
            } else {
                SetupEvent::Start(self.settings.request())
            });
        }
    }

    /// Sweep or live.
    fn render_mode(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let live = self.settings.live;
        ButtonGroup::new("mode")
            .small()
            .outline()
            .w_full()
            // apart from the fields: it switches what they are
            .mb_2()
            .child(
                Button::new("mode-sweep")
                    .flex_1()
                    .icon(gpui_kit::assets::IconName::AudioWaveform)
                    .label("Sweep")
                    .selected(!live),
            )
            .child(
                Button::new("mode-live")
                    .flex_1()
                    .icon(gpui_kit::assets::IconName::Radio)
                    .label("Live")
                    .selected(live),
            )
            .on_click(cx.listener(|this, clicked: &Vec<usize>, _, cx| {
                this.settings.live = clicked.contains(&1);
                cx.notify();
            }))
    }

    fn render_noise(&self, cx: &App) -> impl IntoElement {
        v_flex()
            .gap_2()
            .child(row(
                cx,
                "Sample rate",
                Select::new(&self.sample_rate).small(),
            ))
            .child(row(cx, "Level", self.render_level(cx)))
            .child(row(cx, "Averaging", Select::new(&self.averaging).small()))
            .child(hint(
                cx,
                "Pink noise plays on the speakers until you stop it; switch them on and off \
                 while it runs. More averaging is steadier, less follows changes faster.",
            ))
    }

    fn render_level(&self, cx: &App) -> impl IntoElement {
        let level = self.settings.level_db;
        h_flex()
            .gap_3()
            .child(div().flex_1().child(Slider::new(&self.level)))
            .child(
                div()
                    .w(px(64.0))
                    .text_sm()
                    .text_color(palette::text_muted(cx))
                    .child(format!("{level:.0} dBFS")),
            )
    }

    fn render_devices(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut body = v_flex().gap_2().child(
            Checkbox::new("simulated")
                .small()
                .label("Simulated device (no hardware)")
                .checked(self.settings.simulated)
                .on_change(cx.listener(|this, checked: &bool, window, cx| {
                    this.settings.simulated = *checked;
                    this.sync_channels(window, cx);
                })),
        );
        if self.settings.simulated {
            return body;
        }
        body = match &self.devices {
            DeviceList::Scanning => body.child(
                h_flex()
                    .gap_2()
                    .text_sm()
                    .text_color(palette::text_muted(cx))
                    .child(Spinner::new().small())
                    .child("Looking for devices…"),
            ),
            DeviceList::Failed(error) => body
                .child(
                    div()
                        .text_sm()
                        .text_color(palette::error(cx))
                        .child(format!("No devices: {error}")),
                )
                .child(
                    Button::new("retry")
                        .small()
                        .label("Retry")
                        .on_click(cx.listener(|this, _, window, cx| this.scan(window, cx))),
                ),
            DeviceList::Ready(_) => body
                .child(row(
                    cx,
                    "Output",
                    Select::new(&self.output_device)
                        .small()
                        .menu_width(px(360.0)),
                ))
                .child(row(
                    cx,
                    "Input",
                    Select::new(&self.input_device)
                        .small()
                        .menu_width(px(360.0)),
                ))
                .child(
                    h_flex().child(
                        Button::new("refresh")
                            .ghost()
                            .xsmall()
                            .icon(IconName::RefreshCw)
                            .label("Refresh devices")
                            .on_click(cx.listener(|this, _, window, cx| this.scan(window, cx))),
                    ),
                ),
        };
        body
    }

    fn render_channels(&self, outputs: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let (rows, channels) = if outputs {
            (&self.outputs, &self.settings.outputs)
        } else {
            (&self.inputs, &self.settings.inputs)
        };
        let prefix = if outputs { "output" } else { "input" };
        let edited = self.edited[usize::from(!outputs)];
        v_flex()
            .gap_1()
            .children(
                rows.iter()
                    .zip(channels)
                    .enumerate()
                    .map(|(index, (row, channel))| {
                        h_flex()
                            .gap_2()
                            .child(
                                div().w(px(LABEL_WIDTH)).child(
                                    Checkbox::new(SharedString::from(format!("{prefix}-{index}")))
                                        .small()
                                        .label(format!("{}", index + 1))
                                        .checked(channel.enabled)
                                        .on_change(cx.listener(
                                            move |this, checked: &bool, _, cx| {
                                                let channels = if outputs {
                                                    &mut this.settings.outputs
                                                } else {
                                                    &mut this.settings.inputs
                                                };
                                                if let Some(channel) = channels.get_mut(index) {
                                                    channel.enabled = *checked;
                                                }
                                                this.edited[usize::from(!outputs)] = Some(index);
                                                cx.notify();
                                            },
                                        )),
                                ),
                            )
                            .child(
                                div().flex_1().child(
                                    Input::new(&row.name)
                                        .small()
                                        .disabled(!channel.enabled)
                                        .when(is_marked(channels, index, edited), |this| {
                                            this.focus_ring(false).border_color(palette::error(cx))
                                        }),
                                ),
                            )
                            .when(is_marked(channels, index, edited), |this| {
                                let kind = if outputs { "speaker" } else { "microphone" };
                                this.child(name_clash(
                                    SharedString::from(format!("{prefix}-clash-{index}")),
                                    format!("Another {kind} is called “{}”", channel.name.trim()),
                                    cx,
                                ))
                            })
                    }),
            )
    }

    fn render_sweep(&self, cx: &App) -> impl IntoElement {
        v_flex()
            .gap_2()
            .child(row(
                cx,
                "Sample rate",
                Select::new(&self.sample_rate).small(),
            ))
            .child(row(cx, "Length", Select::new(&self.sweep_len).small()))
            .child(row(cx, "Level", self.render_level(cx)))
            .child(row(cx, "From", number(cx, &self.freq_low, "Hz")))
            .child(row(cx, "To", number(cx, &self.freq_high, "Hz")))
            .child(row(cx, "Tail", number(cx, &self.tail, "ms")))
            .child(row(cx, "IR window", number(cx, &self.ir_length, "ms")))
            .child(hint(
                cx,
                "The tail is recorded after each sweep: longer than the latency plus the reverb.",
            ))
    }
}

impl Render for Setup {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let problem = self.problem();
        let request = self.settings.request();
        v_flex()
            .size_full()
            .child(
                h_flex()
                    .h(px(36.0))
                    .flex_shrink_0()
                    .px_2()
                    .gap_1()
                    .border_b_1()
                    .border_color(palette::border(cx))
                    .child(
                        Button::new("back")
                            .ghost()
                            .xsmall()
                            .icon(IconName::ArrowLeft)
                            .tooltip("Back to the measurements")
                            .on_click(cx.listener(|_, _, _, cx| cx.emit(SetupEvent::Close))),
                    )
                    .child(div().text_sm().font_medium().child("New measurement")),
            )
            .child(
                div().flex_1().min_h_0().child(
                    v_flex()
                        .id("setup-scroll")
                        .size_full()
                        .overflow_y_scrollbar()
                        .child(
                            v_flex()
                                .p_3()
                                .gap_2()
                                .child(self.render_mode(cx))
                                .when(!self.settings.live, |this| {
                                    this.child(row(
                                        cx,
                                        "Name",
                                        h_flex()
                                            .gap_2()
                                            .child(div().flex_1().child(
                                                Input::new(&self.name).small().when(
                                                    self.name_taken(),
                                                    |this| {
                                                        this.focus_ring(false)
                                                            .border_color(palette::error(cx))
                                                    },
                                                ),
                                            ))
                                            .when(self.name_taken(), |this| {
                                                this.child(name_clash(
                                                    "name-clash",
                                                    format!(
                                                        "There is a folder called “{}” already",
                                                        self.settings.name.trim()
                                                    ),
                                                    cx,
                                                ))
                                            }),
                                    ))
                                })
                                .child(row(cx, "Notes", Textarea::new(&self.notes).h(px(64.0))))
                                .child(section(cx, "Devices"))
                                .child(self.render_devices(cx))
                                .child(section(cx, "Speakers"))
                                .child(self.render_channels(true, cx))
                                .child(section(cx, "Microphones"))
                                .child(self.render_channels(false, cx))
                                .map(|this| {
                                    if self.settings.live {
                                        this.child(section(cx, "Noise"))
                                            .child(self.render_noise(cx))
                                    } else {
                                        this.child(section(cx, "Sweep"))
                                            .child(self.render_sweep(cx))
                                    }
                                }),
                        ),
                ),
            )
            .child(
                v_flex()
                    .flex_shrink_0()
                    .p_3()
                    .gap_2()
                    .border_t_1()
                    .border_color(palette::border(cx))
                    .child(
                        Button::new("start")
                            .primary()
                            .small()
                            .w_full()
                            .icon(IconName::Play)
                            .label(if self.settings.live {
                                "Start live"
                            } else {
                                "Start measurement"
                            })
                            .disabled(problem.is_some())
                            .on_click(cx.listener(|this, _, _, cx| this.start(cx))),
                    )
                    .map(|this| match problem {
                        Some(problem) => this.child(hint(cx, problem)),
                        None if !self.settings.simulated && !self.settings.live => {
                            this.child(hint(
                                cx,
                                format!(
                                    "Takes about {:.0} s. Keep it quiet.",
                                    request.duration().as_secs_f64()
                                ),
                            ))
                        }
                        None => this,
                    })
                    .when_some(self.error.clone(), |this, error| {
                        this.child(div().text_sm().text_color(palette::error(cx)).child(error))
                    }),
            )
    }
}

/// Keeps one name field per channel, filled with the channel's name.
fn sync_rows(
    rows: &mut Vec<ChannelRow>,
    channels: &[Channel],
    outputs: bool,
    window: &mut Window,
    cx: &mut Context<Setup>,
) {
    rows.truncate(channels.len());
    for (index, channel) in channels.iter().enumerate().skip(rows.len()) {
        let name = cx.new(|cx| InputState::new(window, cx).default_value(channel.name.clone()));
        let subscription = cx.subscribe(&name, move |this, input, event: &InputEvent, cx| {
            if !matches!(event, InputEvent::Change) {
                return;
            }
            let channels = if outputs {
                &mut this.settings.outputs
            } else {
                &mut this.settings.inputs
            };
            if let Some(channel) = channels.get_mut(index) {
                channel.name = input.read(cx).value().to_string();
            }
            this.edited[usize::from(!outputs)] = Some(index);
            // names that clash are marked
            cx.notify();
        });
        rows.push(ChannelRow {
            name,
            _subscription: subscription,
        });
    }
}

/// How many noise periods to average, with how long that takes at `sample_rate`.
fn averaging_choices(sample_rate: u32) -> Vec<Choice<usize>> {
    let period = period_len(f64::from(sample_rate)) as f64 / f64::from(sample_rate);
    [1, 2, 4, 8]
        .into_iter()
        .map(|periods| Choice {
            title: format!(
                "{periods} period{} ({:.1} s)",
                if periods == 1 { "" } else { "s" },
                periods as f64 * period
            )
            .into(),
            value: periods,
        })
        .collect()
}

/// The sweep lengths to choose from at `sample_rate`, as powers of two.
fn sweep_lengths(sample_rate: u32) -> Vec<Choice<u32>> {
    (14..=20)
        .map(|exponent| Choice {
            title: format!(
                "{:.1} s (2^{exponent})",
                (1u64 << exponent) as f64 / f64::from(sample_rate)
            )
            .into(),
            value: exponent,
        })
        .collect()
}

fn section(cx: &App, title: &'static str) -> impl IntoElement {
    div()
        .pt_3()
        .pb_1()
        .text_xs()
        .font_medium()
        .text_color(palette::text_muted(cx))
        .child(title)
}

/// A label and its control.
fn row(cx: &App, label: &'static str, control: impl IntoElement) -> impl IntoElement {
    h_flex()
        .gap_2()
        .child(
            div()
                .w(px(LABEL_WIDTH))
                .flex_shrink_0()
                .text_sm()
                .text_color(palette::text_muted(cx))
                .child(label),
        )
        .child(div().flex_1().min_w_0().child(control))
}

fn number(cx: &App, input: &Entity<InputState>, unit: &'static str) -> impl IntoElement {
    Input::new(input).small().suffix(
        div()
            .text_xs()
            .text_color(palette::text_placeholder(cx))
            .child(unit),
    )
}

/// A red warning next to a name that is taken, saying by what.
pub fn name_clash(
    id: impl Into<ElementId>,
    message: impl Into<SharedString>,
    cx: &App,
) -> impl IntoElement {
    let message = message.into();
    div()
        .id(id.into())
        .flex_shrink_0()
        .child(
            gpui_kit::component::Icon::new(IconName::TriangleAlert)
                .small()
                .text_color(palette::error(cx)),
        )
        .tooltip(move |window, cx| {
            gpui_kit::component::tooltip::Tooltip::new(message.clone()).build(window, cx)
        })
}

pub fn hint(cx: &App, text: impl Into<SharedString>) -> impl IntoElement {
    div()
        .text_xs()
        .text_color(palette::text_placeholder(cx))
        .child(text.into())
}

/// The name for the next measurement: the stem of `name` (`Living room` of `Living room 3`,
/// `Measurement` for a name without a number) with the number after the highest one `taken`
/// has with that stem, or 1.
pub fn next_name<'a>(name: &str, taken: impl IntoIterator<Item = &'a str>) -> String {
    let stem = match split_number(name) {
        Some((stem, _)) => stem,
        None => "Measurement",
    };
    let highest = taken
        .into_iter()
        .filter_map(split_number)
        .filter(|(s, _)| same_name(s, stem))
        .map(|(_, number)| number)
        .max()
        .unwrap_or(0);
    format!("{stem} {}", highest + 1)
}

/// `Living room 3` as `("Living room", 3)`.
fn split_number(name: &str) -> Option<(&str, u64)> {
    let (stem, number) = name.trim().rsplit_once(' ')?;
    Some((stem.trim_end(), number.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    // `gpui_kit::*` brings GPUI's test macro, which needs an app
    use core::prelude::v1::test;

    #[test]
    fn names_count_up() {
        assert_eq!(
            next_name("Measurement 3", ["Measurement 3"]),
            "Measurement 4"
        );
        assert_eq!(
            next_name("Living room 9", ["Living room 9", "Measurement 12"]),
            "Living room 10"
        );
        assert_eq!(
            next_name("Living room", ["Living room", "Measurement 1"]),
            "Measurement 2"
        );
        // the highest one counts, not the last one
        assert_eq!(
            next_name("Measurement 2", ["Measurement 5", "Measurement 2"]),
            "Measurement 6"
        );
    }

    #[test]
    fn numbers_start_over_when_all_are_gone() {
        assert_eq!(next_name("Measurement 4", []), "Measurement 1");
        assert_eq!(
            next_name("Measurement 4", ["Measurement 1"]),
            "Measurement 2"
        );
        assert_eq!(next_name("Kitchen 7", ["Measurement 3"]), "Kitchen 1");
    }

    #[test]
    fn sweep_lengths_are_powers_of_two() {
        let lengths = sweep_lengths(48_000);
        assert_eq!(lengths.first().unwrap().value, 14);
        assert_eq!(&*lengths[2].title, "1.4 s (2^16)");
    }
}
