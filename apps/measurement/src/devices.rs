//! The audio devices to choose from. Listing them can take a moment, so the app does it in the
//! background.

use audio_host::{AudioBackend, AudioHost};

/// Channel count shown for a device whose count the backend doesn't report.
const UNKNOWN_CHANNELS: u16 = 2;

#[derive(Clone)]
pub struct Device {
    pub name: String,
    pub num_channels: Option<u16>,
}

#[derive(Clone, Default)]
pub struct Devices {
    pub inputs: Vec<Device>,
    pub outputs: Vec<Device>,
    /// Names of the system default devices, empty if there is none.
    pub default_input: String,
    pub default_output: String,
}

impl Devices {
    /// Channels of the chosen output device, `None` for the default.
    pub fn num_outputs(&self, chosen: Option<&str>) -> u16 {
        num_channels(&self.outputs, chosen.unwrap_or(&self.default_output))
    }

    /// Channels of the chosen input device, `None` for the default.
    pub fn num_inputs(&self, chosen: Option<&str>) -> u16 {
        num_channels(&self.inputs, chosen.unwrap_or(&self.default_input))
    }
}

fn num_channels(devices: &[Device], name: &str) -> u16 {
    devices
        .iter()
        .find(|device| device.name == name)
        .and_then(|device| device.num_channels)
        .unwrap_or(UNKNOWN_CHANNELS)
}

/// Lists the devices. Blocks.
pub fn list() -> Result<Devices, String> {
    let host = AudioHost::new().map_err(|error| error.to_string())?;
    let convert = |devices: Vec<audio_host::DeviceInfo>| {
        devices
            .into_iter()
            .map(|device| Device {
                name: device.name,
                num_channels: device.num_channels,
            })
            .collect()
    };
    Ok(Devices {
        inputs: convert(host.inputs()),
        outputs: convert(host.outputs()),
        default_input: host.input(),
        default_output: host.output(),
    })
}

/// The device list: being listed, listed, or why it couldn't be.
pub enum DeviceList {
    Scanning,
    Ready(Devices),
    Failed(String),
}

impl DeviceList {
    pub fn devices(&self) -> Option<&Devices> {
        match self {
            Self::Ready(devices) => Some(devices),
            _ => None,
        }
    }
}
