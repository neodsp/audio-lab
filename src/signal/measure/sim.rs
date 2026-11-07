//! A simulated audio device for tests: every input hears the outputs through delays and filters.

use audio_blocks::{AudioBlock, Interleaved, Planar};

/// A path from an output to an input: a delay and an FIR filter.
#[derive(Debug, Clone)]
pub(crate) struct Path {
    pub output: u16,
    pub input: u16,
    pub delay: usize,
    pub taps: Vec<f64>,
}

impl Path {
    pub fn new(output: u16, input: u16, delay: usize, taps: &[f64]) -> Self {
        Self {
            output,
            input,
            delay,
            taps: taps.to_vec(),
        }
    }
}

/// Each input hears the sum of the paths into it. Like a real device, the input of a callback
/// only depends on output from earlier callbacks, so every delay must be at least the block
/// size.
pub(crate) struct SimDevice {
    paths: Vec<Path>,
    num_input_channels: u16,
    num_output_channels: u16,
    /// Everything played so far, per output.
    played: Vec<Vec<f64>>,
}

impl SimDevice {
    pub fn new(paths: &[Path], num_input_channels: u16, num_output_channels: u16) -> Self {
        Self {
            paths: paths.to_vec(),
            num_input_channels,
            num_output_channels,
            played: vec![Vec::new(); num_output_channels as usize],
        }
    }

    /// Runs one callback of `num_frames` frames.
    pub fn callback(
        &mut self,
        num_frames: usize,
        process: impl FnOnce(&Planar<f32>, &mut Interleaved<f32>),
    ) {
        let start = self.played.first().map_or(0, Vec::len);
        let mut input = Planar::<f32>::new(self.num_input_channels, num_frames);
        for path in &self.paths {
            assert!(path.delay >= num_frames, "delays must cover a whole block");
            let history = &self.played[path.output as usize];
            for (frame, sample) in input.channel_mut(path.input).iter_mut().enumerate() {
                let t = start + frame;
                let acc: f64 = path
                    .taps
                    .iter()
                    .enumerate()
                    .filter_map(|(k, tap)| t.checked_sub(path.delay + k).map(|i| tap * history[i]))
                    .sum();
                *sample += acc as f32;
            }
        }

        let mut output = Interleaved::<f32>::new(self.num_output_channels, num_frames);
        process(&input, &mut output);
        for (channel, history) in self.played.iter_mut().enumerate() {
            history.extend(output.channel_iter(channel as u16).map(|&s| f64::from(s)));
        }
    }
}
