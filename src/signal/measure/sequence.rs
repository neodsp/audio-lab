use std::time::Duration;

use audio_blocks::{AudioBlock, AudioBlockMut};
use ndarray::Array2;

use crate::signal::TimeSignal;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum RecordError {
    #[error("At least one step is required")]
    NoSteps,
    #[error("At least one input channel is required")]
    NoInputs,
    #[error("Step {step} plays on {outputs} outputs, but its signal has {channels} channels")]
    OutputChannelMismatch {
        step: usize,
        outputs: usize,
        channels: usize,
    },
    #[error("Step {step} plays on output {channel} more than once")]
    DuplicateOutput { step: usize, channel: u16 },
    #[error("Channel {0} is too large, the highest supported channel is 65534")]
    ChannelOutOfRange(u16),
    #[error("Sample rate must be finite and > 0, got {0}")]
    InvalidSampleRate(f64),
    #[error("Step {step} has sample rate {actual} Hz, but step 0 has {expected} Hz")]
    SampleRateMismatch {
        step: usize,
        expected: f64,
        actual: f64,
    },
}

/// A signal to play, see [`Sequence`].
#[derive(Debug, Clone)]
pub struct Step {
    /// Output channels to play on, one per channel of `signal`.
    pub outputs: Vec<u16>,
    pub signal: TimeSignal,
}

impl Step {
    pub fn new(outputs: Vec<u16>, signal: TimeSignal) -> Self {
        Self { outputs, signal }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceConfig {
    /// Input channels to record, one per mic. All of them record during every step.
    pub inputs: Vec<u16>,
    /// Silence before the first step. It isn't part of any recording, and gives the device
    /// time to settle.
    pub warmup: Duration,
    /// Recorded after each step, with silence on the outputs, for reverb tails and device
    /// latency. It must be longer than the round-trip latency of the device, or the end of each
    /// step is cut off. The cpal backend of `audio-host` alone adds 100 ms of input latency by
    /// default, hence the 1 s default.
    pub tail: Duration,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            inputs: vec![0],
            warmup: Duration::from_millis(200),
            tail: Duration::from_secs(1),
        }
    }
}

/// Plays [`Step`]s one after another and records the inputs.
///
/// The timeline is the warmup, then each step followed by the tail:
///
/// ```text
/// [warmup][step 0][tail][step 1][tail]...
/// ```
///
/// Call [`process`](Self::process) from the audio callback until [`is_done`](Self::is_done),
/// then get one recording per step from [`into_recordings`](Self::into_recordings). Run all
/// steps in one stream: restarting the stream between steps can change the device latency,
/// which breaks the time alignment between steps.
///
/// To play several speakers at the same time, give one step several outputs. To play them one
/// after another, give each its own step.
///
/// ```
/// use std::time::Duration;
///
/// use audio_lab::signal::audio_blocks::Interleaved;
/// use audio_lab::signal::measure::{Sequence, SequenceConfig, Step};
/// use audio_lab::signal::{SweepConfig, generate_sweep};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let sweep = generate_sweep(48_000, 20.0..20_000.0, SweepConfig::default())?;
/// // left speaker, then right speaker, recorded by two mics
/// let steps = vec![Step::new(vec![0], sweep.clone()), Step::new(vec![1], sweep)];
/// let config = SequenceConfig {
///     inputs: vec![0, 1],
///     tail: Duration::from_millis(500),
///     ..Default::default()
/// };
/// let mut sequence = Sequence::new(steps, &config)?;
///
/// // normally the audio callback provides the blocks
/// let input = Interleaved::<f32>::new(sequence.num_input_channels(), 512);
/// let mut output = Interleaved::<f32>::new(sequence.num_output_channels(), 512);
/// while !sequence.is_done() {
///     sequence.process(&input, &mut output);
/// }
///
/// let recordings = sequence.into_recordings();
/// assert_eq!(recordings.len(), 2); // one per step
/// assert_eq!(recordings[0].num_channels(), 2); // one per mic
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Sequence {
    steps: Vec<PreparedStep>,
    inputs: Vec<u16>,
    sample_rate: f64,
    warmup_frames: usize,
    tail_frames: usize,
    total_frames: usize,
    num_input_channels: u16,
    num_output_channels: u16,
    /// One buffer per input, covering the timeline after the warmup.
    recordings: Vec<Vec<f32>>,
    /// Frames processed so far.
    position: usize,
    /// The first step whose tail hasn't ended yet.
    current_step: usize,
}

#[derive(Debug, Clone)]
struct PreparedStep {
    outputs: Vec<u16>,
    /// One per output.
    channels: Vec<Vec<f32>>,
    /// Timeline frame where the step starts.
    start: usize,
    len: usize,
}

impl Sequence {
    pub fn new(steps: Vec<Step>, config: &SequenceConfig) -> Result<Self, RecordError> {
        if steps.is_empty() {
            return Err(RecordError::NoSteps);
        }
        if config.inputs.is_empty() {
            return Err(RecordError::NoInputs);
        }
        let num_input_channels = num_channels_for(&config.inputs)?;

        let sample_rate = steps[0].signal.sample_rate();
        let warmup_frames = duration_to_frames(config.warmup, sample_rate);
        let tail_frames = duration_to_frames(config.tail, sample_rate);

        let mut prepared = Vec::with_capacity(steps.len());
        let mut num_output_channels = 0;
        let mut cursor = warmup_frames;
        for (index, step) in steps.into_iter().enumerate() {
            let Step { outputs, signal } = step;
            if signal.sample_rate() != sample_rate {
                return Err(RecordError::SampleRateMismatch {
                    step: index,
                    expected: sample_rate,
                    actual: signal.sample_rate(),
                });
            }
            if outputs.len() != signal.num_channels() {
                return Err(RecordError::OutputChannelMismatch {
                    step: index,
                    outputs: outputs.len(),
                    channels: signal.num_channels(),
                });
            }
            for (i, &channel) in outputs.iter().enumerate() {
                if outputs[..i].contains(&channel) {
                    return Err(RecordError::DuplicateOutput {
                        step: index,
                        channel,
                    });
                }
            }
            num_output_channels = num_output_channels.max(num_channels_for(&outputs)?);

            let len = signal.num_time_steps();
            let channels = signal
                .channel_iter()
                .map(|channel| channel.iter().map(|&s| s as f32).collect())
                .collect();
            prepared.push(PreparedStep {
                outputs,
                channels,
                start: cursor,
                len,
            });
            cursor += len + tail_frames;
        }
        let total_frames = cursor;

        let recordings = config
            .inputs
            .iter()
            .map(|_| vec![0.0; total_frames - warmup_frames])
            .collect();

        Ok(Self {
            steps: prepared,
            inputs: config.inputs.clone(),
            sample_rate,
            warmup_frames,
            tail_frames,
            total_frames,
            num_input_channels,
            num_output_channels,
            recordings,
            position: 0,
            current_step: 0,
        })
    }

    /// The input channels the stream must have: the highest recorded input plus one.
    pub fn num_input_channels(&self) -> u16 {
        self.num_input_channels
    }

    /// The output channels the stream must have: the highest output played on plus one.
    pub fn num_output_channels(&self) -> u16 {
        self.num_output_channels
    }

    /// The sample rate of the steps. The stream must run at it.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Length of the whole timeline in frames, including the warmup.
    pub fn total_frames(&self) -> usize {
        self.total_frames
    }

    /// Frames processed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether the whole timeline has been played and recorded.
    pub fn is_done(&self) -> bool {
        self.position >= self.total_frames
    }

    /// Plays the next block on `output` and records `input`.
    ///
    /// Every output channel of the block is written: silence where no step plays. Inputs and
    /// outputs that the block doesn't have are skipped (missing inputs record silence), so
    /// open the stream with [`num_input_channels`](Self::num_input_channels) and
    /// [`num_output_channels`](Self::num_output_channels). If the blocks have different
    /// lengths, the output block sets the pace and only as much input as both share is
    /// recorded. After the timeline ends, the output is silent and nothing is recorded.
    ///
    /// Real-time safe: doesn't allocate, lock or block.
    pub fn process(&mut self, input: &impl AudioBlock<f32>, output: &mut impl AudioBlockMut<f32>) {
        let num_frames = output.num_frames();
        let block_start = self.position;
        let block_end = block_start + num_frames;

        for channel in 0..output.num_channels() {
            output.channel_iter_mut(channel).for_each(|s| *s = 0.0);
        }

        // skip steps whose tail ended before this block
        while self.current_step < self.steps.len() {
            let step = &self.steps[self.current_step];
            if step.start + step.len + self.tail_frames > block_start {
                break;
            }
            self.current_step += 1;
        }

        for step in &self.steps[self.current_step..] {
            if step.start >= block_end {
                break;
            }
            let from = block_start.max(step.start);
            let to = block_end.min(step.start + step.len);
            if from >= to {
                continue;
            }
            for (&channel, samples) in step.outputs.iter().zip(&step.channels) {
                if channel >= output.num_channels() {
                    continue;
                }
                let src = &samples[from - step.start..to - step.start];
                output
                    .channel_iter_mut(channel)
                    .skip(from - block_start)
                    .zip(src)
                    .for_each(|(dst, &s)| *dst = s);
            }
        }

        // the part of this block that falls into the recorded timeline
        let record_from = block_start.max(self.warmup_frames);
        let record_to = block_end
            .min(block_start + input.num_frames())
            .min(self.total_frames);
        if record_from < record_to {
            for (&channel, recording) in self.inputs.iter().zip(&mut self.recordings) {
                if channel >= input.num_channels() {
                    continue;
                }
                let dst = &mut recording
                    [record_from - self.warmup_frames..record_to - self.warmup_frames];
                input
                    .channel_iter(channel)
                    .skip(record_from - block_start)
                    .zip(dst)
                    .for_each(|(&s, d)| *d = s);
            }
        }

        self.position = block_end;
    }

    /// One recording per step, with one channel per input (in the order of
    /// [`SequenceConfig::inputs`]). Each starts when its step starts and is the step's length
    /// plus the tail. Parts of the timeline that weren't processed yet are silent.
    pub fn into_recordings(self) -> Vec<TimeSignal> {
        let num_inputs = self.inputs.len();
        self.steps
            .iter()
            .map(|step| {
                let offset = step.start - self.warmup_frames;
                let len = step.len + self.tail_frames;
                let data = Array2::from_shape_fn((num_inputs, len), |(input, frame)| {
                    f64::from(self.recordings[input][offset + frame])
                });
                let mut signal = TimeSignal::new(data, self.sample_rate)
                    .expect("the sample rate was valid for the step's signal");
                for (index, channel) in self.inputs.iter().enumerate() {
                    signal.set_channel_label(index, Some(&format!("input {channel}")));
                }
                signal
            })
            .collect()
    }
}

fn duration_to_frames(duration: Duration, sample_rate: f64) -> usize {
    (duration.as_secs_f64() * sample_rate).round() as usize
}

/// The number of channels a stream needs to include all of `channels`.
fn num_channels_for(channels: &[u16]) -> Result<u16, RecordError> {
    match channels.iter().max() {
        Some(&u16::MAX) => Err(RecordError::ChannelOutOfRange(u16::MAX)),
        Some(&max) => Ok(max + 1),
        None => Ok(0),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use audio_blocks::{Interleaved, Planar};

    use super::*;
    use crate::dsp::{DeconvolutionConfig, deconvolve};
    use crate::signal::measure::sim::{Path, SimDevice};
    use crate::signal::{SweepConfig, generate_sweep};

    const SAMPLE_RATE: f64 = 48_000.0;

    /// Runs a sequence on a simulated device, see [`SimDevice`]. `block_sizes` repeats until
    /// the sequence is done.
    fn simulate(mut sequence: Sequence, paths: &[Path], block_sizes: &[usize]) -> Vec<TimeSignal> {
        let mut device = SimDevice::new(
            paths,
            sequence.num_input_channels(),
            sequence.num_output_channels(),
        );
        let mut sizes = block_sizes.iter().cycle();
        while !sequence.is_done() {
            device.callback(*sizes.next().unwrap(), |input, output| {
                sequence.process(input, output)
            });
        }
        sequence.into_recordings()
    }

    fn impulse(len: usize, at: usize) -> TimeSignal {
        let mut data = Array2::zeros((1, len));
        data[[0, at]] = 1.0;
        TimeSignal::new(data, SAMPLE_RATE).unwrap()
    }

    fn sweep(len: usize) -> TimeSignal {
        generate_sweep(len, 20.0..20_000.0, SweepConfig::default()).unwrap()
    }

    fn config(inputs: Vec<u16>, warmup: usize, tail: usize) -> SequenceConfig {
        SequenceConfig {
            inputs,
            warmup: Duration::from_secs_f64(warmup as f64 / SAMPLE_RATE),
            tail: Duration::from_secs_f64(tail as f64 / SAMPLE_RATE),
        }
    }

    fn argmax(signal: &TimeSignal, channel: usize) -> usize {
        signal
            .channel(channel)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0
    }

    /// The expected recording: `signal` through `taps`, delayed by `delay`, cut to `len`.
    fn filtered(signal: &TimeSignal, delay: usize, taps: &[f64], len: usize) -> Vec<f64> {
        let x: Vec<f64> = signal.channel(0).iter().map(|&s| s as f32 as f64).collect();
        (0..len)
            .map(|t| {
                taps.iter()
                    .enumerate()
                    .filter_map(|(k, tap)| {
                        t.checked_sub(delay + k)
                            .map(|i| tap * x.get(i).copied().unwrap_or(0.0))
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn recording_shows_the_device_delay() {
        let steps = vec![Step::new(vec![0], impulse(100, 10))];
        let sequence = Sequence::new(steps, &config(vec![0], 300, 2000)).unwrap();
        let recordings = simulate(sequence, &[Path::new(0, 0, 700, &[1.0])], &[256]);

        assert_eq!(recordings.len(), 1);
        assert_eq!(recordings[0].num_channels(), 1);
        assert_eq!(recordings[0].num_time_steps(), 2100);
        assert_eq!(argmax(&recordings[0], 0), 710);
        assert_abs_diff_eq!(recordings[0].channel(0).sum(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn filter_and_delay_for_every_block_size() {
        let excitation = sweep(5000);
        let taps = [0.5, -0.3, 0.2, 0.1];
        let delay = 1031;
        let tail = 1500;
        let expected = filtered(&excitation, delay, &taps, 5000 + tail);

        // block sizes that don't divide the signal length, and ones that change
        for block_sizes in [&[1][..], &[64], &[333], &[1024], &[7, 512, 100, 1, 999]] {
            let steps = vec![Step::new(vec![0], excitation.clone())];
            let sequence = Sequence::new(steps, &config(vec![0], 123, tail)).unwrap();
            let recordings = simulate(sequence, &[Path::new(0, 0, delay, &taps)], block_sizes);

            let recording = recordings[0].channel(0);
            assert_eq!(recording.len(), expected.len());
            for (actual, expected) in recording.iter().zip(&expected) {
                assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn several_mics_record_at_once() {
        let steps = vec![Step::new(vec![0], impulse(10, 0))];
        // the mics are on inputs 3 and 1, in that order
        let sequence = Sequence::new(steps, &config(vec![3, 1], 0, 2000)).unwrap();
        assert_eq!(sequence.num_input_channels(), 4);
        let paths = [Path::new(0, 3, 600, &[0.5]), Path::new(0, 1, 900, &[0.25])];
        let recordings = simulate(sequence, &paths, &[128]);

        let recording = &recordings[0];
        assert_eq!(recording.num_channels(), 2);
        assert_eq!(recording.channel_label(0), Some("input 3"));
        assert_eq!(recording.channel_label(1), Some("input 1"));
        assert_eq!(argmax(recording, 0), 600);
        assert_eq!(argmax(recording, 1), 900);
        assert_abs_diff_eq!(recording.channel(0)[600], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(recording.channel(1)[900], 0.25, epsilon = 1e-6);
    }

    #[test]
    fn speakers_one_after_another_dont_leak_into_each_other() {
        // each mic hears both speakers, at different delays
        let paths = [
            Path::new(0, 0, 600, &[1.0]),
            Path::new(0, 1, 800, &[0.5]),
            Path::new(1, 0, 900, &[0.25]),
            Path::new(1, 1, 700, &[0.75]),
        ];
        let steps = vec![
            Step::new(vec![0], impulse(500, 0)),
            Step::new(vec![1], impulse(500, 0)),
        ];
        let sequence = Sequence::new(steps, &config(vec![0, 1], 200, 1000)).unwrap();
        assert_eq!(sequence.total_frames(), 200 + 2 * 1500);
        let recordings = simulate(sequence, &paths, &[512]);

        assert_eq!(recordings.len(), 2);
        let expected = [[(600, 1.0), (800, 0.5)], [(900, 0.25), (700, 0.75)]];
        for (recording, expected) in recordings.iter().zip(expected) {
            assert_eq!(recording.num_time_steps(), 1500);
            for (mic, (at, gain)) in expected.into_iter().enumerate() {
                let channel = recording.channel(mic);
                assert_abs_diff_eq!(channel[at], gain, epsilon = 1e-6);
                // nothing else, in particular nothing from the other step
                assert_abs_diff_eq!(channel.sum(), gain, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn several_speakers_in_one_step_play_at_once() {
        let mut data = Array2::zeros((2, 100));
        data[[0, 0]] = 1.0;
        data[[1, 50]] = 1.0;
        let signal = TimeSignal::new(data, SAMPLE_RATE).unwrap();
        let steps = vec![Step::new(vec![2, 0], signal)];
        let sequence = Sequence::new(steps, &config(vec![0], 0, 1000)).unwrap();
        assert_eq!(sequence.num_output_channels(), 3);
        let paths = [Path::new(2, 0, 600, &[1.0]), Path::new(0, 0, 600, &[0.5])];
        let recordings = simulate(sequence, &paths, &[256]);

        let channel = recordings[0].channel(0);
        assert_abs_diff_eq!(channel[600], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(channel[650], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(channel.sum(), 1.5, epsilon = 1e-6);
    }

    #[test]
    fn deconvolving_the_recording_finds_the_delay() {
        let excitation = sweep(1 << 14);
        let delay = 2000;
        let steps = vec![Step::new(vec![0], excitation.clone())];
        let sequence = Sequence::new(steps, &config(vec![0], 0, 4000)).unwrap();
        let recordings = simulate(sequence, &[Path::new(0, 0, delay, &[0.5])], &[512]);

        let ir = deconvolve(&recordings[0], &excitation, &DeconvolutionConfig::default()).unwrap();
        assert_eq!(argmax(&ir, 0), delay);
    }

    #[test]
    fn output_is_silent_during_warmup_tail_and_after_the_end() {
        let signal = TimeSignal::new(Array2::ones((1, 10)), SAMPLE_RATE).unwrap();
        let mut sequence = Sequence::new(
            vec![
                Step::new(vec![1], signal.clone()),
                Step::new(vec![1], signal),
            ],
            &config(vec![0], 5, 3),
        )
        .unwrap();
        let input = Interleaved::<f32>::new(1, 40);
        let mut output = Interleaved::<f32>::new(2, 40);
        // stale data in the block must be overwritten
        output.channel_iter_mut(0).for_each(|s| *s = 9.0);
        sequence.process(&input, &mut output);

        let played: Vec<f32> = output.channel_iter(1).copied().collect();
        let mut expected = vec![0.0; 40];
        expected[5..15].fill(1.0);
        expected[18..28].fill(1.0);
        assert_eq!(played, expected);
        assert!(output.channel_iter(0).all(|&s| s == 0.0));

        // 5 + 13 + 13 = 31 frames, so the first block finished the sequence
        assert_eq!(sequence.total_frames(), 31);
        assert_eq!(sequence.position(), 40);
        assert!(sequence.is_done());

        sequence.process(&input, &mut output);
        assert!(output.channel_iter(1).all(|&s| s == 0.0));
    }

    #[test]
    fn warmup_is_not_recorded() {
        let steps = vec![Step::new(vec![0], impulse(10, 0))];
        let mut sequence = Sequence::new(steps, &config(vec![0], 4, 2)).unwrap();
        let input = Planar::<f32>::from_slice(&[(0..16).map(|i| i as f32).collect::<Vec<_>>()]);
        let mut output = Interleaved::<f32>::new(1, 16);
        sequence.process(&input, &mut output);
        assert!(sequence.is_done());

        let recording = &sequence.into_recordings()[0];
        let expected: Vec<f64> = (4..16).map(f64::from).collect();
        assert_eq!(recording.channel(0).to_vec(), expected);
    }

    #[test]
    fn missing_channels_are_skipped() {
        let steps = vec![Step::new(vec![3], impulse(10, 0))];
        let mut sequence = Sequence::new(steps, &config(vec![0, 5], 0, 0)).unwrap();
        // the stream has fewer channels than the sequence needs
        let input = Planar::<f32>::from_slice(&[vec![1.0; 10], vec![2.0; 10]]);
        let mut output = Interleaved::<f32>::new(2, 10);
        sequence.process(&input, &mut output);
        assert!(output.raw_data().iter().all(|&s| s == 0.0));

        let recording = &sequence.into_recordings()[0];
        assert!(recording.channel(0).iter().all(|&s| s == 1.0));
        assert!(recording.channel(1).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn shorter_input_block_records_only_its_frames() {
        let steps = vec![Step::new(vec![0], impulse(10, 0))];
        let mut sequence = Sequence::new(steps, &config(vec![0], 0, 0)).unwrap();
        let input = Planar::<f32>::from_slice(&[vec![1.0; 4]]);
        let mut output = Interleaved::<f32>::new(1, 6);
        sequence.process(&input, &mut output);
        assert_eq!(sequence.position(), 6);

        let recording = &sequence.into_recordings()[0];
        let expected = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(recording.channel(0).to_vec(), expected);
    }

    #[test]
    fn empty_steps_only_record_the_tail() {
        let empty = TimeSignal::new(Array2::zeros((1, 0)), SAMPLE_RATE).unwrap();
        let sequence =
            Sequence::new(vec![Step::new(vec![0], empty)], &config(vec![0], 0, 8)).unwrap();
        let recordings = simulate(sequence, &[], &[3]);
        assert_eq!(recordings[0].num_time_steps(), 8);
    }

    #[test]
    fn invalid_sequences_are_rejected() {
        let mono = impulse(10, 0);
        let new = |steps, inputs| Sequence::new(steps, &config(inputs, 0, 0)).unwrap_err();

        assert_eq!(new(vec![], vec![0]), RecordError::NoSteps);
        assert_eq!(
            new(vec![Step::new(vec![0], mono.clone())], vec![]),
            RecordError::NoInputs
        );
        assert_eq!(
            new(vec![Step::new(vec![0, 1], mono.clone())], vec![0]),
            RecordError::OutputChannelMismatch {
                step: 0,
                outputs: 2,
                channels: 1
            }
        );

        let stereo = TimeSignal::new(Array2::zeros((2, 10)), SAMPLE_RATE).unwrap();
        assert_eq!(
            new(vec![Step::new(vec![1, 1], stereo)], vec![0]),
            RecordError::DuplicateOutput {
                step: 0,
                channel: 1
            }
        );
        assert_eq!(
            new(vec![Step::new(vec![u16::MAX], mono.clone())], vec![0]),
            RecordError::ChannelOutOfRange(u16::MAX)
        );
        assert_eq!(
            new(vec![Step::new(vec![0], mono.clone())], vec![u16::MAX]),
            RecordError::ChannelOutOfRange(u16::MAX)
        );

        let other_rate = TimeSignal::new(Array2::zeros((1, 10)), 44_100.0).unwrap();
        assert_eq!(
            new(
                vec![Step::new(vec![0], mono), Step::new(vec![0], other_rate)],
                vec![0]
            ),
            RecordError::SampleRateMismatch {
                step: 1,
                expected: SAMPLE_RATE,
                actual: 44_100.0
            }
        );
    }

    #[test]
    fn durations_round_to_frames() {
        let config = SequenceConfig {
            inputs: vec![0],
            warmup: Duration::from_millis(10),
            tail: Duration::from_secs_f64(2.6 / SAMPLE_RATE),
        };
        let sequence = Sequence::new(vec![Step::new(vec![0], impulse(10, 0))], &config).unwrap();
        assert_eq!(sequence.total_frames(), 480 + 10 + 3);
    }
}
