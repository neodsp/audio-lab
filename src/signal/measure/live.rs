use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use audio_blocks::{AudioBlock, AudioBlockMut};
use ndarray::Array2;

use crate::signal::TimeSignal;
use crate::signal::measure::RecordError;

#[derive(Debug, Clone, PartialEq)]
pub struct LiveConfig {
    /// Input channels to record, one per mic.
    pub inputs: Vec<u16>,
    pub sample_rate: f64,
    /// Frames kept for [`LiveReader::latest`].
    pub history: usize,
    /// Frames the audio thread can queue before the reader picks them up. Read more often than
    /// this fills up, or frames are dropped.
    pub queue: usize,
    /// The largest block the writer accepts. Larger blocks play silence and are dropped.
    pub max_block: usize,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            inputs: vec![0],
            sample_rate: 48_000.0,
            history: 1 << 17,
            queue: 1 << 17,
            max_block: 32_768,
        }
    }
}

/// Creates the two ends of a live measurement: the [`LiveWriter`] runs in the audio callback,
/// the [`LiveReader`] on another thread. They share a lock-free queue, so the audio thread never
/// waits and no samples are lost as long as the reader keeps up.
pub fn live_channel(config: &LiveConfig) -> Result<(LiveWriter, LiveReader), RecordError> {
    if config.inputs.is_empty() {
        return Err(RecordError::NoInputs);
    }
    if let Some(&channel) = config.inputs.iter().find(|&&c| c == u16::MAX) {
        return Err(RecordError::ChannelOutOfRange(channel));
    }
    if !config.sample_rate.is_finite() || config.sample_rate <= 0.0 {
        return Err(RecordError::InvalidSampleRate(config.sample_rate));
    }
    // inputs plus the reference
    let stride = config.inputs.len() + 1;
    let (producer, consumer) = rtrb::RingBuffer::new(config.queue.max(1) * stride);
    let dropped = Arc::new(AtomicU64::new(0));
    let writer = LiveWriter {
        producer,
        inputs: config.inputs.clone(),
        reference: vec![0.0; config.max_block],
        dropped: Arc::clone(&dropped),
    };
    let reader = LiveReader {
        consumer,
        inputs: config.inputs.clone(),
        sample_rate: config.sample_rate,
        history: vec![vec![0.0; config.history]; stride],
        write: 0,
        len: 0,
        dropped,
    };
    Ok((writer, reader))
}

/// The audio-thread end of a live measurement, see [`live_channel`].
pub struct LiveWriter {
    producer: rtrb::Producer<f32>,
    inputs: Vec<u16>,
    reference: Vec<f32>,
    dropped: Arc<AtomicU64>,
}

impl LiveWriter {
    /// The input channels the stream needs: the highest recorded input plus one.
    pub fn num_input_channels(&self) -> u16 {
        self.inputs.iter().max().map_or(0, |&max| max + 1)
    }

    /// Lets `source` fill `output` and the reference, then queues the inputs and the reference.
    ///
    /// `source` gets the output block and a reference buffer with one sample per frame, cleared
    /// to silence: the signal the measurement is relative to, e.g. the noise before any
    /// processing. Inputs the block doesn't have record silence. If the input block is shorter
    /// than the output block, only the frames both have are queued. If the queue is full, the
    /// block is dropped and counted in [`LiveReader::dropped_frames`].
    ///
    /// Real-time safe: doesn't allocate, lock or block.
    pub fn process<O: AudioBlockMut<f32>>(
        &mut self,
        input: &impl AudioBlock<f32>,
        output: &mut O,
        source: impl FnOnce(&mut O, &mut [f32]),
    ) {
        let num_frames = output.num_frames();
        if num_frames > self.reference.len() {
            for channel in 0..output.num_channels() {
                output.channel_iter_mut(channel).for_each(|s| *s = 0.0);
            }
            self.dropped.fetch_add(num_frames as u64, Ordering::Relaxed);
            return;
        }
        let reference = &mut self.reference[..num_frames];
        reference.fill(0.0);
        source(output, reference);

        let frames = num_frames.min(input.num_frames());
        let stride = self.inputs.len() + 1;
        let Ok(mut chunk) = self.producer.write_chunk(frames * stride) else {
            self.dropped.fetch_add(frames as u64, Ordering::Relaxed);
            return;
        };
        let (first, second) = chunk.as_mut_slices();
        let values = (0..frames).flat_map(|frame| {
            self.inputs
                .iter()
                .map(move |&channel| {
                    if channel < input.num_channels() {
                        input.sample(channel, frame)
                    } else {
                        0.0
                    }
                })
                .chain(std::iter::once(reference[frame]))
        });
        for (slot, value) in first.iter_mut().chain(second.iter_mut()).zip(values) {
            *slot = value;
        }
        chunk.commit_all();
    }
}

/// The most recent frames of a live measurement, time-aligned.
#[derive(Debug, Clone)]
pub struct LiveSnapshot {
    /// One channel per input, labelled `input N`.
    pub inputs: TimeSignal,
    /// Mono: the reference written by the source.
    pub reference: TimeSignal,
}

/// The reading end of a live measurement, see [`live_channel`]. Keeps the most recent
/// [`LiveConfig::history`] frames.
pub struct LiveReader {
    consumer: rtrb::Consumer<f32>,
    inputs: Vec<u16>,
    sample_rate: f64,
    /// One ring per input, then the reference.
    history: Vec<Vec<f32>>,
    /// Next ring position to write.
    write: usize,
    /// Frames in the rings, at most the history length.
    len: usize,
    dropped: Arc<AtomicU64>,
}

impl LiveReader {
    /// Moves everything queued by the writer into the history.
    pub fn update(&mut self) {
        let capacity = self.history[0].len();
        let Ok(chunk) = self.consumer.read_chunk(self.consumer.slots()) else {
            return;
        };
        let stride = self.history.len();
        let (first, second) = chunk.as_slices();
        let mut channel = 0;
        for &value in first.iter().chain(second) {
            if capacity > 0 {
                self.history[channel][self.write] = value;
            }
            channel += 1;
            if channel == stride {
                channel = 0;
                if capacity > 0 {
                    self.write = (self.write + 1) % capacity;
                    self.len = (self.len + 1).min(capacity);
                }
            }
        }
        chunk.commit_all();
    }

    /// Frames available for [`latest`](Self::latest), after the last update.
    pub fn available(&self) -> usize {
        self.len
    }

    /// Frames lost because the queue was full or a block was too large, since the start.
    pub fn dropped_frames(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Updates, then returns the most recent `num_frames` frames, or all available if fewer.
    pub fn latest(&mut self, num_frames: usize) -> LiveSnapshot {
        self.update();
        let n = num_frames.min(self.len);
        let capacity = self.history[0].len();
        let start = (self.write + capacity - n) % capacity.max(1);
        let read = |ring: &Vec<f32>, i: usize| f64::from(ring[(start + i) % capacity]);

        let num_inputs = self.inputs.len();
        let inputs = Array2::from_shape_fn((num_inputs, n), |(c, i)| read(&self.history[c], i));
        let reference = Array2::from_shape_fn((1, n), |(_, i)| read(&self.history[num_inputs], i));
        let mut inputs = TimeSignal::new(inputs, self.sample_rate).expect("valid sample rate");
        for (index, channel) in self.inputs.iter().enumerate() {
            inputs.set_channel_label(index, Some(&format!("input {channel}")));
        }
        let mut reference =
            TimeSignal::new(reference, self.sample_rate).expect("valid sample rate");
        reference.set_channel_label(0, Some("reference"));
        LiveSnapshot { inputs, reference }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use audio_blocks::{Interleaved, Planar};
    use num::Complex;

    use super::*;
    use crate::dsp::{TransferFunctionConfig, transfer_function};
    use crate::signal::measure::sim::{Path, SimDevice};
    use crate::signal::measure::{NoiseSource, NoiseSourceConfig};

    fn config(inputs: Vec<u16>, history: usize, queue: usize) -> LiveConfig {
        LiveConfig {
            inputs,
            history,
            queue,
            max_block: 64,
            ..Default::default()
        }
    }

    /// Input channel `c` counts up from `c * 1000`; the reference counts down from -1.
    fn process_counting(writer: &mut LiveWriter, num_channels: u16, start: usize, n: usize) {
        let channels: Vec<Vec<f32>> = (0..num_channels)
            .map(|c| {
                (start..start + n)
                    .map(|i| (c as usize * 1000 + i) as f32)
                    .collect()
            })
            .collect();
        let input = Planar::<f32>::from_slice(&channels);
        let mut output = Interleaved::<f32>::new(1, n);
        writer.process(&input, &mut output, |_, reference| {
            for (i, r) in reference.iter_mut().enumerate() {
                *r = -((start + i + 1) as f32);
            }
        });
    }

    #[test]
    fn inputs_and_reference_stay_aligned() {
        let (mut writer, mut reader) = live_channel(&config(vec![2, 0], 100, 100)).unwrap();
        assert_eq!(writer.num_input_channels(), 3);
        process_counting(&mut writer, 3, 0, 10);
        process_counting(&mut writer, 3, 10, 7);

        let snapshot = reader.latest(5);
        assert_eq!(reader.available(), 17);
        assert_eq!(snapshot.inputs.channel_label(0), Some("input 2"));
        assert_eq!(snapshot.inputs.channel_label(1), Some("input 0"));
        assert_eq!(
            snapshot.inputs.channel(0).to_vec(),
            [2012.0, 2013.0, 2014.0, 2015.0, 2016.0]
        );
        assert_eq!(
            snapshot.inputs.channel(1).to_vec(),
            [12.0, 13.0, 14.0, 15.0, 16.0]
        );
        assert_eq!(
            snapshot.reference.channel(0).to_vec(),
            [-13.0, -14.0, -15.0, -16.0, -17.0]
        );
        assert_eq!(reader.dropped_frames(), 0);
    }

    #[test]
    fn history_keeps_the_most_recent_frames() {
        let (mut writer, mut reader) = live_channel(&config(vec![0], 8, 100)).unwrap();
        for block in 0..5 {
            process_counting(&mut writer, 1, block * 5, 5);
        }
        let snapshot = reader.latest(100);
        assert_eq!(snapshot.inputs.num_time_steps(), 8);
        let expected: Vec<f64> = (17..25).map(f64::from).collect();
        assert_eq!(snapshot.inputs.channel(0).to_vec(), expected);
    }

    #[test]
    fn full_queue_drops_whole_blocks() {
        let (mut writer, mut reader) = live_channel(&config(vec![0], 100, 12)).unwrap();
        process_counting(&mut writer, 1, 0, 10);
        process_counting(&mut writer, 1, 10, 10); // doesn't fit
        assert_eq!(reader.dropped_frames(), 10);
        reader.update();
        process_counting(&mut writer, 1, 20, 10);
        let snapshot = reader.latest(100);
        let expected: Vec<f64> = (0..10).chain(20..30).map(f64::from).collect();
        assert_eq!(snapshot.inputs.channel(0).to_vec(), expected);
    }

    #[test]
    fn oversized_blocks_play_silence_and_are_dropped() {
        let (mut writer, reader) = live_channel(&config(vec![0], 100, 100)).unwrap();
        let input = Planar::<f32>::new(1, 65);
        let mut output = Interleaved::<f32>::new(1, 65);
        output.raw_data_mut().fill(1.0);
        writer.process(&input, &mut output, |_, _| unreachable!());
        assert!(output.raw_data().iter().all(|&s| s == 0.0));
        assert_eq!(reader.dropped_frames(), 65);
    }

    #[test]
    fn missing_inputs_record_silence() {
        let (mut writer, mut reader) = live_channel(&config(vec![0, 4], 100, 100)).unwrap();
        process_counting(&mut writer, 2, 0, 4);
        let snapshot = reader.latest(4);
        assert_eq!(snapshot.inputs.channel(0).to_vec(), [0.0, 1.0, 2.0, 3.0]);
        assert!(snapshot.inputs.channel(1).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn empty_until_something_is_written() {
        let (_writer, mut reader) = live_channel(&config(vec![0], 100, 100)).unwrap();
        let snapshot = reader.latest(10);
        assert_eq!(snapshot.inputs.num_time_steps(), 0);
        assert_eq!(snapshot.reference.num_time_steps(), 0);
    }

    #[test]
    fn invalid_configs() {
        let invalid = |config: LiveConfig| live_channel(&config).err().unwrap();
        assert_eq!(invalid(config(vec![], 10, 10)), RecordError::NoInputs);
        assert_eq!(
            invalid(config(vec![u16::MAX], 10, 10)),
            RecordError::ChannelOutOfRange(u16::MAX)
        );
        assert_eq!(
            invalid(LiveConfig {
                sample_rate: 0.0,
                ..Default::default()
            }),
            RecordError::InvalidSampleRate(0.0)
        );
    }

    /// Two speakers play the same pink noise at once with different delays and filters, two
    /// mics record. The transfer function from the reference to each mic is the sum of the
    /// paths into it.
    #[test]
    fn transfer_function_of_periodic_pink_noise_through_a_simulated_room() {
        let period = 2048;
        let mut noise = NoiseSource::new(&NoiseSourceConfig {
            outputs: vec![0, 1],
            period,
            gain: 0.5,
            seed: Some(3),
            ..Default::default()
        })
        .unwrap();
        let paths = [
            Path::new(0, 0, 300, &[1.0, 0.5]),
            Path::new(1, 0, 450, &[-0.25]),
            Path::new(0, 1, 380, &[0.75]),
            Path::new(1, 1, 256, &[0.5, 0.0, 0.25]),
        ];
        let (mut writer, mut reader) = live_channel(&LiveConfig {
            inputs: vec![0, 1],
            history: 4 * period,
            queue: 1024,
            max_block: 256,
            ..Default::default()
        })
        .unwrap();
        let mut device = SimDevice::new(&paths, 2, noise.num_output_channels());
        // long enough for the room to settle, with block sizes that don't divide the period
        for block in 0..200 {
            let n = [256, 100, 177][block % 3];
            device.callback(n, |input, output| {
                writer.process(input, output, |output, reference| {
                    noise.fill(output, reference)
                });
            });
            // like a UI, read while it plays
            reader.update();
        }
        assert_eq!(reader.dropped_frames(), 0);

        let snapshot = reader.latest(3 * period);
        let config = TransferFunctionConfig::default()
            .with_segment_len(period)
            .with_regularization(0.0);
        let h = transfer_function(&snapshot.inputs, &snapshot.reference, &config).unwrap();
        assert_eq!(h.channel_label(1), Some("input 1"));

        for mic in 0..2u16 {
            for (bin, value) in h.channel(mic as usize).iter().enumerate() {
                let expected: Complex<f64> = paths
                    .iter()
                    .filter(|p| p.input == mic)
                    .flat_map(|p| {
                        p.taps.iter().enumerate().map(move |(k, tap)| {
                            let phase = std::f64::consts::TAU * bin as f64 * (p.delay + k) as f64
                                / period as f64;
                            Complex::from_polar(*tap, -phase)
                        })
                    })
                    .sum();
                assert_abs_diff_eq!(value.re, expected.re, epsilon = 1e-3);
                assert_abs_diff_eq!(value.im, expected.im, epsilon = 1e-3);
            }
        }
    }
}
