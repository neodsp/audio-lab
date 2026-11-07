//! Play signals on an audio device, and play while recording for measurements.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use audio_host::{
    AudioBackend, AudioBlock, AudioBlockMut, AudioHost, BlockMut, Config, MAX_NUM_FRAMES,
    StreamError, StreamStatus,
};

use crate::signal::TimeSignal;
use crate::signal::measure::{
    LiveConfig, LiveReader, LiveSnapshot, RecordError, Sequence, SequenceConfig, Step, live_channel,
};

#[derive(Debug, thiserror::Error)]
pub enum PlaybackError {
    #[error(transparent)]
    Record(#[from] RecordError),
    #[error(transparent)]
    Host(#[from] audio_host::Error),
    #[error("Audio devices need a whole-number sample rate, got {0} Hz")]
    FractionalSampleRate(f64),
    #[error("{0} dropouts while recording, the recording has gaps")]
    Dropouts(u64),
    #[error(transparent)]
    Stream(#[from] StreamError),
    #[error("The audio device stopped calling back before the recording finished")]
    Timeout,
}

/// Which devices to use and how to open the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceConfig {
    /// Input device name, `None` for the system default. An exact match wins, otherwise the
    /// first device whose name contains it.
    pub input_device: Option<String>,
    /// Output device name, `None` for the system default. Matched like `input_device`.
    pub output_device: Option<String>,
    /// Requested block size. The device may use another one.
    pub num_frames: usize,
    /// Input latency that the cpal backend adds to duplex streams to absorb timing differences
    /// between input and output. Lower values can cause dropouts.
    pub duplex_latency_ms: u32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        let host = Config::default();
        Self {
            input_device: None,
            output_device: None,
            num_frames: host.num_frames,
            duplex_latency_ms: host.duplex_latency_ms,
        }
    }
}

impl DeviceConfig {
    fn open(&self) -> Result<AudioHost, PlaybackError> {
        let mut host = AudioHost::new()?;
        if let Some(name) = &self.input_device {
            host.set_input(name)?;
        }
        if let Some(name) = &self.output_device {
            host.set_output(name)?;
        }
        Ok(host)
    }

    fn stream_config(
        &self,
        sample_rate: f64,
        num_input_channels: u16,
        num_output_channels: u16,
    ) -> Result<Config, PlaybackError> {
        Ok(Config::default()
            .with_input_channels(num_input_channels)
            .with_output_channels(num_output_channels)
            .with_sample_rate(whole_sample_rate(sample_rate)?)
            .with_num_frames(self.num_frames)
            .with_duplex_latency_ms(self.duplex_latency_ms))
    }
}

fn whole_sample_rate(sample_rate: f64) -> Result<u32, PlaybackError> {
    if sample_rate.fract() != 0.0 || sample_rate < 1.0 || sample_rate > f64::from(u32::MAX) {
        return Err(PlaybackError::FractionalSampleRate(sample_rate));
    }
    Ok(sample_rate as u32)
}

/// Plays `steps` one after another and records `config.inputs` the whole time, in one duplex
/// stream. Returns one recording per step, with one channel per input. Blocks until done.
///
/// See [`Sequence`] for the timeline and how to measure several speakers. The stream runs at the
/// steps' sample rate; if the device can't, the backend's error is returned instead of
/// resampling. Any dropout fails the measurement, since the recording would have gaps.
///
/// ```no_run
/// use audio_lab::dsp::{DeconvolutionConfig, deconvolve};
/// use audio_lab::signal::measure::{SequenceConfig, Step};
/// use audio_lab::signal::playback::{DeviceConfig, play_and_record};
/// use audio_lab::signal::{SweepConfig, generate_sweep};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let sweep = generate_sweep(1 << 18, 20.0..20_000.0, SweepConfig::default())?;
/// let steps = vec![Step::new(vec![0], sweep.clone())];
/// let recordings = play_and_record(steps, &SequenceConfig::default(), &DeviceConfig::default())?;
/// let impulse_response = deconvolve(&recordings[0], &sweep, &DeconvolutionConfig::default())?;
/// # Ok(())
/// # }
/// ```
pub fn play_and_record(
    steps: Vec<Step>,
    config: &SequenceConfig,
    device: &DeviceConfig,
) -> Result<Vec<TimeSignal>, PlaybackError> {
    let sequence = Sequence::new(steps, config)?;
    let stream_config = device.stream_config(
        sequence.sample_rate(),
        sequence.num_input_channels(),
        sequence.num_output_channels(),
    )?;
    let duration = Duration::from_secs_f64(sequence.total_frames() as f64 / sequence.sample_rate());
    let mut host = device.open()?;

    // the callback owns the sequence and hands it back once it is done
    let (done_tx, done_rx) = mpsc::sync_channel(1);
    let mut running = Some(sequence);
    host.start(stream_config, move |input, mut output| {
        let Some(sequence) = running.as_mut() else {
            for channel in 0..output.num_channels() {
                output.channel_iter_mut(channel).for_each(|s| *s = 0.0);
            }
            return;
        };
        sequence.process(&input, &mut output);
        if sequence.is_done() {
            let _ = done_tx.try_send(running.take());
        }
    })?;

    // a generous deadline: the device may start late, but a stuck stream must not hang forever
    let deadline = Instant::now() + duration * 2 + Duration::from_secs(5);
    let result = loop {
        match done_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Some(sequence)) => break Ok(sequence),
            Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                break Err(PlaybackError::Timeout);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(error) = host.status().error {
                    break Err(error.into());
                }
                if Instant::now() > deadline {
                    break Err(PlaybackError::Timeout);
                }
            }
        }
    };
    let status = host.status();
    host.stop()?;

    let sequence = result?;
    if status.xruns > 0 {
        return Err(PlaybackError::Dropouts(status.xruns));
    }
    Ok(sequence.into_recordings())
}

#[derive(Debug, Clone, PartialEq)]
pub struct LiveSessionConfig {
    /// Input channels to record, one per mic.
    pub inputs: Vec<u16>,
    /// Output channels the stream opens, e.g.
    /// [`NoiseSource::num_output_channels`](crate::signal::measure::NoiseSource::num_output_channels).
    pub num_output_channels: u16,
    pub sample_rate: f64,
    /// How much is kept for [`LiveSession::latest`]. Also how long the audio thread can run
    /// without `latest` being called before frames are dropped.
    pub history: Duration,
}

impl Default for LiveSessionConfig {
    fn default() -> Self {
        Self {
            inputs: vec![0],
            num_output_channels: 2,
            sample_rate: 48_000.0,
            history: Duration::from_secs(3),
        }
    }
}

/// Plays a source and records inputs and the source's reference continuously, e.g. pink noise
/// on several speakers for a live transfer function. Runs until [`stop`](Self::stop)ped or
/// dropped.
///
/// ```no_run
/// use audio_lab::dsp::{TransferFunctionConfig, transfer_function};
/// use audio_lab::signal::measure::{NoiseSource, NoiseSourceConfig};
/// use audio_lab::signal::playback::{DeviceConfig, LiveSession, LiveSessionConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut noise = NoiseSource::new(&NoiseSourceConfig {
///     outputs: vec![0, 1],
///     ..Default::default()
/// })?;
/// let period = noise.period();
/// let config = LiveSessionConfig {
///     inputs: vec![0, 1],
///     num_output_channels: noise.num_output_channels(),
///     ..Default::default()
/// };
/// let mut session = LiveSession::start(
///     move |output, reference| noise.fill(output, reference),
///     &config,
///     &DeviceConfig::default(),
/// )?;
///
/// std::thread::sleep(std::time::Duration::from_secs(2));
/// let snapshot = session.latest(4 * period);
/// let tf_config = TransferFunctionConfig::default().with_segment_len(period);
/// let response = transfer_function(&snapshot.inputs, &snapshot.reference, &tf_config)?;
/// session.stop()?;
/// # Ok(())
/// # }
/// ```
pub struct LiveSession {
    host: AudioHost,
    reader: LiveReader,
}

impl LiveSession {
    /// Starts the stream. `source` runs on the audio thread for every block: it fills the
    /// output block and the reference (one sample per frame, the signal the measurement is
    /// relative to), see
    /// [`LiveWriter::process`](crate::signal::measure::LiveWriter::process). It must be real-time safe.
    pub fn start(
        mut source: impl FnMut(&mut BlockMut<'_>, &mut [f32]) + Send + 'static,
        config: &LiveSessionConfig,
        device: &DeviceConfig,
    ) -> Result<Self, PlaybackError> {
        let history = (config.history.as_secs_f64() * config.sample_rate).ceil() as usize;
        let (mut writer, reader) = live_channel(&LiveConfig {
            inputs: config.inputs.clone(),
            sample_rate: config.sample_rate,
            history,
            queue: history,
            max_block: MAX_NUM_FRAMES,
        })?;
        let stream_config = device.stream_config(
            config.sample_rate,
            writer.num_input_channels(),
            config.num_output_channels,
        )?;
        let mut host = device.open()?;
        host.start(stream_config, move |input, mut output| {
            writer.process(&input, &mut output, |output, reference| {
                source(output, reference)
            });
        })?;
        Ok(Self { host, reader })
    }

    /// The most recent `num_frames` frames of every input and of the reference, or all that
    /// were recorded if fewer. Call it regularly, at least once per
    /// [`history`](LiveSessionConfig::history).
    pub fn latest(&mut self, num_frames: usize) -> LiveSnapshot {
        self.reader.latest(num_frames)
    }

    /// Frames lost because `latest` wasn't called often enough, since the start.
    pub fn dropped_frames(&self) -> u64 {
        self.reader.dropped_frames()
    }

    /// Whether the stream still runs, why it stopped and how many dropouts the device had.
    pub fn status(&self) -> StreamStatus {
        self.host.status()
    }

    pub fn stop(mut self) -> Result<(), PlaybackError> {
        self.host.stop()?;
        Ok(())
    }
}

pub fn play(signal: &TimeSignal) -> Result<(), audio_host::Error> {
    let num_output_channels = signal.num_channels() as u16;
    let sample_rate = signal.sample_rate().round() as u32;
    let num_frames = 512;
    let interleaved: Vec<f32> = signal.time_data().t().iter().map(|&s| s as f32).collect();
    let total_frames = signal.num_time_steps();
    let mut next_frame = 0usize;

    let mut host = AudioHost::new()?;
    host.start(
        Config::default()
            .with_input_channels(0)
            .with_output_channels(num_output_channels)
            .with_sample_rate(sample_rate)
            .with_num_frames(num_frames),
        move |_input, mut output| {
            for frame in 0..output.num_frames() {
                let src_frame = next_frame;
                next_frame += 1;

                for ch in 0..output.num_channels() as usize {
                    let sample = if src_frame < total_frames {
                        interleaved[src_frame * num_output_channels as usize + ch]
                    } else {
                        0.0
                    };
                    *output.sample_mut(ch as u16, frame) = sample;
                }
            }
        },
    )?;

    std::thread::sleep(Duration::from_secs_f64(
        signal.length_in_seconds() + num_frames as f64 / sample_rate as f64,
    ));
    host.stop()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn play_audio() {
        use std::f64::consts::TAU;

        let sample_rate = 48_000.0;
        let seconds = 2.0;
        let num_time_steps = (sample_rate * seconds) as usize;
        let frequency = 440.0;

        let data = ndarray::Array2::from_shape_fn((2, num_time_steps), |(_, frame)| {
            let t = frame as f64 / sample_rate;
            0.2 * (TAU * frequency * t).sin()
        });

        let signal = TimeSignal::new(data, sample_rate).unwrap();
        play(&signal).unwrap();
    }

    /// Needs a device with a loopback (output 0 wired to input 0) or a speaker and mic.
    #[test]
    #[ignore]
    fn play_and_record_a_sweep() {
        use crate::dsp::{DeconvolutionConfig, deconvolve};
        use crate::signal::{SweepConfig, generate_sweep};

        let sweep = generate_sweep(1 << 16, 20.0..20_000.0, SweepConfig::default()).unwrap();
        let steps = vec![Step::new(vec![0], sweep.clone())];
        let recordings =
            play_and_record(steps, &SequenceConfig::default(), &DeviceConfig::default()).unwrap();
        assert_eq!(recordings.len(), 1);
        assert_eq!(
            recordings[0].num_time_steps(),
            sweep.num_time_steps() + 48_000
        );
        let ir = deconvolve(&recordings[0], &sweep, &DeconvolutionConfig::default()).unwrap();
        println!("{ir}");
    }

    /// Needs a device with a loopback (output 0 wired to input 0) or a speaker and mic.
    #[test]
    #[ignore]
    fn live_session_records_noise() {
        use crate::signal::measure::{NoiseSource, NoiseSourceConfig};

        let mut noise = NoiseSource::new(&NoiseSourceConfig::default()).unwrap();
        let period = noise.period();
        let config = LiveSessionConfig {
            num_output_channels: noise.num_output_channels(),
            ..Default::default()
        };
        let mut session = LiveSession::start(
            move |output, reference| noise.fill(output, reference),
            &config,
            &DeviceConfig::default(),
        )
        .unwrap();
        std::thread::sleep(Duration::from_secs(2));
        let snapshot = session.latest(2 * period);
        assert_eq!(snapshot.inputs.num_time_steps(), 2 * period);
        assert!(snapshot.reference.channel(0).iter().any(|&s| s != 0.0));
        assert_eq!(session.dropped_frames(), 0);
        session.stop().unwrap();
    }

    #[test]
    fn invalid_live_config_fails_before_opening_a_device() {
        let config = LiveSessionConfig {
            inputs: vec![],
            ..Default::default()
        };
        let result = LiveSession::start(|_, _| {}, &config, &DeviceConfig::default());
        assert!(matches!(
            result,
            Err(PlaybackError::Record(RecordError::NoInputs))
        ));
    }

    #[test]
    fn sample_rate_must_be_whole() {
        assert_eq!(whole_sample_rate(48_000.0).unwrap(), 48_000);
        for rate in [44_100.5, 0.0, 0.5, f64::NAN, f64::INFINITY, 1e12] {
            assert!(matches!(
                whole_sample_rate(rate),
                Err(PlaybackError::FractionalSampleRate(_))
            ));
        }
    }

    #[test]
    fn stream_config_uses_the_device_settings() {
        let device = DeviceConfig {
            num_frames: 128,
            duplex_latency_ms: 40,
            ..Default::default()
        };
        let config = device.stream_config(44_100.0, 3, 2).unwrap();
        assert_eq!(config.num_input_channels, 3);
        assert_eq!(config.num_output_channels, 2);
        assert_eq!(config.sample_rate, 44_100);
        assert_eq!(config.num_frames, 128);
        assert_eq!(config.duplex_latency_ms, 40);
    }

    #[test]
    fn invalid_steps_fail_before_opening_a_device() {
        let result = play_and_record(vec![], &SequenceConfig::default(), &DeviceConfig::default());
        assert!(matches!(
            result,
            Err(PlaybackError::Record(RecordError::NoSteps))
        ));
    }
}
