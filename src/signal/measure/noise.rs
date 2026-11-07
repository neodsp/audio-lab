use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use audio_blocks::AudioBlockMut;
use ndarray::Array2;

use crate::signal::{NoiseConfig, NoiseError, SignalError, Spectrum, TimeSignal, generate_noise};

#[derive(Debug, thiserror::Error)]
pub enum NoiseSourceError {
    #[error("The period must be > 0")]
    PeriodZero,
    #[error(transparent)]
    Noise(#[from] NoiseError),
    #[error(transparent)]
    Signal(#[from] SignalError),
}

#[derive(Debug, Clone)]
pub struct NoiseSourceConfig {
    /// Output channels to play the noise on, all with the same noise.
    pub outputs: Vec<u16>,
    /// Pink by default: white noise has most of its power in the treble, which stresses
    /// tweeters, while pink puts more energy into the bass, where room noise is highest.
    pub spectrum: Spectrum,
    /// Length of the looped noise buffer. Analyze segments of this length (see
    /// [`transfer_function`](fn@crate::dsp::transfer_function)) to get exact results without a
    /// window. It must be longer than the device latency plus the room's impulse response:
    /// the default 16384 is 341 ms at 48 kHz.
    pub period: usize,
    /// Peak level at the start, see [`NoiseControls::set_gain`].
    pub gain: f32,
    pub sample_rate: f64,
    /// Seed for the noise, `None` for a random one.
    pub seed: Option<u64>,
}

impl Default for NoiseSourceConfig {
    fn default() -> Self {
        Self {
            outputs: vec![0],
            spectrum: Spectrum::Pink,
            period: 16_384,
            gain: 0.25,
            sample_rate: 48_000.0,
            seed: None,
        }
    }
}

/// Plays looped noise on some outputs and writes it as the reference of a live measurement,
/// see [`LiveWriter::process`](crate::signal::measure::LiveWriter::process).
///
/// The gain and on/off can change while it plays, through [`NoiseControls`]. Changes ramp
/// over one block to avoid clicks. To process the noise (e.g. a crossover or a delay per
/// output), call [`fill`](Self::fill) and then change the output block: the reference stays the
/// raw noise, so the transfer function covers the processing too.
#[derive(Debug)]
pub struct NoiseSource {
    buffer: Vec<f32>,
    position: usize,
    outputs: Vec<u16>,
    sample_rate: f64,
    controls: NoiseControls,
    /// The gain at the end of the last block, where the next ramp starts.
    current_gain: f32,
}

impl NoiseSource {
    pub fn new(config: &NoiseSourceConfig) -> Result<Self, NoiseSourceError> {
        if config.period == 0 {
            return Err(NoiseSourceError::PeriodZero);
        }
        // the pink filter works on the whole buffer in the frequency domain, so the noise is
        // periodic and the loop has no seam
        let noise = generate_noise(
            config.period,
            NoiseConfig {
                spectrum: config.spectrum,
                amplitude: 1.0,
                sample_rate: config.sample_rate,
                num_channels: 1,
                seed: config.seed,
            },
        )?;
        let controls = NoiseControls(Arc::new(ControlState {
            gain: AtomicU32::new(config.gain.to_bits()),
            enabled: AtomicBool::new(true),
        }));
        Ok(Self {
            buffer: noise.channel(0).iter().map(|&s| s as f32).collect(),
            position: 0,
            outputs: config.outputs.clone(),
            sample_rate: config.sample_rate,
            current_gain: config.gain,
            controls,
        })
    }

    /// A handle to change the gain and switch the noise on and off while it plays.
    pub fn controls(&self) -> NoiseControls {
        self.controls.clone()
    }

    pub fn period(&self) -> usize {
        self.buffer.len()
    }

    /// One period of the noise, at a peak level of 1.
    pub fn period_signal(&self) -> TimeSignal {
        let data =
            Array2::from_shape_fn((1, self.buffer.len()), |(_, i)| f64::from(self.buffer[i]));
        TimeSignal::new(data, self.sample_rate).expect("the sample rate was valid for the noise")
    }

    /// The output channels the stream needs: the highest output plus one.
    pub fn num_output_channels(&self) -> u16 {
        self.outputs.iter().max().map_or(0, |&max| max + 1)
    }

    /// Writes the next block of noise to the outputs and to `reference`. Clears every other
    /// output channel. Writes `min(output frames, reference.len())` frames. Real-time safe.
    pub fn fill(&mut self, output: &mut impl AudioBlockMut<f32>, reference: &mut [f32]) {
        for channel in 0..output.num_channels() {
            output.channel_iter_mut(channel).for_each(|s| *s = 0.0);
        }
        let num_frames = output.num_frames().min(reference.len());
        let target = if self.controls.is_enabled() {
            self.controls.gain()
        } else {
            0.0
        };
        let start = self.current_gain;
        let step = (target - start) / num_frames.max(1) as f32;

        for (frame, reference) in reference[..num_frames].iter_mut().enumerate() {
            let gain = start + step * (frame + 1) as f32;
            let sample = self.buffer[self.position] * gain;
            self.position = (self.position + 1) % self.buffer.len();
            *reference = sample;
            for &channel in &self.outputs {
                if channel < output.num_channels() {
                    *output.sample_mut(channel, frame) = sample;
                }
            }
        }
        if num_frames > 0 {
            self.current_gain = target;
        }
    }
}

/// Changes a [`NoiseSource`] while it plays, from any thread.
#[derive(Debug, Clone)]
pub struct NoiseControls(Arc<ControlState>);

#[derive(Debug)]
struct ControlState {
    gain: AtomicU32,
    enabled: AtomicBool,
}

impl NoiseControls {
    /// Linear gain; the noise peaks at this level.
    pub fn set_gain(&self, gain: f32) {
        self.0.gain.store(gain.to_bits(), Ordering::Relaxed);
    }

    pub fn gain(&self) -> f32 {
        f32::from_bits(self.0.gain.load(Ordering::Relaxed))
    }

    /// Off plays silence (after a ramp down) and writes a silent reference.
    pub fn set_enabled(&self, enabled: bool) {
        self.0.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.0.enabled.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use audio_blocks::{AudioBlock, Interleaved};

    use super::*;

    fn source(config: NoiseSourceConfig) -> NoiseSource {
        NoiseSource::new(&NoiseSourceConfig {
            seed: Some(1),
            ..config
        })
        .unwrap()
    }

    fn run(
        source: &mut NoiseSource,
        num_channels: u16,
        num_frames: usize,
    ) -> (Interleaved<f32>, Vec<f32>) {
        let mut output = Interleaved::<f32>::new(num_channels, num_frames);
        let mut reference = vec![0.0; num_frames];
        source.fill(&mut output, &mut reference);
        (output, reference)
    }

    #[test]
    fn loops_the_period_on_every_output() {
        let mut noise = source(NoiseSourceConfig {
            outputs: vec![2, 0],
            period: 100,
            gain: 1.0,
            ..Default::default()
        });
        assert_eq!(noise.num_output_channels(), 3);
        let period = noise.period_signal();
        assert_eq!(period.num_time_steps(), 100);
        assert_abs_diff_eq!(
            period.data().max_abs_per_channel()[0].1,
            1.0,
            epsilon = 1e-6
        );

        let (output, reference) = run(&mut noise, 4, 250);
        for (i, &r) in reference.iter().enumerate() {
            assert_abs_diff_eq!(f64::from(r), period.channel(0)[i % 100], epsilon = 1e-6);
            assert_eq!(output.sample(0, i), r);
            assert_eq!(output.sample(2, i), r);
            assert_eq!(output.sample(1, i), 0.0);
            assert_eq!(output.sample(3, i), 0.0);
        }
    }

    #[test]
    fn gain_changes_ramp_over_one_block() {
        let mut noise = source(NoiseSourceConfig {
            period: 64,
            gain: 1.0,
            ..Default::default()
        });
        let period = noise.period_signal();
        let controls = noise.controls();
        controls.set_gain(0.5);
        assert_eq!(controls.gain(), 0.5);

        let (_, ramp) = run(&mut noise, 1, 10);
        for (i, &r) in ramp.iter().enumerate() {
            let gain = 1.0 - 0.05 * (i + 1) as f64;
            assert_abs_diff_eq!(f64::from(r), gain * period.channel(0)[i], epsilon = 1e-6);
        }
        let (_, steady) = run(&mut noise, 1, 10);
        for (i, &r) in steady.iter().enumerate() {
            assert_abs_diff_eq!(
                f64::from(r),
                0.5 * period.channel(0)[10 + i],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn disabled_noise_is_silent_after_the_ramp() {
        let mut noise = source(NoiseSourceConfig::default());
        let controls = noise.controls();
        controls.set_enabled(false);
        assert!(!controls.is_enabled());
        let (_, ramp) = run(&mut noise, 1, 32);
        assert_eq!(*ramp.last().unwrap(), 0.0);
        let (output, reference) = run(&mut noise, 1, 32);
        assert!(reference.iter().all(|&s| s == 0.0));
        assert!(output.raw_data().iter().all(|&s| s == 0.0));

        controls.set_enabled(true);
        let (_, reference) = run(&mut noise, 1, 32);
        assert!(reference.iter().any(|&s| s != 0.0));
    }

    #[test]
    fn shorter_reference_limits_the_frames() {
        let mut noise = source(NoiseSourceConfig::default());
        let mut output = Interleaved::<f32>::new(1, 16);
        output.raw_data_mut().fill(9.0);
        let mut reference = vec![0.0; 4];
        noise.fill(&mut output, &mut reference);
        assert!(output.channel_iter(0).skip(4).all(|&s| s == 0.0));
        assert_eq!(noise.position, 4);
    }

    #[test]
    fn pink_by_default_with_equal_power_per_octave() {
        let noise = source(NoiseSourceConfig::default());
        let spectrum = noise.period_signal().into_freq();
        let df = 48_000.0 / noise.period() as f64;
        let band_power = |low: f64| -> f64 {
            let bins = (low / df) as usize..(2.0 * low / df) as usize;
            spectrum
                .channel(0)
                .slice(ndarray::s![bins])
                .iter()
                .map(|c| c.norm_sqr())
                .sum()
        };
        let ratio = band_power(200.0) / band_power(6400.0);
        assert!((0.7..1.4).contains(&ratio), "ratio {ratio}");
    }

    #[test]
    fn period_must_not_be_zero() {
        let config = NoiseSourceConfig {
            period: 0,
            ..Default::default()
        };
        assert!(matches!(
            NoiseSource::new(&config),
            Err(NoiseSourceError::PeriodZero)
        ));
    }
}
