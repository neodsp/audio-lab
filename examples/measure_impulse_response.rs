//! Measure the impulse response from each speaker to each microphone with sweeps.
//!
//! Every speaker plays a sweep, one after another, while all microphones record. All of it runs
//! in one audio stream, so the device latency is the same for every speaker and the delays
//! between them stay comparable. The result is one impulse response per speaker and mic: the
//! example prints where each one starts and plots the frequency responses, magnitude and
//! phase, with adjustable smoothing. Each phase is relative to the peak of its impulse response,
//! so the delays are taken out; compare them with the printed start times.
//!
//! ```sh
//! # speakers on outputs 0 and 1, mics on inputs 0 and 1
//! cargo run --release --example measure_impulse_response -- --outputs 0,1 --inputs 0,1
//! ```
//!
//! Options (all optional):
//!
//! - `--outputs 0,1`: output channels of the speakers, measured one after another (default `0`)
//! - `--inputs 0,1`: input channels of the microphones, all recorded at once (default `0`)
//! - `--level -12`: sweep peak level in dBFS (default `-12`). Start low and turn it up.
//! - `--input-device NAME`, `--output-device NAME`: device names or parts of them
//!
//! The recorded impulse responses include the device latency (100 ms or more with cpal), so the
//! printed start times are only meaningful relative to each other. Wire an output back to an
//! input and measure it as one more speaker to get the latency itself.

use std::f64::consts::TAU;
use std::time::Duration;

use audio_lab::dsp::{DeconvolutionConfig, deconvolve, window_and_trim};
use audio_lab::plot::freq::{FreqPlotOptions, show_magnitude_and_phase};
use audio_lab::signal::measure::{SequenceConfig, Step};
use audio_lab::signal::playback::{DeviceConfig, play_and_record};
use audio_lab::signal::{FreqSignal, SweepConfig, SweepType, TimeSignal, generate_sweep};
use num::complex::Complex64;

const SAMPLE_RATE: f64 = 48_000.0;
const FREQUENCY_RANGE: std::ops::Range<f64> = 20.0..20_000.0;
/// 2^16 samples, about 1.4 s at 48 kHz. Longer sweeps give more signal-to-noise.
const SWEEP_LEN: usize = 1 << 16;
/// How much of each impulse response to keep, after its start.
const IR_LENGTH_SECONDS: f64 = 0.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;

    let sweep = generate_sweep(
        SWEEP_LEN,
        FREQUENCY_RANGE,
        SweepConfig {
            amplitude: 10f64.powf(args.level_db / 20.0),
            sample_rate: SAMPLE_RATE,
            num_channels: 1,
            // a full cycle of 20 Hz, so the sweep doesn't start with a click
            fade_in: (0.05 * SAMPLE_RATE) as usize,
            fade_out: 90,
            sweep_type: SweepType::Exponential,
        },
    )?;

    // one step per speaker: they play one after another
    let steps = args
        .outputs
        .iter()
        .map(|&output| Step::new(vec![output], sweep.clone()))
        .collect();
    let config = SequenceConfig {
        inputs: args.inputs.clone(),
        warmup: Duration::from_millis(300),
        // room for the reverb tail and the device latency
        tail: Duration::from_millis(1500),
    };
    let device = DeviceConfig {
        input_device: args.input_device.clone(),
        output_device: args.output_device.clone(),
        ..Default::default()
    };

    let seconds = args.outputs.len() as f64 * (sweep.length_in_seconds() + 1.5);
    println!(
        "Measuring {} speaker(s) with {} mic(s), about {seconds:.0} s...",
        args.outputs.len(),
        args.inputs.len()
    );
    let recordings = play_and_record(steps, &config, &device)?;

    let deconvolution = DeconvolutionConfig::default().with_frequency_range(FREQUENCY_RANGE);
    let mut responses = Vec::new();

    println!("\nspeaker  mic  peak (dBFS)  IR start (ms)");
    for (&output, recording) in args.outputs.iter().zip(&recordings) {
        for (index, &input) in args.inputs.iter().enumerate() {
            let label = format!("out {output} -> in {input}");
            let mic = channel(recording, index);
            let peak = mic.channel(0).iter().fold(0.0f64, |m, s| m.max(s.abs()));
            if peak >= 1.0 {
                println!("{label}: the recording clips, lower --level or the input gain");
            }

            let ir = deconvolve(&mic, &sweep, &deconvolution)?;
            let (ir, starts) = match window_and_trim(&ir, IR_LENGTH_SECONDS) {
                Ok(trimmed) => trimmed,
                Err(error) => {
                    println!("{label}: no impulse response found ({error})");
                    continue;
                }
            };
            println!(
                "{output:>7}  {input:>3}  {:>11.1}  {:>13.2}",
                20.0 * peak.log10(),
                starts[0] as f64 / SAMPLE_RATE * 1000.0
            );

            let mut response = response_from_peak(&ir);
            response.set_comment(Some(&label));
            responses.push(response);
        }
    }

    if responses.is_empty() {
        return Err("no impulse response found, check the routing and levels".into());
    }
    let responses: Vec<_> = responses.iter().collect();
    let options = FreqPlotOptions {
        // only the range the sweep covered: outside it there is nothing to measure
        freq_range: Some((FREQUENCY_RANGE.start, FREQUENCY_RANGE.end)),
        // 1/3 octave to start with, adjustable in the window
        smoothing: Some(3.0),
        ..Default::default()
    };
    show_magnitude_and_phase("Frequency responses", &responses, options)?;
    Ok(())
}

/// The frequency response of `ir`, with the phase relative to the impulse response's peak: the
/// delay up to it would turn the phase so quickly that it wraps every few Hz.
fn response_from_peak(ir: &TimeSignal) -> FreqSignal {
    let peak = ir
        .channel(0)
        .iter()
        .enumerate()
        .fold(
            (0, 0.0),
            |best, (i, s)| if s.abs() > best.1 { (i, s.abs()) } else { best },
        )
        .0;
    let mut response = ir.to_freq();
    let freqs = response.freq_bins().to_owned();
    for (h, f) in response.channel_mut(0).iter_mut().zip(freqs) {
        *h *= Complex64::from_polar(1.0, TAU * f * peak as f64 / SAMPLE_RATE);
    }
    response
}

/// One channel of `signal` as a mono signal.
fn channel(signal: &TimeSignal, index: usize) -> TimeSignal {
    let data = signal
        .channel(index)
        .to_owned()
        .insert_axis(ndarray::Axis(0));
    TimeSignal::new(data, signal.sample_rate()).expect("valid sample rate")
}

struct Args {
    outputs: Vec<u16>,
    inputs: Vec<u16>,
    level_db: f64,
    input_device: Option<String>,
    output_device: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            outputs: vec![0],
            inputs: vec![0],
            level_db: -12.0,
            input_device: None,
            output_device: None,
        };
        let mut iter = std::env::args().skip(1);
        while let Some(flag) = iter.next() {
            let value = iter.next().ok_or(format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--outputs" => args.outputs = channels(&value)?,
                "--inputs" => args.inputs = channels(&value)?,
                "--level" => {
                    args.level_db = value.parse().map_err(|_| format!("bad level {value}"))?;
                    if args.level_db > 0.0 {
                        return Err("--level must be <= 0 dBFS".into());
                    }
                }
                "--input-device" => args.input_device = Some(value),
                "--output-device" => args.output_device = Some(value),
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        Ok(args)
    }
}

fn channels(list: &str) -> Result<Vec<u16>, String> {
    list.split(',')
        .map(|c| c.trim().parse().map_err(|_| format!("bad channel {c}")))
        .collect()
}
