use ndarray::s;

use audio_signal::signal::{SignalError, TimeSignal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    After,
    Before,
}

pub fn pad_zeros(
    signal: &TimeSignal,
    pad_width: usize,
    mode: PadMode,
) -> Result<TimeSignal, SignalError> {
    let mut padded = TimeSignal::zeros(
        signal.num_channels(),
        signal.num_time_steps() + pad_width,
        signal.sample_rate(),
    )?;

    for (mut padded_channel, channel) in padded.channel_iter_mut().zip(signal.channel_iter()) {
        match mode {
            PadMode::After => {
                padded_channel
                    .slice_mut(s![..signal.num_time_steps()])
                    .assign(&channel);
            }
            PadMode::Before => {
                padded_channel.slice_mut(s![pad_width..]).assign(&channel);
            }
        }
    }

    Ok(padded)
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;

    #[test]
    fn pad_zeros_before_and_after() {
        let signal = TimeSignal::new(arr2(&[[0.0, 1.0], [2.0, 3.0]]), 8.0).unwrap();

        let after = pad_zeros(&signal, 2, PadMode::After).unwrap();
        assert_eq!(after.channel(0), arr1(&[0.0, 1.0, 0.0, 0.0]));
        assert_eq!(after.channel(1), arr1(&[2.0, 3.0, 0.0, 0.0]));

        let before = pad_zeros(&signal, 2, PadMode::Before).unwrap();
        assert_eq!(before.channel(0), arr1(&[0.0, 0.0, 0.0, 1.0]));
        assert_eq!(before.channel(1), arr1(&[0.0, 0.0, 2.0, 3.0]));
    }
}
