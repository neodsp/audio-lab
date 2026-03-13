use audio_blocks::{AudioBlock, Planar};
use ndarray::Array2;

use crate::signal::{SignalError, TimeSignal};

#[derive(Debug, thiserror::Error)]
pub enum BlockAdapterError {
    #[error("Frames per block must be > 0")]
    FramesPerBlockZero,
    #[error("Sample Rate must be > 0")]
    SampleRateZeroOrNeg,
    #[error("At least one block is required")]
    EmptyBlockList,
    #[error("All blocks must have the same number of channels")]
    ChannelMismatch,
    #[error("All blocks must have the same number of frames")]
    FrameMismatch,
    #[error("Number of channels must fit into u16")]
    TooManyChannels,
    #[error(transparent)]
    Signal(#[from] SignalError),
}

pub fn signal_to_blocks(
    signal: &TimeSignal,
    frames_per_block: usize,
) -> Result<Vec<Planar<f64>>, BlockAdapterError> {
    if frames_per_block == 0 {
        return Err(BlockAdapterError::FramesPerBlockZero);
    }

    let num_channels =
        u16::try_from(signal.num_channels()).map_err(|_| BlockAdapterError::TooManyChannels)?;
    let num_samples = signal.num_time_steps();
    let num_blocks = num_samples.div_ceil(frames_per_block);
    let mut blocks = Vec::with_capacity(num_blocks);

    for block_index in 0..num_blocks {
        let start = block_index * frames_per_block;
        let end = usize::min(start + frames_per_block, num_samples);
        let mut block = Planar::<f64>::new(num_channels, frames_per_block);

        for (channel_index, dst_channel) in block.channels_mut().enumerate() {
            let src_channel = signal.channel(channel_index);
            let len = end - start;
            for (dst, src) in dst_channel[..len]
                .iter_mut()
                .zip(src_channel.iter().skip(start).take(len))
            {
                *dst = *src;
            }
        }

        blocks.push(block);
    }

    Ok(blocks)
}

pub fn signal_to_block(signal: &TimeSignal) -> Result<Planar<f64>, BlockAdapterError> {
    let num_channels =
        u16::try_from(signal.num_channels()).map_err(|_| BlockAdapterError::TooManyChannels)?;
    let mut block = Planar::<f64>::new(num_channels, signal.num_time_steps());

    for (channel_index, dst_channel) in block.channels_mut().enumerate() {
        for (dst, src) in dst_channel
            .iter_mut()
            .zip(signal.channel(channel_index).iter())
        {
            *dst = *src;
        }
    }

    Ok(block)
}

pub fn signal_from_blocks<B>(
    blocks: impl IntoIterator<Item = B>,
    sample_rate: f64,
) -> Result<TimeSignal, BlockAdapterError>
where
    B: AudioBlock<f64>,
{
    if sample_rate <= 0.0 {
        return Err(BlockAdapterError::SampleRateZeroOrNeg);
    }

    let mut iter = blocks.into_iter();
    let Some(first_block) = iter.next() else {
        return Err(BlockAdapterError::EmptyBlockList);
    };

    let num_channels = first_block.num_channels() as usize;
    let frames_per_block = first_block.num_frames();
    let mut owned_blocks = vec![first_block];

    for block in iter {
        if block.num_channels() as usize != num_channels {
            return Err(BlockAdapterError::ChannelMismatch);
        }
        if block.num_frames() != frames_per_block {
            return Err(BlockAdapterError::FrameMismatch);
        }
        owned_blocks.push(block);
    }

    let total_frames = frames_per_block * owned_blocks.len();
    let mut data = Array2::zeros((num_channels, total_frames));

    for (block_index, block) in owned_blocks.iter().enumerate() {
        let start = block_index * frames_per_block;
        let end = start + frames_per_block;

        for (channel_index, src_channel) in block.channels_iter().enumerate() {
            for (frame_index, sample) in src_channel.enumerate() {
                data[(channel_index, start + frame_index)] = *sample;
            }
        }

        debug_assert_eq!(end - start, frames_per_block);
    }

    TimeSignal::new(data, sample_rate).map_err(BlockAdapterError::from)
}

pub fn signal_from_block<B>(block: B, sample_rate: f64) -> Result<TimeSignal, BlockAdapterError>
where
    B: AudioBlock<f64>,
{
    signal_from_blocks([block], sample_rate)
}

#[cfg(test)]
mod tests {
    use audio_blocks::Planar;
    use ndarray::arr2;

    use super::*;

    #[test]
    fn signal_to_blocks_pads_last_block_with_zeros() -> Result<(), BlockAdapterError> {
        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 48_000.0)?;

        let blocks = signal_to_blocks(&signal, 2)?;

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].channel(0), &[1.0, 2.0]);
        assert_eq!(blocks[0].channel(1), &[4.0, 5.0]);
        assert_eq!(blocks[1].channel(0), &[3.0, 0.0]);
        assert_eq!(blocks[1].channel(1), &[6.0, 0.0]);
        Ok(())
    }

    #[test]
    fn signal_from_blocks_concatenates_blocks() -> Result<(), BlockAdapterError> {
        let first = Planar::from_slice(&[&[1.0, 2.0][..], &[10.0, 20.0][..]]);
        let second = Planar::from_slice(&[&[3.0, 0.0][..], &[30.0, 0.0][..]]);

        let signal = signal_from_blocks([first, second], 48_000.0)?;

        assert_eq!(
            signal.time_data(),
            arr2(&[[1.0, 2.0, 3.0, 0.0], [10.0, 20.0, 30.0, 0.0]])
        );
        Ok(())
    }

    #[test]
    fn signal_to_block_converts_whole_signal_to_single_block() -> Result<(), BlockAdapterError> {
        let signal = TimeSignal::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 48_000.0)?;

        let block = signal_to_block(&signal)?;

        assert_eq!(block.channel(0), &[1.0, 2.0, 3.0]);
        assert_eq!(block.channel(1), &[4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn signal_from_block_supports_single_block() -> Result<(), BlockAdapterError> {
        let block = Planar::from_slice(&[&[1.0, 2.0, 3.0][..]]);

        let signal = signal_from_block(block, 48_000.0)?;

        assert_eq!(signal.time_data(), arr2(&[[1.0, 2.0, 3.0]]));
        Ok(())
    }

    #[test]
    fn signal_to_blocks_rejects_zero_block_size() {
        let signal = TimeSignal::zeros(1, 4, 48_000.0).unwrap();

        let result = signal_to_blocks(&signal, 0);

        assert!(matches!(result, Err(BlockAdapterError::FramesPerBlockZero)));
    }

    #[test]
    fn signal_from_blocks_rejects_empty_input() {
        let result = signal_from_blocks::<Planar<f64>>(Vec::new(), 48_000.0);

        assert!(matches!(result, Err(BlockAdapterError::EmptyBlockList)));
    }

    #[test]
    fn signal_from_blocks_rejects_channel_mismatch() {
        let mono = Planar::from_slice(&[&[1.0, 2.0][..]]);
        let stereo = Planar::from_slice(&[&[3.0, 4.0][..], &[5.0, 6.0][..]]);

        let result = signal_from_blocks([mono, stereo], 48_000.0);

        assert!(matches!(result, Err(BlockAdapterError::ChannelMismatch)));
    }

    #[test]
    fn signal_from_blocks_rejects_frame_mismatch() {
        let first = Planar::from_slice(&[&[1.0, 2.0][..]]);
        let second = Planar::from_slice(&[&[3.0, 4.0, 5.0][..]]);

        let result = signal_from_blocks([first, second], 48_000.0);

        assert!(matches!(result, Err(BlockAdapterError::FrameMismatch)));
    }

    #[test]
    fn signal_from_blocks_rejects_invalid_sample_rate() {
        let block = Planar::from_slice(&[&[1.0, 2.0][..]]);

        let result = signal_from_block(block, 0.0);

        assert!(matches!(
            result,
            Err(BlockAdapterError::SampleRateZeroOrNeg)
        ));
    }
}
