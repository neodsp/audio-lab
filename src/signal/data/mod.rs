pub use axis::XAxis;
pub use complex_data::ComplexData;
pub use real_data::RealData;

mod axis;
mod complex_data;
mod real_data;

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("x data has not the same length as y data")]
    NotMatching,
    #[error("x data is not increasing")]
    NotIncreasing,
    #[error(
        "uniform axis needs a finite start and a finite, positive step (start: {start}, step: {step})"
    )]
    InvalidUniformAxis { start: f64, step: f64 },
    #[error("there are {labels} channel labels for {channels} channels")]
    ChannelLabelMismatch { labels: usize, channels: usize },
}
