pub use complex_data::ComplexData;
pub use real_data::RealData;

pub mod complex_data;
pub mod real_data;

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("x data has not the same length as y data")]
    NotMatching,
    #[error("x data is not increasing")]
    NotIncreasing,
}
