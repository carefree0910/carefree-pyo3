use crate::{df::DataFrame, toolkit::array::AFloat};
use anyhow::Result;
use std::future::Future;

#[cfg(feature = "io-source-opendal")]
pub mod s3;

pub trait Source<T: AFloat> {
    /// read data from source, based on `date` and `key`
    fn read(&self, date: &str, key: &str) -> impl Future<Output = Result<DataFrame<T>>>;
    /// write data of specific `date` and `key` to source
    fn write(&self, date: &str, key: &str, df: &DataFrame<T>) -> impl Future<Output = Result<()>>;
}
