use crate::{df::DataFrame, toolkit::array::AFloat};
use anyhow::Result;
use itertools::izip;
use std::{future::Future, iter::zip};

#[cfg(feature = "io-source-opendal")]
pub mod s3;

pub trait Source<T: AFloat> {
    /// read data from source, based on `date` and `key`
    fn read(&self, date: &str, key: &str) -> impl Future<Output = Result<DataFrame<T>>>;
    /// write data of specific `date` and `key` to source
    fn write(&self, date: &str, key: &str, df: &DataFrame<T>) -> impl Future<Output = Result<()>>;
    /// batch version of `read`
    fn batch_read(
        &self,
        dates: &[&str],
        keys: &[&str],
    ) -> impl Future<Output = Result<Vec<DataFrame<T>>>> {
        let futures = zip(dates, keys)
            .map(|(date, key)| self.read(date, key))
            .collect::<Vec<_>>();
        futures::future::try_join_all(futures)
    }
    /// batch version of `write`
    fn batch_write(
        &self,
        dates: &[&str],
        keys: &[&str],
        dfs: &[&DataFrame<T>],
    ) -> impl Future<Output = Result<Vec<()>>> {
        let futures = izip!(dates, keys, dfs)
            .map(|(date, key, df)| self.write(date, key, df))
            .collect::<Vec<_>>();
        futures::future::try_join_all(futures)
    }
}
