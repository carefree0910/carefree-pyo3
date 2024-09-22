//! # io/temporal/mem/fetchers
//!
//! a `Fetcher` is defined as a one-time data fetcher.
//!
//! the general workflow of a complete mem-data fetching process is as follows:
//! 1. determine a data fetching schema based on the format of the underlying mem-data
//! (e.g., row-contiguous or column-contiguous)
//! 2. separate the final task into tiny tasks, each of which will fetch a small slice of data
//! > the 'slice' here indicates a continuous piece of data laid out in memory
//! 3. create a `Fetcher` for each tiny task
//!
//! this module focuses on the final step, and the first two steps are implemented in `../mem`.

use crate::toolkit::array::AFloat;
use numpy::{ndarray::CowArray, Ix1};
use std::future::Future;

pub mod shm;

/// arguments for the `Fetcher::fetch` method
///
/// # concepts
///
/// - `compact data`: we assume that the data of each date is flattened
/// and then concatenated into a single array, and this array is called `compact data`
/// - `date data`: the flattened data of the specific date
///
/// # fields
///
/// - `c`: the index of the current fetcher among all fetchers in a batch get mode
/// > if not in batch get mode, this field should be `None`
/// - `start_idx`: the start index of the `compact data` to fetch
/// - `end_idx`: the end index of the `compact data` to fetch
/// - `date_idx`: the index of the date among all dates
/// - `date_col_idx`: the index of the column in the `date data` to fetch
/// - `date_start_idx`: the index where the `date data` starts in the `compact data`
/// - `time_start_idx`: the start index of the `date data` to fetch
/// - `time_end_idx`: the end index of the `date data` to fetch
/// - `num_ticks_per_day`: the number of ticks per day
/// - `data_len`: the length of the data to fetch, i.e. `end_idx - start_idx` or `time_end_idx - time_start_idx`
#[derive(Debug)]
pub struct FetcherArgs {
    pub c: Option<usize>,
    pub start_idx: i64,
    pub end_idx: i64,
    pub date_idx: i64,
    pub date_col_idx: i64,
    pub date_start_idx: i64,
    pub time_start_idx: i64,
    pub time_end_idx: i64,
    pub num_ticks_per_day: i64,
    pub data_len: i64,
}

pub trait Fetcher<T: AFloat> {
    fn can_batch_fetch(&self) -> bool {
        false
    }
    fn fetch(&self, args: FetcherArgs) -> CowArray<T, Ix1> {
        if self.can_batch_fetch() {
            self.batch_fetch(vec![args]).pop().unwrap()
        } else {
            unreachable!("should implement `fetch` when `can_batch_fetch` returns false");
        }
    }
    fn batch_fetch(&self, _args: Vec<FetcherArgs>) -> Vec<CowArray<T, Ix1>> {
        unreachable!("`batch_fetch` should be implemented when `can_batch_fetch` returns true");
    }
}

pub trait AsyncFetcher<T: AFloat> {
    fn fetch(&self, args: FetcherArgs) -> impl Future<Output = CowArray<T, Ix1>>;
}
