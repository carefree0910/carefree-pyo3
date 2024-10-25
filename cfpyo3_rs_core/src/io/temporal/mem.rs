//! # io/temporal/mem
//!
//! this module aims to accelerate 'random-fetching' of temporal data stored in 'memory'.
//!
//! 'memory' here refers to an implementation which can access the data 'quickly'. it
//! does not necessarily mean that the data is stored in the RAM.
//!
//! but of course, we support fetching data from RAM. in this case, we suggest storing the
//! data in shared memory (SHM) to save memory copy time on the user side.
//!
//! > that's also why some public interfaces start with `shm`.
//!
//! the general workflow of a complete mem-data fetching process is as follows:
//! 1. determine a data fetching schema based on the format of the underlying mem-data
//!    (e.g., row-contiguous or column-contiguous)
//! 2. separate the final task into tiny tasks, each of which will fetch a small slice of data
//! > the 'slice' here indicates a continuous piece of data laid out in memory
//! 3. create a `Fetcher` for each tiny task
//!
//! this module focuses on the first two steps, and the final step is implemented in sub-modules.

#![allow(clippy::too_many_arguments)]

use crate::{
    df::ColumnsDtype,
    toolkit::array::{batch_searchsorted, searchsorted, unique, AFloat, UnsafeSlice},
};
use anyhow::Result;
use itertools::Itertools;
use numpy::{
    ndarray::{s, Array1, ArrayView1, CowArray},
    Ix1,
};
#[cfg(feature = "io-mem-redis")]
use redis::{RedisClient, RedisFetcher, RedisKey, RedisKeyRepr};
use shm::SHMFetcher;
use std::{
    collections::HashMap,
    future::Future,
    iter::zip,
    sync::{mpsc::channel, Arc, Mutex},
};

#[cfg(feature = "io-mem-redis")]
pub mod redis;
pub mod shm;

// core implementations

#[derive(Debug)]
struct MemError(String);
impl MemError {
    fn new(msg: &str) -> Self {
        Self(msg.to_string())
    }
    fn datetime_index_not_continuous() -> Result<()> {
        Err(MemError::new("`datetime_index` is not continuous").into())
    }
    fn data_getter_not_contiguous() -> Result<()> {
        Err(MemError::new("`data_getter` is not returning contiguous data").into())
    }
}
impl std::error::Error for MemError {}
impl std::fmt::Display for MemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error occurred in `mem` module: {}", self.0)
    }
}

macro_rules! getter_as_slice_or_err {
    ($data:expr) => {
        match $data.as_slice() {
            Some(data) => data,
            None => return MemError::data_getter_not_contiguous(),
        }
    };
}

/// arguments for the `Fetcher::fetch` method
///
/// # concepts
///
/// - `compact data`: we assume that the data of each date is flattened
///   and then concatenated into a single array, and this array is called `compact data`
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
    fn fetch(&self, args: FetcherArgs) -> Result<CowArray<T, Ix1>> {
        if self.can_batch_fetch() {
            let mut rv = self.batch_fetch(vec![args])?;
            rv.pop().ok_or(Err(MemError::new("empty result"))?)
        } else {
            unreachable!("should implement `fetch` when `can_batch_fetch` returns false");
        }
    }
    fn batch_fetch(&self, _args: Vec<FetcherArgs>) -> Result<Vec<CowArray<T, Ix1>>> {
        unreachable!("`batch_fetch` should be implemented when `can_batch_fetch` returns true");
    }
}

pub trait AsyncFetcher<T: AFloat> {
    fn fetch(&self, args: FetcherArgs) -> impl Future<Output = Result<CowArray<T, Ix1>>>;
}

/// a typically useful function for random-fetching temporal row-contiguous data.
///
/// # constants
///
/// - `Nd`: total number of the dates.
/// - `Sn`: number of columns of day 'n'. Notice that `len(Sn) = Nd`.
/// - `S`: total number of columns, equals to `sum(Sn)`.
/// - `T`: number of ticks per day, we assume that the number of ticks per day remains the same.
///
/// # arguments
///
/// - `datetime_start` - the start of the datetime range to fetch.
/// - `datetime_end` - the end of the datetime range to fetch.
/// - `datetime_len` - the length of the datetime range to fetch.
/// - `columns` - the columns to fetch.
/// - `num_ticks_per_day` - equals to `T`.
/// - `full_index` - the full datetime index.
/// - `date_columns_offset` (`Nd + 1`, equals to `[0, *cumsum(Sn)]`) - the offset of each date.
/// - `columns_getter` - the getter function for columns, args: `(start_idx, end_idx)`.
/// - `data_getter` - the getter function for data, args: `(start_idx, end_idx)`.
/// - `flattened` (`datetime_len * columns.len()`) - the flattened array to fill.
///
/// # notes
///
/// this function is not very practical to use, so we are neither utilizing the `fetcher` module
/// nor the `ColumnIndicesGetter` trait for simplicity.
pub fn row_contiguous<'a, T>(
    datetime_start: i64,
    datetime_end: i64,
    datetime_len: i64,
    columns: ArrayView1<ColumnsDtype>,
    num_ticks_per_day: i64,
    full_index: ArrayView1<i64>,
    date_columns_offset: ArrayView1<i64>,
    columns_getter: &dyn Fn(i64, i64) -> ArrayView1<'a, ColumnsDtype>,
    data_getter: &dyn Fn(i64, i64) -> ArrayView1<'a, T>,
    flattened: &mut [T],
) -> Result<()>
where
    T: AFloat,
{
    use anyhow::Ok;

    let time_start_idx = searchsorted(&full_index, &datetime_start);
    let time_end_idx = time_start_idx + datetime_len as usize - 1;
    if full_index[time_end_idx] != datetime_end {
        MemError::datetime_index_not_continuous()
    } else {
        let date_idxs = (time_start_idx..=time_end_idx)
            .map(|x| x as i64 / num_ticks_per_day)
            .collect_vec();
        let (unique_date_idxs, date_counts) = unique(&date_idxs);
        let columns_per_day = Vec::from_iter(unique_date_idxs.iter().map(|&date_idx| {
            let start_idx = date_columns_offset[date_idx as usize];
            let end_idx = date_columns_offset[(date_idx + 1) as usize];
            columns_getter(start_idx, end_idx)
        }));
        let num_columns_per_day = Array1::from_iter(columns_per_day.iter().map(|x| x.len() as i64));
        let num_columns = columns.len();
        let start_idx = {
            let offset_before =
                date_columns_offset[unique_date_idxs[0] as usize] * num_ticks_per_day;
            let date_offset = (time_start_idx as i64 % num_ticks_per_day) * num_columns_per_day[0];
            offset_before + date_offset
        };

        let mut cursor: i64 = 0;
        let mut time_idx: i64 = 0;
        zip(date_counts, columns_per_day).try_for_each(|(date_count, date_columns)| {
            let columns_len = date_columns.len();
            let cursor_end = cursor + date_count * columns_len as i64;
            let time_idx_end = time_idx + date_count;
            let columns_idx = batch_searchsorted(&date_columns, &columns);
            let mut corrected_columns_idx = columns_idx.clone();
            columns_idx.iter().enumerate().for_each(|(i, &column_idx)| {
                if column_idx >= columns_len {
                    corrected_columns_idx[i] = 0;
                }
            });
            let selected_data = data_getter(start_idx + cursor, start_idx + cursor_end)
                .into_shape((date_count as usize, columns_len))?;
            let selected_data = selected_data.t();
            zip(corrected_columns_idx, columns_idx)
                .enumerate()
                .try_for_each(|(i, (corrected_idx, idx))| {
                    if date_columns[corrected_idx] == columns[i] {
                        let i_selected_data = selected_data.row(idx);
                        i_selected_data.iter().enumerate().for_each(|(j, &x)| {
                            flattened[(time_idx as usize + j) * num_columns + i] = x
                        });
                    } else {
                        (0..date_count as usize).for_each(|j| {
                            flattened[(time_idx as usize + j) * num_columns + i] = T::nan();
                        });
                    }
                    Ok(())
                })?;
            cursor = cursor_end;
            time_idx = time_idx_end;
            Ok(())
        })?;
        Ok(())
    }
}

pub trait ColumnIndicesGetter {
    fn get(&self, i: usize, date_columns: &ArrayView1<ColumnsDtype>) -> Vec<usize>;
}
struct CachedGetter<'a>(
    Arc<Mutex<HashMap<usize, Vec<usize>>>>,
    ArrayView1<'a, ColumnsDtype>,
);
impl<'a> CachedGetter<'a> {
    fn new(columns: ArrayView1<'a, ColumnsDtype>) -> Self {
        Self(Arc::new(Mutex::new(HashMap::new())), columns)
    }
    fn new_vec(columns: &'a [ArrayView1<'a, ColumnsDtype>]) -> Vec<Self> {
        columns.iter().map(|x| Self::new(x.view())).collect()
    }
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0), self.1)
    }
}
impl<'a> ColumnIndicesGetter for CachedGetter<'a> {
    fn get(&self, i: usize, date_columns: &ArrayView1<ColumnsDtype>) -> Vec<usize> {
        let mut cache = self.0.lock().unwrap();
        if let Some(indices) = cache.get(&i) {
            indices.clone()
        } else {
            let indices = batch_searchsorted(date_columns, &self.1);
            cache.insert(i, indices.clone());
            indices
        }
    }
}

/// a struct that represents the offsets for each data item at the 'channel' dimension.
///
/// # concepts
///
/// a data item is typically a single fp32, so the data we fetched from the `Fetcher` looks like:
///
/// [i0, i1, i2, i3, i4, ..., iN]
///
/// and a typical scenario is that we fetch multiple data, and stack them together to become a 2D array:
///
/// [i0_0, i1_0, i2_0, i3_0, i4_0, ..., iN_0,
///  i0_1, i1_1, i2_1, i3_1, i4_1, ..., iN_1,
///  ...,
///  i0_C, i1_C, i2_C, i3_C, i4_C, ..., iN_C]
///
/// > notice that we use 'stack' / '2D' just for illustration, the actual implementation will
/// > manipulate the data in '1D', zero copy way.
///
/// here, `C` is the number of 'channels' that we want to fetch.
///
/// however, if we want to fetch more contiguous data at once, we can aggregate multiple data items
/// among the channel dimension, and treat them as one. in this case, the data we fetched from the
/// `Fetcher` will look like:
///
/// [(i00, i01, ..., i0k), (i10, i11, ..., i1k), ..., (iN0, iN1, ..., iNk)]
///
/// where `k` is the `multiplier` for each data item.
///
/// > notice that the parentheses do not actually exist in the data, they are just showing that we
/// > treat `k` data items as one.
///
/// plus, if we have multiple aggregates, we need to stack them together along the channel dimension.
/// consider we have 3 aggregates, each aggregates p, q, r data items, then the data we fetched will look like:
///
/// [(i00, i01, ..., i0p), (i10, i11, ..., i1p), ..., (iN0, iN1, ..., iNp)]
/// [(j00, j01, ..., j0q), (j10, j11, ..., j1q), ..., (jN0, jN1, ..., jNq)]
/// [(k00, k01, ..., k0r), (k10, k11, ..., k1r), ..., (kN0, kN1, ..., kNr)]
///
/// > notice that p+q+r=C.
///
/// now, we want to stack them into the final 2D array:
///
/// [i00, i10, i20, ..., iN0
///  i01, i11, i21, ..., iN1
///  ...,
///  i0p, i1p, i2p, ..., iNp,
///  j00, j10, j20, ..., jN0,
///  j01, j11, j21, ..., jN1,
///  ...,
///  j0q, j1q, j2q, ..., jNq,
///  ...,
///  k0r, k1r, k2r, ..., kNr]
///
/// so you will notice that in order to fill those data into the 2D array, we need to know
/// the 'position' of the current data in each 'column' - or precisely, the 'channel'. that's
/// why we need the `pad_start` and `pad_end`.
///
/// in the above example, the `multiplier`, `pad_start` and `pad_end` of each data will be:
///
/// - (p, 0, q+r) for `i`
/// - (q, p, r) for `j`
/// - (r, p+q, 0) for `k`
///
pub struct ChannelOffsets {
    multiplier: i64,
    pad_start: i64,
    pad_end: i64,
}
impl ChannelOffsets {
    fn total_multiplier(&self) -> i64 {
        self.multiplier + self.pad_start + self.pad_end
    }
}

/// a typically useful function for random-fetching temporal column-contiguous data.
///
/// # constants
///
/// - `Nd`: total number of the dates.
/// - `Sn`: number of columns of day 'n'. Notice that `len(Sn) = Nd`.
/// - `S`: total number of columns, equals to `sum(Sn)`.
/// - `T`: number of ticks per day, we assume that the number of ticks per day remains the same.
///
/// # arguments
///
/// - `c` - the index of the current fetcher among all fetchers in a batch get mode.
/// > if not in batch get mode, this field should be `None`.
/// - `datetime_start` - the start of the datetime range to fetch.
/// - `datetime_end` - the end of the datetime range to fetch.
/// - `datetime_len` - the length of the datetime range to fetch.
/// - `columns` - the columns to fetch.
/// - `num_ticks_per_day` - equals to `T`.
/// - `full_index` - the full datetime index.
/// - `date_columns_offset` (`Nd + 1`, equals to `[0, *cumsum(Sn)]`) - the offset of each date.
/// - `columns_getter` - the getter function for columns, args: `(start_idx, end_idx)`.
/// - `columns_indices_getter` - the getter function for columns indices, args: `(i, date_columns)`.
/// > it should in fact return `searchsorted_array(&date_columns.view(), columns)`, but we provide
/// > this interface so you can optimize the performance by caching the result (depends on `i`).
/// - `data_getter` - the getter function for data, args: `(start_idx, end_idx)`.
/// - `flattened` (`datetime_len * columns.len()`) - the flattened array to fill.
/// - `column_offset` - the column offset, see the `column_offset` concept below.
/// - `channel_offsets` - the channel offsets for each data item, see documentation for [`ChannelOffsets`].
///
/// # `column_offset` concept
///
/// typically, we will spawn tasks among two dimensions: batch size (`B`) and number of features (`C`).
/// but when the features are aggregated, the feature dimension (or, the 'channel' dimension) will be
/// reduced by a factor of `Ck`, where `Ck` is the number of features in each aggregate, and will become
/// the `multiplier` argument in [`ChannelOffsets`].
///
/// since `Ck` is often ≥ 50, we will need to seek new ways to increase the number of tasks to be spawned
/// in order to fully utilize the CPU. hence, we breaks the columns (`N`) into `Nk` groups to balance
/// the situation.
///
/// in this case, `column_offset` will be important, as it tells us in which column group the current
/// fetcher is in.
///
/// for example, with the above annotations, if the current fetcher is in the `n`th group, then the
/// `column_offset` should be `n * N / Nk`
///
/// # dim order
///
/// it is worth mentioning that, depends on whether `channel_offsets` is given, the target array will
/// have different dimensions order:
///
/// - if `channel_offsets` is `None`, the target dim order will `(C, N, datetime_len)`.
/// - if `channel_offsets` is `Some`, the target dim order will `(N, datetime_len, C)`, because
///   in this case, we will aggregate multiple data items and treat them as one, as mentioned
///   in the [`ChannelOffsets`] documentation.
pub fn column_contiguous<'a, T>(
    c: Option<usize>,
    datetime_start: i64,
    datetime_end: i64,
    datetime_len: i64,
    columns: ArrayView1<ColumnsDtype>,
    num_ticks_per_day: i64,
    full_index: ArrayView1<i64>,
    date_columns_offset: ArrayView1<i64>,
    columns_getter: &dyn Fn(i64, i64) -> ArrayView1<'a, ColumnsDtype>,
    columns_indices_getter: &dyn ColumnIndicesGetter,
    data_getter: &dyn Fetcher<T>,
    flattened: &mut UnsafeSlice<T>,
    column_offset: Option<i64>,
    channel_offsets: Option<ChannelOffsets>,
) -> Result<()>
where
    T: AFloat,
{
    let time_start_idx = searchsorted(&full_index, &datetime_start);
    let time_end_idx = time_start_idx + datetime_len as usize - 1;
    if full_index[time_end_idx] != datetime_end {
        MemError::datetime_index_not_continuous()
    } else {
        let start_date_idx = time_start_idx as i64 / num_ticks_per_day;
        let end_date_idx = time_end_idx as i64 / num_ticks_per_day;
        let start_idx = date_columns_offset[start_date_idx as usize] * num_ticks_per_day;
        let unique_date_idxs = Vec::from_iter(start_date_idx..=end_date_idx);
        let columns_per_day = Vec::from_iter(unique_date_idxs.iter().map(|&date_idx| {
            let start_idx = date_columns_offset[date_idx as usize];
            let end_idx = date_columns_offset[(date_idx + 1) as usize];
            columns_getter(start_idx, end_idx)
        }));
        let num_columns_per_day: Vec<usize> = columns_per_day.iter().map(|x| x.len()).collect();
        let mut fill_data = |f_start_idx: usize, data_slice: &[T]| match channel_offsets {
            None => flattened.copy_from_slice(f_start_idx, data_slice),
            Some(ChannelOffsets {
                multiplier,
                pad_start,
                pad_end,
            }) => {
                let time_len = data_slice.len() / multiplier as usize;
                let total_multiplier = multiplier + pad_start + pad_end;
                (0..time_len).for_each(|t| {
                    let t_start = t as i64 * total_multiplier + pad_start;
                    let ft_start = f_start_idx + t_start as usize;
                    let dt_start = t * multiplier as usize;
                    let dt_end = (t + 1) * multiplier as usize;
                    flattened.copy_from_slice(ft_start, &data_slice[dt_start..dt_end]);
                });
            }
        };

        let mut i_time_offset = 0;
        let mut total_columns_offset = 0;
        let column_offset = column_offset.unwrap_or(0);
        let total_multiplier = channel_offsets
            .as_ref()
            .map(|offsets| offsets.total_multiplier());
        columns_per_day
            .iter()
            .enumerate()
            .try_for_each(|(i, date_columns)| -> Result<()> {
                let i_offset = num_ticks_per_day * total_columns_offset;
                let i_start_idx = start_idx + i_offset;
                let i_time_start_idx = if i == 0 {
                    time_start_idx as i64 % num_ticks_per_day
                } else {
                    0
                };
                let mut i_time_end_idx = if i == columns_per_day.len() - 1 {
                    (time_end_idx as i64 + 1) % num_ticks_per_day
                } else {
                    num_ticks_per_day
                };
                if i_time_end_idx == 0 {
                    i_time_end_idx = num_ticks_per_day;
                }
                let i_rows_idx = columns_indices_getter.get(i, &date_columns.view());
                let mut i_corrected_rows_idx = i_rows_idx.clone();
                i_rows_idx.iter().enumerate().for_each(|(j, &row_idx)| {
                    if row_idx >= num_columns_per_day[i] {
                        i_corrected_rows_idx[j] = 0;
                    }
                });
                let mut tasks = Vec::new();
                let mut f_start_indices = Vec::new();
                zip(i_corrected_rows_idx, i_rows_idx)
                    .enumerate()
                    .try_for_each(|(j, (corrected_idx, idx))| -> Result<()> {
                        let date_start_idx = i_start_idx + idx as i64 * num_ticks_per_day;
                        let mut f_start_idx =
                            (j as i64 + column_offset) * datetime_len + i_time_offset;
                        if let Some(total_multiplier) = total_multiplier {
                            f_start_idx *= total_multiplier;
                        }
                        let f_start_idx = f_start_idx as usize;
                        if date_columns[corrected_idx] == columns[j] {
                            let corrected_idx = corrected_idx as i64;
                            let col_offset = corrected_idx * num_ticks_per_day;
                            let args = FetcherArgs {
                                c,
                                start_idx: date_start_idx + i_time_start_idx,
                                end_idx: date_start_idx + i_time_end_idx,
                                date_idx: unique_date_idxs[i],
                                date_col_idx: corrected_idx,
                                date_start_idx,
                                time_start_idx: col_offset + i_time_start_idx,
                                time_end_idx: col_offset + i_time_end_idx,
                                num_ticks_per_day,
                                data_len: i_time_end_idx - i_time_start_idx,
                            };
                            if data_getter.can_batch_fetch() {
                                tasks.push(args);
                                f_start_indices.push(f_start_idx);
                            } else {
                                let data = data_getter.fetch(args)?;
                                let data = getter_as_slice_or_err!(data);
                                fill_data(f_start_idx, data);
                            }
                        } else {
                            let mut nan_len = i_time_end_idx - i_time_start_idx;
                            if let Some(offsets) = &channel_offsets {
                                nan_len *= offsets.multiplier;
                            }
                            fill_data(f_start_idx, vec![T::nan(); nan_len as usize].as_slice());
                        }
                        Ok(())
                    })?;
                if !tasks.is_empty() {
                    let batch_data = data_getter.batch_fetch(tasks)?;
                    zip(batch_data, f_start_indices).try_for_each(|(data, f_start_idx)| {
                        let data = getter_as_slice_or_err!(data);
                        fill_data(f_start_idx, data);
                        Ok(())
                    })?;
                }
                i_time_offset = i_time_offset + i_time_end_idx - i_time_start_idx;
                total_columns_offset += num_columns_per_day[i] as i64;
                Ok(())
            })?;
        Ok(())
    }
}

/// an async version of `column_contiguous`.
pub async fn async_column_contiguous<'a, T, F>(
    c: Option<usize>,
    datetime_start: i64,
    datetime_end: i64,
    datetime_len: i64,
    columns: ArrayView1<'_, ColumnsDtype>,
    num_ticks_per_day: i64,
    full_index: ArrayView1<'_, i64>,
    date_columns_offset: ArrayView1<'_, i64>,
    columns_getter: &dyn Fn(i64, i64) -> ArrayView1<'a, ColumnsDtype>,
    columns_indices_getter: &dyn ColumnIndicesGetter,
    data_getter: F,
    mut flattened: UnsafeSlice<'_, T>,
    column_offset: Option<i64>,
    channel_offsets: Option<ChannelOffsets>,
) -> Result<()>
where
    T: AFloat,
    F: AsyncFetcher<T>,
{
    let time_start_idx = searchsorted(&full_index, &datetime_start);
    let time_end_idx = time_start_idx + datetime_len as usize - 1;
    if full_index[time_end_idx] != datetime_end {
        MemError::datetime_index_not_continuous()
    } else {
        let start_date_idx = time_start_idx as i64 / num_ticks_per_day;
        let end_date_idx = time_end_idx as i64 / num_ticks_per_day;
        let start_idx = date_columns_offset[start_date_idx as usize] * num_ticks_per_day;
        let unique_date_idxs = Vec::from_iter(start_date_idx..=end_date_idx);
        let columns_per_day = Vec::from_iter(unique_date_idxs.iter().map(|&date_idx| {
            let start_idx = date_columns_offset[date_idx as usize];
            let end_idx = date_columns_offset[(date_idx + 1) as usize];
            columns_getter(start_idx, end_idx)
        }));
        let num_columns_per_day: Vec<usize> = columns_per_day.iter().map(|x| x.len()).collect();
        let mut fill_data = |f_start_idx: usize, data_slice: &[T]| match channel_offsets {
            None => flattened.copy_from_slice(f_start_idx, data_slice),
            Some(ChannelOffsets {
                multiplier,
                pad_start,
                pad_end,
            }) => {
                let time_len = data_slice.len() / multiplier as usize;
                let total_multiplier = multiplier + pad_start + pad_end;
                (0..time_len).for_each(|t| {
                    let t_start = t as i64 * total_multiplier + pad_start;
                    let ft_start = f_start_idx + t_start as usize;
                    let dt_start = t * multiplier as usize;
                    let dt_end = (t + 1) * multiplier as usize;
                    flattened.copy_from_slice(ft_start, &data_slice[dt_start..dt_end]);
                });
            }
        };

        let mut columns_offset = 0;
        let mut i_time_offset = 0;
        let column_offset = column_offset.unwrap_or(0);
        let total_multiplier = channel_offsets
            .as_ref()
            .map(|offsets| offsets.total_multiplier());
        let mut tasks = Vec::new();
        let mut f_start_indices = Vec::new();
        columns_per_day
            .iter()
            .enumerate()
            .for_each(|(i, date_columns)| {
                let i_offset = num_ticks_per_day * columns_offset;
                let i_start_idx = start_idx + i_offset;
                let i_time_start_idx = if i == 0 {
                    time_start_idx as i64 % num_ticks_per_day
                } else {
                    0
                };
                let mut i_time_end_idx = if i == columns_per_day.len() - 1 {
                    (time_end_idx as i64 + 1) % num_ticks_per_day
                } else {
                    num_ticks_per_day
                };
                if i_time_end_idx == 0 {
                    i_time_end_idx = num_ticks_per_day;
                }
                let i_rows_idx = columns_indices_getter.get(i, &date_columns.view());
                let mut i_corrected_rows_idx = i_rows_idx.clone();
                i_rows_idx.iter().enumerate().for_each(|(j, &row_idx)| {
                    if row_idx >= num_columns_per_day[i] {
                        i_corrected_rows_idx[j] = 0;
                    }
                });
                zip(i_corrected_rows_idx, i_rows_idx).enumerate().for_each(
                    |(j, (corrected_idx, idx))| {
                        let date_start_idx = i_start_idx + idx as i64 * num_ticks_per_day;
                        let mut f_start_idx =
                            (j as i64 + column_offset) * datetime_len + i_time_offset;
                        if let Some(total_multiplier) = total_multiplier {
                            f_start_idx *= total_multiplier;
                        }
                        let f_start_idx = f_start_idx as usize;
                        if date_columns[corrected_idx] == columns[j] {
                            let col_offset = corrected_idx as i64 * num_ticks_per_day;
                            let args = FetcherArgs {
                                c,
                                start_idx: date_start_idx + i_time_start_idx,
                                end_idx: date_start_idx + i_time_end_idx,
                                date_idx: unique_date_idxs[i],
                                date_col_idx: corrected_idx as i64,
                                date_start_idx,
                                time_start_idx: col_offset + i_time_start_idx,
                                time_end_idx: col_offset + i_time_end_idx,
                                num_ticks_per_day,
                                data_len: i_time_end_idx - i_time_start_idx,
                            };
                            tasks.push(args);
                            f_start_indices.push(f_start_idx);
                        } else {
                            let mut nan_len = i_time_end_idx - i_time_start_idx;
                            if let Some(offsets) = &channel_offsets {
                                nan_len *= offsets.multiplier;
                            }
                            fill_data(f_start_idx, vec![T::nan(); nan_len as usize].as_slice());
                        }
                    },
                );
                i_time_offset = i_time_offset + i_time_end_idx - i_time_start_idx;
                columns_offset += num_columns_per_day[i] as i64;
            });
        let futures = tasks.into_iter().map(|args| data_getter.fetch(args));
        let batch_data = futures::future::try_join_all(futures).await?;
        zip(batch_data, f_start_indices).try_for_each(|(data, f_start_idx)| {
            let data = getter_as_slice_or_err!(data);
            fill_data(f_start_idx, data);
            Ok(())
        })?;
        Ok(())
    }
}

pub trait FetcherGetter<'a, T: AFloat>: Sync {
    fn get<'b>(&self, c: usize) -> Box<dyn Fetcher<T> + 'b>
    where
        'a: 'b;
}

/// a typically useful function for random-fetching temporal column-contiguous data in batch.
///
/// # constants
///
/// - `B`: batch size.
/// - `C`: number of features.
/// - `Nd`: total number of the dates.
/// - `Sn`: number of columns of day 'n'. Notice that `len(Sn) = Nd`.
/// - `S`: total number of columns, equals to `sum(Sn)`.
/// - `T`: number of ticks per day, we assume that the number of ticks per day remains the same.
///
/// # arguments
///
/// - [len = `B`] `datetime_start` - the start of the datetime range to fetch.
/// - [len = `B`] `datetime_end` - the end of the datetime range to fetch.
/// - [ scalar ]  `datetime_len` - the length of the datetime range to fetch.
/// - [len = `B`] `columns` - the columns to fetch.
/// - [ scalar ]  `num_ticks_per_day` - equals to `T`.
/// - [len = `C`] `full_index` - the full datetime index.
/// - [len = `C`] `date_columns_offset` (`Nd + 1`, equals to `[0, *cumsum(Sn)]`) - the offset of each date.
/// - [len = `C`] `compact_columns` (`S`) - the full, compact columns.
/// - [ object ]  `fetcher_getter` - the getter of the [`Fetcher`] object.
/// - [ scalar ]  `num_threads` - the number of threads to use.
///
/// # dim order
///
/// the target dim order (shape) will be `(B, C, N, datetime_len)`.
pub fn batch_column_contiguous<'a, T: AFloat>(
    datetime_start: &[i64],
    datetime_end: &[i64],
    datetime_len: i64,
    columns: &'a [ArrayView1<'a, ColumnsDtype>],
    num_ticks_per_day: i64,
    full_index: &[ArrayView1<i64>],
    date_columns_offset: &[ArrayView1<i64>],
    compact_columns: &[ArrayView1<ColumnsDtype>],
    fetcher_getter: &dyn FetcherGetter<T>,
    num_threads: usize,
) -> Result<Vec<T>> {
    let nc = full_index.len();
    let num_data_per_task = datetime_len as usize * columns[0].len();
    let num_tasks = datetime_start.len() * nc;
    let mut flattened = vec![T::zero(); num_tasks * num_data_per_task];
    let flattened_slice = flattened.as_mut_slice();
    let columns_indices_getters = CachedGetter::new_vec(columns);
    let num_threads = num_threads.min(num_tasks);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()?;
    let (sender, receiver) = channel();
    pool.scope(move |s| {
        let flattened_slice = UnsafeSlice::new(flattened_slice);
        full_index.iter().enumerate().for_each(|(c, &full_index)| {
            let date_columns_offset = date_columns_offset[c];
            let compact_columns = &compact_columns[c];
            let columns_getter = |start_idx: i64, end_idx: i64| {
                compact_columns.slice(s![start_idx as isize..end_idx as isize])
            };
            datetime_start
                .iter()
                .enumerate()
                .for_each(|(i, &datetime_start)| {
                    let datetime_end = datetime_end[i];
                    let columns = columns[i];
                    let offset = (i * nc + c) * num_data_per_task;
                    let columns_indices_getter = columns_indices_getters[i].clone();
                    let i_sender = sender.clone();
                    s.spawn(move |_| {
                        let rv = column_contiguous(
                            Some(c),
                            datetime_start,
                            datetime_end,
                            datetime_len,
                            columns,
                            num_ticks_per_day,
                            full_index,
                            date_columns_offset,
                            &columns_getter,
                            &columns_indices_getter,
                            fetcher_getter.get(c).as_ref(),
                            &mut flattened_slice.slice(offset, offset + num_data_per_task),
                            None,
                            None,
                        );
                        i_sender.send(rv).unwrap();
                    });
                })
        });
    });
    receiver.iter().try_for_each(|x| x)?;
    Ok(flattened)
}

/// a typically useful function for random-fetching temporal grouped-column-contiguous data in batch.
///
/// # constants
///
/// - `B`: batch size.
/// - `G`: number of groups, usually there are multiple features in each group.
/// - `Nd`: total number of the dates.
/// - `Sn`: number of columns of day 'n'. Notice that `len(Sn) = Nd`.
/// - `S`: total number of columns, equals to `sum(Sn)`.
/// - `T`: number of ticks per day, we assume that the number of ticks per day remains the same.
///
/// # arguments
///
/// - [len = `B`] `datetime_start` - the start of the datetime range to fetch.
/// - [len = `B`] `datetime_end` - the end of the datetime range to fetch.
/// - [ scalar ]  `datetime_len` - the length of the datetime range to fetch.
/// - [len = `B`] `columns` - the columns to fetch.
/// - [ scalar ]  `num_ticks_per_day` - equals to `T`.
/// - [len = `G`] `full_index` - the full datetime index.
/// - [len = `G`] `date_columns_offset` (`Nd + 1`, equals to `[0, *cumsum(Sn)]`) - the offset of each date.
/// - [len = `G`] `compact_columns` (`S`) - the full, compact columns.
/// - [ object ]  `fetcher_getter` - the getter of the [`Fetcher`] object.
/// - [len = `G`] `multipliers` - the `multiplier` for each group, see [`ChannelOffsets`] for more information.
/// - [ bool ]    `use_batch` - whether to use batch fetching.
/// - [ scalar ]  `num_columns_chunks` - the number of chunks to split the columns to each task.
/// > for example, `num_columns_chunks = 4` means we will split the columns to 4 parts,
/// > and each part will be processed in a separate task.
/// - [ scalar ]  `num_threads` - the number of threads to use.
///
/// # dim order
///
/// the target dim order (shape) will be `(B, N, datetime_len, C)`, where `C` is sum of all `multipliers`.
pub fn batch_grouped_column_contiguous<T: AFloat>(
    datetime_start: &[i64],
    datetime_end: &[i64],
    datetime_len: i64,
    columns: &[ArrayView1<ColumnsDtype>],
    num_ticks_per_day: i64,
    full_index: &[ArrayView1<i64>],
    date_columns_offset: &[ArrayView1<i64>],
    compact_columns: &[ArrayView1<ColumnsDtype>],
    fetcher_getter: &dyn FetcherGetter<T>,
    multipliers: &[i64],
    use_batch: bool,
    num_columns_chunks: usize,
    num_threads: usize,
) -> Result<Vec<T>> {
    let bz = datetime_start.len();
    let nc = multipliers.iter().sum::<i64>();
    let n_groups = full_index.len();
    let num_columns = columns[0].len();
    let num_data_per_batch = num_columns * datetime_len as usize * nc as usize;
    let mut flattened = vec![T::zero(); bz * num_data_per_batch];
    let flattened_slice = flattened.as_mut_slice();
    let num_columns_chunks = num_columns_chunks.max(1).min(num_columns);
    let num_columns_per_task = (num_columns / num_columns_chunks).max(10.min(num_columns));
    let num_columns_task = num_columns / num_columns_per_task;
    let num_tasks = bz * n_groups * num_columns_task;
    let num_threads = num_threads.min(num_tasks);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()?;
    let (sender, receiver) = channel();
    pool.scope(move |s| {
        let flattened_slice = UnsafeSlice::new(flattened_slice);
        let num_total_columns_task = bz * num_columns_task;
        let mut columns_indices_getters = Vec::with_capacity(num_total_columns_task);
        for columns in columns.iter().take(bz) {
            for n in 0..num_columns_task {
                let column_start = n * num_columns_per_task;
                let column_end = if n == num_columns_task - 1 {
                    num_columns
                } else {
                    (n + 1) * num_columns_per_task
                };
                columns_indices_getters.push(CachedGetter::new(
                    columns.slice(s![column_start..column_end]),
                ));
            }
        }

        let mut channel_pad_start = 0;
        let mut channel_pad_end = nc;
        for g in 0..n_groups {
            let full_index = full_index[g];
            let date_columns_offset = date_columns_offset[g];
            let compact_columns = &compact_columns[g];
            let columns_getter = |start_idx: i64, end_idx: i64| {
                compact_columns.slice(s![start_idx as isize..end_idx as isize])
            };
            let multiplier = multipliers[g];
            let next_pad_start = channel_pad_start + multiplier;
            channel_pad_end -= multiplier;
            for b in 0..bz {
                let datetime_start = datetime_start[b];
                let datetime_end = datetime_end[b];
                let offset = b * num_data_per_batch;
                for n in 0..num_columns_task {
                    let bn_index = b * num_columns_task + n;
                    let column_start = n * num_columns_per_task;
                    let column_end = if n == num_columns_task - 1 {
                        num_columns
                    } else {
                        (n + 1) * num_columns_per_task
                    };
                    let columns_indices_getter = columns_indices_getters[bn_index].clone();
                    let i_sender = sender.clone();
                    s.spawn(move |_| {
                        let rv = column_contiguous(
                            if use_batch { Some(g) } else { None },
                            datetime_start,
                            datetime_end,
                            datetime_len,
                            columns[b].slice(s![column_start..column_end]),
                            num_ticks_per_day,
                            full_index,
                            date_columns_offset,
                            &columns_getter,
                            &columns_indices_getter,
                            fetcher_getter.get(g).as_ref(),
                            &mut flattened_slice.slice(offset, offset + num_data_per_batch),
                            Some(column_start as i64),
                            Some(ChannelOffsets {
                                multiplier,
                                pad_start: channel_pad_start,
                                pad_end: channel_pad_end,
                            }),
                        );
                        i_sender.send(rv).unwrap();
                    });
                }
            }
            channel_pad_start = next_pad_start;
        }
    });
    receiver.iter().try_for_each(|x| x)?;
    Ok(flattened)
}

// public interfaces

/// random-fetching temporal data with row-contiguous compact data & compact columns.
///
/// # constants
///
/// - `Nd`: total number of the dates
/// - `Sn`: number of columns of day 'n'. Notice that `len(Sn) = Nd`
/// - `S`: total number of columns, equals to `sum(Sn)`
/// - `T`: number of ticks per day, we assume that the number of ticks per day remains the same.
///
/// # arguments
///
/// - `datetime_start` - the start of the datetime range to fetch.
/// - `datetime_end` - the end of the datetime range to fetch.
/// - `datetime_len` - the length of the datetime range to fetch.
/// - `columns` - the columns to fetch.
/// - `num_ticks_per_day` - equals to `T`.
/// - `full_index` - the full datetime index.
/// - `date_columns_offset` (`Nd + 1`, equals to `[0, *cumsum(Sn)]`) - the offset of each date.
/// - `compact_columns` (`S`) - the full, compact columns.
/// - `compact_data` (`T * S`) - the full, compact data.
pub fn shm_row_contiguous<T: AFloat>(
    datetime_start: i64,
    datetime_end: i64,
    datetime_len: i64,
    columns: ArrayView1<ColumnsDtype>,
    num_ticks_per_day: i64,
    full_index: ArrayView1<i64>,
    date_columns_offset: ArrayView1<i64>,
    compact_columns: ArrayView1<ColumnsDtype>,
    compact_data: ArrayView1<T>,
) -> Result<Vec<T>> {
    let mut flattened = vec![T::zero(); datetime_len as usize * columns.len()];
    let flattened_slice = flattened.as_mut_slice();
    row_contiguous(
        datetime_start,
        datetime_end,
        datetime_len,
        columns,
        num_ticks_per_day,
        full_index,
        date_columns_offset,
        &|start_idx, end_idx| compact_columns.slice(s![start_idx as isize..end_idx as isize]),
        &|start_idx, end_idx| compact_data.slice(s![start_idx as isize..end_idx as isize]),
        flattened_slice,
    )?;
    Ok(flattened)
}

/// random-fetching temporal data with column contiguous compact data & compact columns.
///
/// # constants
///
/// - `Nd`: total number of the dates
/// - `Sn`: number of columns of day 'n'. Notice that `len(Sn) = Nd`
/// - `S`: total number of columns, equals to `sum(Sn)`
/// - `T`: number of ticks per day, we assume that the number of ticks per day remains the same.
///
/// # arguments
///
/// - `datetime_start` - the start of the datetime range to fetch.
/// - `datetime_end` - the end of the datetime range to fetch.
/// - `datetime_len` - the length of the datetime range to fetch.
/// - `columns` - the columns to fetch.
/// - `num_ticks_per_day` - equals to `T`.
/// - `full_index` - the full datetime index.
/// - `date_columns_offset` (`Nd + 1`, equals to `[0, *cumsum(Sn)]`) - the offset of each date.
/// - `compact_columns` (`S`) - the full, compact columns.
/// - `compact_data` (`T * S`) - the full, compact data.
pub fn shm_column_contiguous<T: AFloat>(
    datetime_start: i64,
    datetime_end: i64,
    datetime_len: i64,
    columns: ArrayView1<ColumnsDtype>,
    num_ticks_per_day: i64,
    full_index: ArrayView1<i64>,
    date_columns_offset: ArrayView1<i64>,
    compact_columns: ArrayView1<ColumnsDtype>,
    compact_data: ArrayView1<T>,
) -> Result<Vec<T>> {
    let mut flattened = vec![T::zero(); datetime_len as usize * columns.len()];
    let compact_data = compact_data.view();
    column_contiguous(
        None,
        datetime_start,
        datetime_end,
        datetime_len,
        columns,
        num_ticks_per_day,
        full_index,
        date_columns_offset,
        &|start_idx, end_idx| compact_columns.slice(s![start_idx as isize..end_idx as isize]),
        &CachedGetter::new(columns.view()),
        &SHMFetcher::new(compact_data),
        &mut UnsafeSlice::new(&mut flattened),
        None,
        None,
    )?;
    Ok(flattened)
}

struct SHMFetcherGetter<'a, T: AFloat>(&'a [ArrayView1<'a, T>]);
impl<'a, T: AFloat> FetcherGetter<'a, T> for SHMFetcherGetter<'a, T> {
    fn get<'b>(&self, c: usize) -> Box<dyn Fetcher<T> + 'b>
    where
        'a: 'b,
    {
        Box::new(SHMFetcher::new(self.0[c].view()))
    }
}
/// a batched version of `shm_column_contiguous`.
pub fn shm_batch_column_contiguous<'a, T: AFloat>(
    datetime_start: &[i64],
    datetime_end: &[i64],
    datetime_len: i64,
    columns: &'a [ArrayView1<'a, ColumnsDtype>],
    num_ticks_per_day: i64,
    full_index: &[ArrayView1<i64>],
    date_columns_offset: &[ArrayView1<i64>],
    compact_columns: &[ArrayView1<ColumnsDtype>],
    compact_data: &'a [ArrayView1<'a, T>],
    num_threads: usize,
) -> Result<Vec<T>> {
    let fetcher_getter = SHMFetcherGetter(compact_data);
    batch_column_contiguous(
        datetime_start,
        datetime_end,
        datetime_len,
        columns,
        num_ticks_per_day,
        full_index,
        date_columns_offset,
        compact_columns,
        &fetcher_getter,
        num_threads,
    )
}

#[cfg(feature = "io-mem-redis")]
struct RedisFetcherGetter<'a, T: AFloat>(&'a RedisClient<T>, &'a [ArrayView1<'a, RedisKey>]);
#[cfg(feature = "io-mem-redis")]
impl<'a, T: AFloat> FetcherGetter<'a, T> for RedisFetcherGetter<'a, T> {
    fn get<'b>(&self, _: usize) -> Box<dyn Fetcher<T> + 'b>
    where
        'a: 'b,
    {
        Box::new(RedisFetcher::new(
            self.0,
            None,
            RedisKeyRepr::Batched(self.1),
        ))
    }
}
/// similar to `shm_batch_column_contiguous`, but with a redis client.
#[cfg(feature = "io-mem-redis")]
pub fn redis_column_contiguous<'a, T: AFloat>(
    datetime_start: &[i64],
    datetime_end: &[i64],
    datetime_len: i64,
    columns: &'a [ArrayView1<'a, ColumnsDtype>],
    num_ticks_per_day: i64,
    full_index: &[ArrayView1<i64>],
    date_columns_offset: &[ArrayView1<i64>],
    compact_columns: &[ArrayView1<ColumnsDtype>],
    redis_keys: &'a [ArrayView1<'a, RedisKey>],
    redis_client: &'a RedisClient<T>,
    num_threads: usize,
) -> Result<Vec<T>> {
    let fetcher_getter = RedisFetcherGetter(redis_client, redis_keys);
    batch_column_contiguous(
        datetime_start,
        datetime_end,
        datetime_len,
        columns,
        num_ticks_per_day,
        full_index,
        date_columns_offset,
        compact_columns,
        &fetcher_getter,
        num_threads,
    )
}

#[cfg(feature = "io-mem-redis")]
struct GroupedRedisFetcherGetter<'a, T: AFloat> {
    use_batch: bool,
    multipliers: &'a [i64],
    redis_client: &'a RedisClient<T>,
    redis_keys: &'a [ArrayView1<'a, RedisKey>],
}
#[cfg(feature = "io-mem-redis")]
impl<'a, T: AFloat> FetcherGetter<'a, T> for GroupedRedisFetcherGetter<'a, T> {
    fn get<'b>(&self, g: usize) -> Box<dyn Fetcher<T> + 'b>
    where
        'a: 'b,
    {
        let redis_keys = if self.use_batch {
            RedisKeyRepr::Batched(self.redis_keys)
        } else {
            RedisKeyRepr::Single(self.redis_keys[g])
        };
        Box::new(RedisFetcher::new(
            self.redis_client,
            Some(self.multipliers[g]),
            redis_keys,
        ))
    }
}
/// a grouped version of `redis_column_contiguous`.
#[cfg(feature = "io-mem-redis")]
pub fn redis_grouped_column_contiguous<'a, T: AFloat>(
    datetime_start: &[i64],
    datetime_end: &[i64],
    datetime_len: i64,
    columns: &[ArrayView1<ColumnsDtype>],
    num_ticks_per_day: i64,
    full_index: &[ArrayView1<i64>],
    date_columns_offset: &[ArrayView1<i64>],
    compact_columns: &[ArrayView1<ColumnsDtype>],
    redis_keys: &'a [ArrayView1<'a, RedisKey>],
    redis_client: &'a RedisClient<T>,
    multipliers: &'a [i64],
    use_batch: bool,
    num_columns_chunks: usize,
    num_threads: usize,
) -> Result<Vec<T>> {
    let fetcher_getter = GroupedRedisFetcherGetter {
        use_batch,
        multipliers,
        redis_client,
        redis_keys,
    };
    batch_grouped_column_contiguous(
        datetime_start,
        datetime_end,
        datetime_len,
        columns,
        num_ticks_per_day,
        full_index,
        date_columns_offset,
        compact_columns,
        &fetcher_getter,
        multipliers,
        use_batch,
        num_columns_chunks,
        num_threads,
    )
}

#[cfg(test)]
mod tests {
    use crate::df::COLUMNS_NBYTES;

    use super::*;
    use numpy::ndarray::prelude::*;

    fn to_columns_array(columns: Array1<i32>) -> Array1<ColumnsDtype> {
        columns.mapv_into_any(|x| {
            let mut out = [0u8; COLUMNS_NBYTES];
            let x_bytes = i32::to_le_bytes(x);
            out.as_mut_slice()[..4].copy_from_slice(&x_bytes);
            ColumnsDtype::from(out)
        })
    }

    fn assert_vec_eq(va: Array2<f32>, vb: Array2<f32>) {
        if va.shape() != vb.shape() {
            panic!("shape not equal: {:?} != {:?}", va.shape(), vb.shape());
        }
        let mut eq = true;
        for (a, b) in zip(va.iter(), vb.iter()) {
            if a.is_nan() != b.is_nan() {
                eq = false;
                break;
            }
            if !a.is_nan() && a != b {
                eq = false;
                break;
            }
        }
        if !eq {
            panic!("not equal:\n{:#?}\n!=\n{:#?}", va, vb);
        }
    }

    #[test]
    fn test_shm_row_contiguous() {
        let datetime_start = 1;
        let datetime_end = 8;
        let datetime_len = 6;
        let columns = to_columns_array(array![0, 1, 2, 3]);
        let num_ticks_per_day = 2;
        let full_index = array![0, 1, 2, 3, 4, 6, 8, 10];
        let date_columns_offset = array![0, 4, 9, 15, 22];
        let compact_columns = to_columns_array(array![
            0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6
        ]);
        let compact_data = Array::from_iter((0..44).map(|x| x as f32));

        let flattened = shm_row_contiguous(
            datetime_start,
            datetime_end,
            datetime_len,
            columns.view(),
            num_ticks_per_day,
            full_index.view(),
            date_columns_offset.view(),
            compact_columns.view(),
            compact_data.view(),
        )
        .unwrap();
        let result = Array::from_shape_vec((6, 4), flattened).unwrap();
        assert_vec_eq(
            result,
            array![
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [13., 14., 15., 16.],
                [18., 19., 20., 21.],
                [24., 25., 26., 27.],
                [30., 31., 32., 33.],
            ],
        );

        let columns = to_columns_array(array![1, 2, 3, 4]);
        let flattened = shm_row_contiguous(
            datetime_start,
            datetime_end,
            datetime_len,
            columns.view(),
            num_ticks_per_day,
            full_index.view(),
            date_columns_offset.view(),
            compact_columns.view(),
            compact_data.view(),
        )
        .unwrap();
        let result = Array::from_shape_vec((6, 4), flattened).unwrap();
        assert_vec_eq(
            result,
            array![
                [5., 6., 7., f32::NAN],
                [9., 10., 11., 12.],
                [14., 15., 16., 17.],
                [19., 20., 21., 22.],
                [25., 26., 27., 28.],
                [31., 32., 33., 34.],
            ],
        );

        let columns = to_columns_array(array![2, 3, 4, 5]);
        let flattened = shm_row_contiguous(
            datetime_start,
            datetime_end,
            datetime_len,
            columns.view(),
            num_ticks_per_day,
            full_index.view(),
            date_columns_offset.view(),
            compact_columns.view(),
            compact_data.view(),
        )
        .unwrap();
        let result = Array::from_shape_vec((6, 4), flattened).unwrap();
        assert_vec_eq(
            result,
            array![
                [6., 7., f32::NAN, f32::NAN],
                [10., 11., 12., f32::NAN],
                [15., 16., 17., f32::NAN],
                [20., 21., 22., 23.],
                [26., 27., 28., 29.],
                [32., 33., 34., 35.],
            ],
        );
    }

    #[test]
    fn test_shm_column_contiguous_data() {
        let datetime_start = 1;
        let datetime_end = 8;
        let datetime_len = 6;
        let columns = to_columns_array(array![0, 1, 2, 3]);
        let num_ticks_per_day = 2;
        let full_index = array![0, 1, 2, 3, 4, 6, 8, 10];
        let date_columns_offset = array![0, 4, 9, 15, 22];
        let compact_symbols = to_columns_array(array![
            0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6
        ]);
        let compact_data = Array::from_iter((0..44).map(|x| x as f32));

        let flattened = shm_column_contiguous(
            datetime_start,
            datetime_end,
            datetime_len,
            columns.view(),
            num_ticks_per_day,
            full_index.view(),
            date_columns_offset.view(),
            compact_symbols.view(),
            compact_data.view(),
        )
        .unwrap();
        let result = Array::from_shape_vec((4, 6), flattened)
            .unwrap()
            .t()
            .to_owned();
        assert_vec_eq(
            result,
            array![
                [1.0, 3.0, 5.0, 7.0],
                [8.0, 10.0, 12.0, 14.0],
                [9.0, 11.0, 13.0, 15.0],
                [18.0, 20.0, 22.0, 24.0],
                [19.0, 21.0, 23.0, 25.0],
                [30.0, 32.0, 34.0, 36.0],
            ],
        );

        let columns = to_columns_array(array![1, 2, 3, 4]);
        let flattened = shm_column_contiguous(
            datetime_start,
            datetime_end,
            datetime_len,
            columns.view(),
            num_ticks_per_day,
            full_index.view(),
            date_columns_offset.view(),
            compact_symbols.view(),
            compact_data.view(),
        )
        .unwrap();
        let result = Array::from_shape_vec((4, 6), flattened)
            .unwrap()
            .t()
            .to_owned();
        assert_vec_eq(
            result,
            array![
                [3.0, 5.0, 7.0, f32::NAN],
                [10.0, 12.0, 14.0, 16.0],
                [11.0, 13.0, 15.0, 17.0],
                [20.0, 22.0, 24.0, 26.0],
                [21.0, 23.0, 25.0, 27.0],
                [32.0, 34.0, 36.0, 38.0],
            ],
        );

        let columns = to_columns_array(array![2, 3, 4, 5]);
        let flattened = shm_column_contiguous(
            datetime_start,
            datetime_end,
            datetime_len,
            columns.view(),
            num_ticks_per_day,
            full_index.view(),
            date_columns_offset.view(),
            compact_symbols.view(),
            compact_data.view(),
        )
        .unwrap();
        let result = Array::from_shape_vec((4, 6), flattened)
            .unwrap()
            .t()
            .to_owned();
        let gt = array![
            [5.0, 7.0, f32::NAN, f32::NAN],
            [12.0, 14.0, 16.0, f32::NAN],
            [13.0, 15.0, 17.0, f32::NAN],
            [22.0, 24.0, 26.0, 28.0],
            [23.0, 25.0, 27.0, 29.0],
            [34.0, 36.0, 38.0, 40.0],
        ];
        assert_vec_eq(result, gt.clone());
        let flattened = shm_batch_column_contiguous(
            &[datetime_start],
            &[datetime_end],
            datetime_len,
            &[columns.view()],
            num_ticks_per_day,
            &[full_index.view(), full_index.view()],
            &[date_columns_offset.view(), date_columns_offset.view()],
            &[compact_symbols.view(), compact_symbols.view()],
            &[compact_data.view(), compact_data.view()],
            1,
        )
        .unwrap();
        let result = Array::from_shape_vec((2, 4, 6), flattened).unwrap();
        for mat in result.axis_iter(Axis(0)) {
            assert_vec_eq(mat.t().to_owned(), gt.clone());
        }
    }
}
