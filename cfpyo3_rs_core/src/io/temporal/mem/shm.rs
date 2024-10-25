use super::{Fetcher, FetcherArgs};
use crate::toolkit::array::AFloat;
use anyhow::Result;
use numpy::{
    ndarray::{s, ArrayView1, CowArray},
    Ix1,
};

pub struct SHMFetcher<'a, T: AFloat> {
    data: ArrayView1<'a, T>,
}

impl<'a, T: AFloat> SHMFetcher<'a, T> {
    pub fn new(data: ArrayView1<'a, T>) -> SHMFetcher<'a, T> {
        SHMFetcher { data }
    }
}

impl<'a, T: AFloat> Fetcher<T> for SHMFetcher<'a, T> {
    fn fetch(&self, args: FetcherArgs) -> Result<CowArray<T, Ix1>> {
        Ok(self
            .data
            .slice(s![args.start_idx as isize..args.end_idx as isize])
            .into())
    }
}
