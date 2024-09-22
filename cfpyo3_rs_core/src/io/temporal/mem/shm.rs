use super::{Fetcher, FetcherArgs};
use crate::toolkit::array::AFloat;
use numpy::{
    ndarray::{s, ArrayView1, CowArray},
    Ix1,
};

pub struct SHMFetcher<'a, T: AFloat> {
    data: &'a ArrayView1<'a, T>,
}

impl<'a, T: AFloat> SHMFetcher<'a, T> {
    pub fn new(data: &'a ArrayView1<'a, T>) -> SHMFetcher<'a, T> {
        SHMFetcher { data }
    }
}

impl<'a, T: AFloat> Fetcher<T> for SHMFetcher<'a, T> {
    fn fetch(&self, args: FetcherArgs) -> CowArray<T, Ix1> {
        self.data
            .slice(s![args.start_idx as isize..args.end_idx as isize])
            .into()
    }
}

pub struct SlicedSHMFetcher<'a, T: AFloat> {
    data_slices: &'a Vec<&'a ArrayView1<'a, T>>,
    multiplier: Option<i64>,
}

impl<'a, T: AFloat> SlicedSHMFetcher<'a, T> {
    pub fn new(
        data_slices: &'a Vec<&'a ArrayView1<'a, T>>,
        multiplier: Option<i64>,
    ) -> SlicedSHMFetcher<'a, T> {
        SlicedSHMFetcher {
            data_slices,
            multiplier,
        }
    }
}

impl<'a, T: AFloat> Fetcher<T> for SlicedSHMFetcher<'a, T> {
    fn fetch(&self, args: FetcherArgs) -> CowArray<T, Ix1> {
        let data = self.data_slices[args.date_idx as usize];
        let mut start_idx = args.time_start_idx;
        let mut end_idx = args.time_end_idx;
        if let Some(multiplier) = self.multiplier {
            start_idx *= multiplier;
            end_idx *= multiplier;
        }
        data.slice(s![start_idx as isize..end_idx as isize]).into()
    }
}
