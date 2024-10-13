use super::DataFrame;
use crate::toolkit::array::{nancorr_axis1, nanmean_axis1, AFloat};
use numpy::ndarray::ArrayView2;

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn nanmean_axis1(&'a self, num_threads: Option<usize>) -> Vec<T> {
        nanmean_axis1(&self.values.view(), num_threads.unwrap_or(0))
    }

    pub fn nancorr_with_axis1(
        &'a self,
        other: ArrayView2<T>,
        num_threads: Option<usize>,
    ) -> Vec<T> {
        nancorr_axis1(&self.values.view(), &other.view(), num_threads.unwrap_or(0))
    }
}
