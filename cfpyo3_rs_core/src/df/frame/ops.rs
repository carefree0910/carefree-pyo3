use super::DataFrame;
use crate::toolkit::array::{corr_axis1, mean_axis1, AFloat};
use numpy::ndarray::ArrayView2;

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn mean_axis1(&'a self, num_threads: usize) -> Vec<T> {
        mean_axis1(&self.data.view(), num_threads)
    }

    pub fn corr_with_axis1(&'a self, other: ArrayView2<T>, num_threads: usize) -> Vec<T> {
        corr_axis1(&self.data.view(), &other.view(), num_threads)
    }
}
