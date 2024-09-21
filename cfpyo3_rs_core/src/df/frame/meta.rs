use super::DataFrame;
use crate::{
    df::{ColumnsDtype, IndexDtype},
    toolkit::array::AFloat,
};
use numpy::{ndarray::CowArray, Ix1, Ix2};

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn new(
        index: CowArray<'a, IndexDtype, Ix1>,
        columns: CowArray<'a, ColumnsDtype, Ix1>,
        data: CowArray<'a, T, Ix2>,
    ) -> Self {
        Self {
            index,
            columns,
            data,
        }
    }
}
