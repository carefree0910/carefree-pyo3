use super::DataFrame;
use crate::{
    df::{ColumnsDtype, IndexDtype},
    toolkit::array::AFloat,
};
use numpy::ndarray::{ArrayView1, ArrayView2};

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn new(
        index: ArrayView1<'a, IndexDtype>,
        columns: ArrayView1<'a, ColumnsDtype>,
        data: ArrayView2<'a, T>,
    ) -> Self {
        Self {
            index,
            columns,
            data,
        }
    }
}
