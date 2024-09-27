use super::DataFrame;
use crate::{
    df::{ColumnsDtype, IndexDtype},
    toolkit::array::AFloat,
};
use numpy::{
    ndarray::{ArrayView1, ArrayView2, CowArray},
    Ix1, Ix2,
};

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
    pub unsafe fn from(
        index_ptr: *const u8,
        index_shape: usize,
        columns_ptr: *const u8,
        columns_shape: usize,
        data_ptr: *const u8,
    ) -> Self {
        let index_array = ArrayView1::<IndexDtype>::from_shape_ptr(
            (index_shape,),
            index_ptr as *const IndexDtype,
        );
        let columns_array = ArrayView1::<ColumnsDtype>::from_shape_ptr(
            (columns_shape,),
            columns_ptr as *const ColumnsDtype,
        );
        let data_array =
            ArrayView2::<T>::from_shape_ptr((index_shape, columns_shape), data_ptr as *const T);
        Self::new(index_array.into(), columns_array.into(), data_array.into())
    }
}
