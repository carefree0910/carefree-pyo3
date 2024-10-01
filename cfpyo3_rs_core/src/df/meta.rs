use super::DataFrame;
use crate::{
    df::{ColumnsDtype, IndexDtype},
    toolkit::array::AFloat,
};
use anyhow::Result;
use numpy::{
    ndarray::{Array1, Array2, ArrayView1, ArrayView2, CowArray},
    Ix1, Ix2,
};

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn new(
        index: CowArray<'a, IndexDtype, Ix1>,
        columns: CowArray<'a, ColumnsDtype, Ix1>,
        values: CowArray<'a, T, Ix2>,
    ) -> Self {
        Self {
            index,
            columns,
            values,
        }
    }

    /// # Safety
    ///
    /// This function requires that:
    /// - pointers are aligned with [`DF_ALIGN`].
    /// - pointers are representing the corresponding data types (i.e., [`IndexDtype`], [`ColumnsDtype`], and `T`).
    /// - the 'owners' of the pointers should NOT be freed before the [`DataFrame`] is dropped.
    pub unsafe fn from_ptr(
        index_ptr: *const u8,
        index_shape: usize,
        columns_ptr: *const u8,
        columns_shape: usize,
        values_ptr: *const u8,
    ) -> Self {
        let index = ArrayView1::<IndexDtype>::from_shape_ptr(
            (index_shape,),
            index_ptr as *const IndexDtype,
        );
        let columns = ArrayView1::<ColumnsDtype>::from_shape_ptr(
            (columns_shape,),
            columns_ptr as *const ColumnsDtype,
        );
        let values =
            ArrayView2::<T>::from_shape_ptr((index_shape, columns_shape), values_ptr as *const T);
        Self::new(index.into(), columns.into(), values.into())
    }

    pub fn from_owned(
        index: Vec<IndexDtype>,
        columns: Vec<ColumnsDtype>,
        values: Vec<T>,
    ) -> Result<Self> {
        let index_shape = index.len();
        let columns_shape = columns.len();
        Ok(Self::new(
            Array1::from_shape_vec((index_shape,), index)?.into(),
            Array1::from_shape_vec((columns_shape,), columns)?.into(),
            Array2::from_shape_vec((index_shape, columns_shape), values)?.into(),
        ))
    }

    pub fn to_owned(self) -> Self {
        let index = self.index.into_owned();
        let columns = self.columns.into_owned();
        let values = self.values.into_owned();
        Self::new(index.into(), columns.into(), values.into())
    }
}

pub const DF_ALIGN: usize = align_of::<DataFrame<f64>>();
pub fn align_nbytes(nbytes: usize) -> usize {
    let remainder = nbytes % DF_ALIGN;
    if remainder == 0 {
        nbytes
    } else {
        nbytes + DF_ALIGN - remainder
    }
}
