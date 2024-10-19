use super::{DataFrame, DataFrameView, OwnedDataFrame};
use crate::{
    df::{ColumnsDtype, IndexDtype},
    toolkit::array::AFloat,
};
use anyhow::Result;
use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

impl<'a, T: AFloat> DataFrame<'a, T> {
    // constructors

    pub fn new_view(
        index: ArrayView1<'a, IndexDtype>,
        columns: ArrayView1<'a, ColumnsDtype>,
        values: ArrayView2<'a, T>,
    ) -> Self {
        Self::View(DataFrameView {
            index,
            columns,
            values,
        })
    }
    pub fn new_owned(
        index: Array1<IndexDtype>,
        columns: Array1<ColumnsDtype>,
        values: Array2<T>,
    ) -> Self {
        Self::Owned(OwnedDataFrame {
            index,
            columns,
            values,
        })
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
        Self::new_view(index, columns, values)
    }
    pub fn from_vec(
        index: Vec<IndexDtype>,
        columns: Vec<ColumnsDtype>,
        values: Vec<T>,
    ) -> Result<Self> {
        let index_shape = index.len();
        let columns_shape = columns.len();
        Ok(Self::new_owned(
            Array1::from_shape_vec((index_shape,), index)?,
            Array1::from_shape_vec((columns_shape,), columns)?,
            Array2::from_shape_vec((index_shape, columns_shape), values)?,
        ))
    }

    // getters

    pub fn index(&self) -> ArrayView1<IndexDtype> {
        match self {
            Self::View(df) => df.index.view(),
            Self::Owned(df) => df.index.view(),
        }
    }
    pub fn columns(&self) -> ArrayView1<ColumnsDtype> {
        match self {
            Self::View(df) => df.columns.view(),
            Self::Owned(df) => df.columns.view(),
        }
    }
    pub fn values(&self) -> ArrayView2<T> {
        match self {
            Self::View(df) => df.values.view(),
            Self::Owned(df) => df.values.view(),
        }
    }

    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }
    pub fn into_owned(self) -> Self {
        match self {
            Self::Owned(df) => Self::Owned(df),
            Self::View(df) => Self::new_owned(
                df.index.to_owned(),
                df.columns.to_owned(),
                df.values.to_owned(),
            ),
        }
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
