//! # df
//!
//! a DataFrame module that mainly focuses on temporal data

use crate::toolkit::array::AFloat;
use numpy::{
    datetime::{units::Nanoseconds, Datetime},
    ndarray::{Array1, Array2, ArrayView1, ArrayView2},
    PyFixedString,
};

mod io;
mod meta;
mod ops;

pub const COLUMNS_NBYTES: usize = 32;
pub type IndexDtype = Datetime<Nanoseconds>;
pub type ColumnsDtype = PyFixedString<COLUMNS_NBYTES>;
pub const INDEX_NBYTES: usize = core::mem::size_of::<IndexDtype>();

#[derive(Debug)]
pub struct OwnedDataFrame<T: AFloat> {
    pub index: Array1<IndexDtype>,
    pub columns: Array1<ColumnsDtype>,
    pub values: Array2<T>,
}
#[derive(Debug)]
pub struct DataFrameView<'a, T: AFloat> {
    pub index: ArrayView1<'a, IndexDtype>,
    pub columns: ArrayView1<'a, ColumnsDtype>,
    pub values: ArrayView2<'a, T>,
}

pub enum DataFrame<'a, T: AFloat> {
    View(DataFrameView<'a, T>),
    Owned(OwnedDataFrame<T>),
}
