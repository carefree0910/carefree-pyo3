//! # df
//!
//! a DataFrame module that mainly focuses on temporal data

use crate::toolkit::array::AFloat;
use numpy::{
    datetime::{units::Nanoseconds, Datetime},
    ndarray::CowArray,
    Ix1, Ix2, PyFixedString,
};

mod io;
mod meta;
mod ops;

pub const COLUMNS_NBYTES: usize = 32;
pub type IndexDtype = Datetime<Nanoseconds>;
pub type ColumnsDtype = PyFixedString<COLUMNS_NBYTES>;
pub const INDEX_NBYTES: usize = core::mem::size_of::<IndexDtype>();

#[derive(Debug)]
pub struct DataFrame<'a, T: AFloat> {
    pub index: CowArray<'a, IndexDtype, Ix1>,
    pub columns: CowArray<'a, ColumnsDtype, Ix1>,
    pub values: CowArray<'a, T, Ix2>,
}
