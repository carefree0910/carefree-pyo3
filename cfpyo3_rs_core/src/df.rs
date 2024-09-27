//! # df
//!
//! a DataFrame module that mainly focuses on temporal data

use numpy::{
    datetime::{units::Nanoseconds, Datetime},
    PyFixedString,
};

pub const COLUMNS_NBYTES: usize = 32;
pub type IndexDtype = Datetime<Nanoseconds>;
pub type ColumnsDtype = PyFixedString<COLUMNS_NBYTES>;
pub const INDEX_NBYTES: usize = core::mem::size_of::<IndexDtype>();

pub mod frame;
