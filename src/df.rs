use numpy::{
    datetime::{units::Nanoseconds, Datetime},
    ndarray::{ArcArray1, ArcArray2},
    PyFixedString,
};
use pyo3::prelude::*;

pub const INDEX_CHAR_LEN: usize = 256;
pub type IndexDtype = Datetime<Nanoseconds>;
pub type ColumnsDtype = PyFixedString<INDEX_CHAR_LEN>;

pub mod indexing;
pub mod meta;

#[pyclass]
pub struct DataFrameF64 {
    pub index: ArcArray1<IndexDtype>,
    pub columns: ArcArray1<ColumnsDtype>,
    pub data: ArcArray2<f64>,
}
