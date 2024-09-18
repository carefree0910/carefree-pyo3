use numpy::ndarray::{ArcArray1, ArcArray2};
use pyo3::prelude::*;

use super::{ColumnsDtype, IndexDtype};

mod indexing;
mod meta;
mod ops;

#[pyclass]
pub struct DataFrameF64 {
    pub index: ArcArray1<IndexDtype>,
    pub columns: ArcArray1<ColumnsDtype>,
    pub data: ArcArray2<f64>,
}
