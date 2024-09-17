use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use super::{ColumnsDtype, DataFrameF64, IndexDtype};

#[pyfunction]
pub fn new(
    index: PyReadonlyArray1<IndexDtype>,
    columns: PyReadonlyArray1<ColumnsDtype>,
    data: PyReadonlyArray2<f64>,
) -> DataFrameF64 {
    DataFrameF64 {
        index: index.as_array().into_owned().into(),
        columns: columns.as_array().into_owned().into(),
        data: data.as_array().into_owned().into(),
    }
}

#[pyfunction]
pub fn shape(df: &DataFrameF64) -> (usize, usize) {
    (df.index.len(), df.columns.len())
}
