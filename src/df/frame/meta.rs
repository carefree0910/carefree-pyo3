use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use super::{ColumnsDtype, DataFrameF64, IndexDtype};

#[pymethods]
impl DataFrameF64 {
    #[staticmethod]
    fn new(
        index: PyReadonlyArray1<IndexDtype>,
        columns: PyReadonlyArray1<ColumnsDtype>,
        data: PyReadonlyArray2<f64>,
    ) -> Self {
        DataFrameF64 {
            index: index.as_array().into_owned().into(),
            columns: columns.as_array().into_owned().into(),
            data: data.as_array().into_owned().into(),
        }
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.index.len(), self.columns.len())
    }
}
