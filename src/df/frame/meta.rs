use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
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
    fn index<'a>(&'a self, py: Python<'a>) -> Bound<PyArray1<IndexDtype>> {
        self.index.to_pyarray_bound(py)
    }

    #[getter]
    fn columns<'a>(&'a self, py: Python<'a>) -> Bound<PyArray1<ColumnsDtype>> {
        self.columns.to_pyarray_bound(py)
    }

    #[getter]
    fn values<'a>(&'a self, py: Python<'a>) -> Bound<PyArray2<f64>> {
        self.data.to_pyarray_bound(py)
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.index.len(), self.columns.len())
    }
}
