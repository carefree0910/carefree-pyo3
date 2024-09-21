use super::DataFrameF64;
use cfpyo3_core::df::frame::DataFrame;
use cfpyo3_core::df::{ColumnsDtype, IndexDtype};
use numpy::{
    ndarray::{ArrayView1, ArrayView2},
    PyArray1, PyArray2, PyArrayMethods,
};
use pyo3::prelude::*;

impl DataFrameF64 {
    pub fn get_index_array<'a>(&'a self, py: Python<'a>) -> ArrayView1<'a, IndexDtype> {
        unsafe { self.index.bind(py).as_array() }
    }
    pub fn get_columns_array<'a>(&'a self, py: Python<'a>) -> ArrayView1<'a, ColumnsDtype> {
        unsafe { self.columns.bind(py).as_array() }
    }
    pub fn get_data_array<'a>(&'a self, py: Python<'a>) -> ArrayView2<'a, f64> {
        unsafe { self.data.bind(py).as_array() }
    }
    pub fn to_core<'a>(&'a self, py: Python<'a>) -> DataFrame<'a, f64> {
        let index = self.get_index_array(py);
        let columns = self.get_columns_array(py);
        let data = self.get_data_array(py);
        DataFrame::<f64> {
            index,
            columns,
            data,
        }
    }
}

#[pymethods]
impl DataFrameF64 {
    #[staticmethod]
    fn new(
        index: Py<PyArray1<IndexDtype>>,
        columns: Py<PyArray1<ColumnsDtype>>,
        data: Py<PyArray2<f64>>,
    ) -> Self {
        DataFrameF64 {
            index,
            columns,
            data,
        }
    }

    #[getter]
    fn index(&self, py: Python) -> Py<PyArray1<IndexDtype>> {
        self.index.clone_ref(py)
    }

    #[getter]
    fn columns(&self, py: Python) -> Py<PyArray1<ColumnsDtype>> {
        self.columns.clone_ref(py)
    }

    #[getter]
    fn values(&self, py: Python) -> Py<PyArray2<f64>> {
        self.data.clone_ref(py)
    }

    #[getter]
    fn shape(&self, py: Python) -> (usize, usize) {
        (
            self.get_index_array(py).len(),
            self.get_columns_array(py).len(),
        )
    }

    fn with_data(&self, py: Python, data: Py<PyArray2<f64>>) -> Self {
        DataFrameF64 {
            index: self.index.clone_ref(py),
            columns: self.columns.clone_ref(py),
            data,
        }
    }
}
