use super::DataFrameF64;
use cfpyo3_core::df::{ColumnsDtype, DataFrame, IndexDtype};
use numpy::{
    ndarray::{ArrayView1, ArrayView2},
    PyArray1, PyArray2, PyArrayMethods, ToPyArray,
};
use pyo3::prelude::*;

impl DataFrameF64 {
    pub(crate) fn get_index_array<'a>(&'a self, py: Python<'a>) -> ArrayView1<'a, IndexDtype> {
        unsafe { self.index.bind(py).as_array() }
    }
    pub(crate) fn get_columns_array<'a>(&'a self, py: Python<'a>) -> ArrayView1<'a, ColumnsDtype> {
        unsafe { self.columns.bind(py).as_array() }
    }
    pub(crate) fn get_values_array<'a>(&'a self, py: Python<'a>) -> ArrayView2<'a, f64> {
        unsafe { self.values.bind(py).as_array() }
    }
    pub(crate) fn to_core<'a>(&'a self, py: Python<'a>) -> DataFrame<'a, f64> {
        let index = self.get_index_array(py);
        let columns = self.get_columns_array(py);
        let values = self.get_values_array(py);
        DataFrame::new(index.into(), columns.into(), values.into())
    }
    pub(crate) fn from_core(py: Python, df: DataFrame<f64>) -> Self {
        DataFrameF64 {
            index: df.index.to_pyarray_bound(py).unbind(),
            columns: df.columns.to_pyarray_bound(py).unbind(),
            values: df.values.to_pyarray_bound(py).unbind(),
        }
    }
}

#[pymethods]
impl DataFrameF64 {
    #[staticmethod]
    fn new(
        index: Py<PyArray1<IndexDtype>>,
        columns: Py<PyArray1<ColumnsDtype>>,
        values: Py<PyArray2<f64>>,
    ) -> Self {
        DataFrameF64 {
            index,
            columns,
            values,
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
        self.values.clone_ref(py)
    }

    #[getter]
    fn shape(&self, py: Python) -> (usize, usize) {
        (
            self.get_index_array(py).len(),
            self.get_columns_array(py).len(),
        )
    }

    fn with_data(&self, py: Python, values: Py<PyArray2<f64>>) -> Self {
        DataFrameF64 {
            index: self.index.clone_ref(py),
            columns: self.columns.clone_ref(py),
            values,
        }
    }
}
