use super::{DataFrameF64, OwnedDataFrameF64};
use cfpyo3_core::df::{ColumnsDtype, DataFrame, IndexDtype};
use numpy::{
    ndarray::{ArrayView1, ArrayView2},
    PyArray1, PyArray2, PyArrayMethods, ToPyArray,
};
use pyo3::prelude::*;

pub trait WithCore {
    fn to_core<'py>(&'py self, py: Python<'py>) -> DataFrame<'py, f64>;
    fn from_core(py: Python, df: DataFrame<f64>) -> Self
    where
        Self: Sized;
}

impl DataFrameF64 {
    pub(crate) fn get_index_array<'py>(&'py self, py: Python<'py>) -> ArrayView1<'py, IndexDtype> {
        unsafe { self.index.bind(py).as_array() }
    }
    pub(crate) fn get_columns_array<'py>(
        &'py self,
        py: Python<'py>,
    ) -> ArrayView1<'py, ColumnsDtype> {
        unsafe { self.columns.bind(py).as_array() }
    }
    pub(crate) fn get_values_array<'py>(&'py self, py: Python<'py>) -> ArrayView2<'py, f64> {
        unsafe { self.values.bind(py).as_array() }
    }
}

impl WithCore for DataFrameF64 {
    fn to_core<'py>(&'py self, py: Python<'py>) -> DataFrame<'py, f64> {
        let index = self.get_index_array(py);
        let columns = self.get_columns_array(py);
        let values = self.get_values_array(py);
        DataFrame::new(index.into(), columns.into(), values.into())
    }
    fn from_core(py: Python, df: DataFrame<f64>) -> Self {
        DataFrameF64 {
            index: df.index.to_pyarray_bound(py).unbind(),
            columns: df.columns.to_pyarray_bound(py).unbind(),
            values: df.values.to_pyarray_bound(py).unbind(),
        }
    }
}

impl WithCore for OwnedDataFrameF64 {
    fn to_core(&self, _: Python) -> DataFrame<f64> {
        DataFrame::new(
            self.index.view().into(),
            self.columns.view().into(),
            self.values.view().into(),
        )
    }
    fn from_core(_: Python, df: DataFrame<f64>) -> Self {
        OwnedDataFrameF64 {
            index: df
                .index
                .try_into_owned_nocopy()
                .unwrap_or_else(|_| panic!("index is not owned")),
            columns: df
                .columns
                .try_into_owned_nocopy()
                .unwrap_or_else(|_| panic!("columns is not owned")),
            values: df
                .values
                .try_into_owned_nocopy()
                .unwrap_or_else(|_| panic!("values is not owned")),
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

    fn to_owned(&self, py: Python) -> OwnedDataFrameF64 {
        OwnedDataFrameF64 {
            index: self.get_index_array(py).to_owned(),
            columns: self.get_columns_array(py).to_owned(),
            values: self.get_values_array(py).to_owned(),
        }
    }
}

#[pymethods]
impl OwnedDataFrameF64 {
    fn to_py(&self, py: Python) -> DataFrameF64 {
        DataFrameF64 {
            index: self.index.to_pyarray_bound(py).unbind(),
            columns: self.columns.to_pyarray_bound(py).unbind(),
            values: self.values.to_pyarray_bound(py).unbind(),
        }
    }
}
