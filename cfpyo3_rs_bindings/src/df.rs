//! A DataFrame binding from [`cfpyo3_core::df::DataFrame`] to Python using [`pyo3`].

use cfpyo3_core::df::{ColumnsDtype, IndexDtype};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

pub mod io;
pub mod meta;
pub mod ops;

#[pyclass(frozen)]
pub struct DataFrameF64 {
    pub index: Py<PyArray1<IndexDtype>>,
    pub columns: Py<PyArray1<ColumnsDtype>>,
    pub values: Py<PyArray2<f64>>,
}
