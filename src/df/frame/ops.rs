use std::borrow::Borrow;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::toolkit::array::{corr_axis1, mean_axis1};

use super::DataFrameF64;

#[pymethods]
impl DataFrameF64 {
    fn mean_axis1<'a>(&'a self, py: Python<'a>) -> Bound<'a, PyArray1<f64>> {
        let data = self.get_data_array(py);
        mean_axis1(data.borrow(), 8).into_pyarray_bound(py)
    }

    fn corr_with_axis1<'a>(
        &'a self,
        py: Python<'a>,
        other: PyReadonlyArray2<f64>,
    ) -> Bound<'a, PyArray1<f64>> {
        let data = self.get_data_array(py);
        let other = other.as_array();
        corr_axis1(data.borrow(), other.borrow(), 8).into_pyarray_bound(py)
    }
}
