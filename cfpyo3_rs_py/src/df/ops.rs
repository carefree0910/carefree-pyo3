use super::DataFrameF64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pymethods]
impl DataFrameF64 {
    fn mean_axis1<'a>(&'a self, py: Python<'a>) -> Bound<'a, PyArray1<f64>> {
        self.to_core(py).mean_axis1(8).into_pyarray_bound(py)
    }

    fn corr_with_axis1<'a>(
        &'a self,
        py: Python<'a>,
        other: PyReadonlyArray2<f64>,
    ) -> Bound<'a, PyArray1<f64>> {
        let other = other.as_array();
        self.to_core(py)
            .corr_with_axis1(other, 8)
            .into_pyarray_bound(py)
    }
}
