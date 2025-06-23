use super::DataFrameF64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pymethods]
impl DataFrameF64 {
    fn nanmean_axis1<'py>(
        &'py self,
        py: Python<'py>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        self.to_core(py).nanmean_axis1(num_threads).into_pyarray(py)
    }

    fn nancorr_with_axis1<'py>(
        &'py self,
        py: Python<'py>,
        other: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let other = other.as_array();
        self.to_core(py)
            .nancorr_with_axis1(other, num_threads)
            .into_pyarray(py)
    }
}
