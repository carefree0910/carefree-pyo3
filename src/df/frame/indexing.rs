use numpy::{ndarray::Axis, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use super::DataFrameF64;

#[pymethods]
impl DataFrameF64 {
    fn rows(&self, py: Python, indices: PyReadonlyArray1<i64>) -> DataFrameF64 {
        let indices: Vec<usize> = indices.as_array().iter().map(|&x| x as usize).collect();
        let indices = indices.as_slice();
        let index = self.get_index_array(py).select(Axis(0), indices);
        let values = self.get_data_array(py).select(Axis(0), indices);
        DataFrameF64 {
            index: index.to_pyarray_bound(py).unbind(),
            columns: self.columns.clone_ref(py),
            data: values.to_pyarray_bound(py).unbind(),
        }
    }
}
