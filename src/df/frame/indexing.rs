use numpy::{
    ndarray::{ArcArray1, Axis},
    PyReadonlyArray1,
};
use pyo3::prelude::*;

use super::DataFrameF64;

#[pymethods]
impl DataFrameF64 {
    fn rows(&self, indices: PyReadonlyArray1<i64>) -> DataFrameF64 {
        let indices: Vec<usize> = indices.as_array().iter().map(|&x| x as usize).collect();
        let indices = indices.as_slice();
        let index = self.index.select(Axis(0), indices);
        let values = self.data.select(Axis(0), indices);
        DataFrameF64 {
            index: index.into(),
            columns: ArcArray1::clone(&self.columns),
            data: values.into(),
        }
    }
}
