use numpy::{
    ndarray::{ArcArray1, Axis},
    PyReadonlyArray1,
};
use pyo3::prelude::*;

use super::DataFrameF64;

#[pymethods]
impl DataFrameF64 {
    fn rows(&self, indices: PyReadonlyArray1<i64>) -> DataFrameF64 {
        let indices = indices
            .as_array()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        let indices = indices.as_slice();
        let index = self.index.select(Axis(0), indices);
        let columns = ArcArray1::clone(&self.columns);
        let data = self.data.select(Axis(0), indices);
        DataFrameF64 {
            index: index.into(),
            columns,
            data: data.into(),
        }
    }
}
