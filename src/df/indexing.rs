use numpy::{
    ndarray::{ArcArray1, Axis},
    PyReadonlyArray1,
};
use pyo3::prelude::*;

use super::DataFrameF64;

#[pyfunction]
pub fn rows(df: &DataFrameF64, indices: PyReadonlyArray1<i64>) -> DataFrameF64 {
    let indices = indices
        .as_array()
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<_>>();
    let indices = indices.as_slice();
    let index = df.index.select(Axis(0), indices);
    let columns = ArcArray1::clone(&df.columns);
    let data = df.data.select(Axis(0), indices);
    DataFrameF64 {
        index: index.into(),
        columns,
        data: data.into(),
    }
}
