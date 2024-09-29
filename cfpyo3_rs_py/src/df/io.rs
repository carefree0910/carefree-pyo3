use super::DataFrameF64;
use cfpyo3_core::df::DataFrame;
use pyo3::prelude::*;

#[pymethods]
impl DataFrameF64 {
    fn save(&self, py: Python, path: &str) -> PyResult<()> {
        self.to_core(py)
            .save(path)
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }
    #[staticmethod]
    fn load(py: Python, path: &str) -> PyResult<Self> {
        Ok(DataFrameF64::from_core(
            py,
            DataFrame::load(path).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?,
        ))
    }
}
