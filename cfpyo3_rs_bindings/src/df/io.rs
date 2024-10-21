use super::DataFrameF64;
use cfpyo3_core::df::DataFrame;
use pyo3::prelude::*;

#[pymethods]
impl DataFrameF64 {
    fn save(&self, py: Python, path: &str) -> PyResult<()> {
        self.to_core(py).save(path)?;
        Ok(())
    }

    #[staticmethod]
    fn load(py: Python, path: &str) -> PyResult<Self>
    where
        Self: Sized,
    {
        Ok(Self::from_core(py, DataFrame::load(path)?))
    }
}
