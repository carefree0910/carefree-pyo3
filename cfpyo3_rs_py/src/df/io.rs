use super::{ArcDataFrameF64, DataFrameF64, IOs};
use cfpyo3_core::df::DataFrame;
use pyo3::prelude::*;

impl IOs for DataFrameF64 {
    fn save(&self, py: Python, path: &str) -> PyResult<()> {
        self.to_core(py)
            .save(path)
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }

    fn load(py: Python, path: &str) -> PyResult<Self> {
        Ok(DataFrameF64::from_core(
            py,
            DataFrame::load(path).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?,
        ))
    }
}

impl IOs for ArcDataFrameF64 {
    fn save(&self, _: Python, path: &str) -> PyResult<()> {
        self.to_core()
            .save(path)
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }

    fn load(_: Python, path: &str) -> PyResult<Self> {
        Ok(ArcDataFrameF64::from_core(
            DataFrame::load(path).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?,
        ))
    }
}

// bindings

#[pymethods]
impl DataFrameF64 {
    fn save(&self, py: Python, path: &str) -> PyResult<()> {
        IOs::save(self, py, path)
    }
    #[staticmethod]
    fn load(py: Python, path: &str) -> PyResult<Self> {
        <DataFrameF64 as IOs>::load(py, path)
    }
}
#[pymethods]
impl ArcDataFrameF64 {
    fn save(&self, py: Python, path: &str) -> PyResult<()> {
        IOs::save(self, py, path)
    }
    #[staticmethod]
    fn load(py: Python, path: &str) -> PyResult<Self> {
        <ArcDataFrameF64 as IOs>::load(py, path)
    }
}
