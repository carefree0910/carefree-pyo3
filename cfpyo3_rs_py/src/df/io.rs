use super::{meta::WithCore, ArcDataFrameF64, DataFrameF64};
use cfpyo3_core::df::DataFrame;
use pyo3::prelude::*;

pub(super) trait IOs: WithCore {
    fn save(&self, py: Python, path: &str) -> PyResult<()> {
        self.to_core(py)
            .save(path)
            .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)
    }

    fn load(py: Python, path: &str) -> PyResult<Self>
    where
        Self: Sized,
    {
        Ok(Self::from_core(
            py,
            DataFrame::load(path).map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?,
        ))
    }
}

macro_rules! ios_bindings_impl {
    ($type:ty) => {
        impl IOs for $type {}

        #[pymethods]
        impl $type {
            fn save(&self, py: Python, path: &str) -> PyResult<()> {
                IOs::save(self, py, path)
            }
            #[staticmethod]
            fn load(py: Python, path: &str) -> PyResult<Self> {
                IOs::load(py, path)
            }
        }
    };
}

ios_bindings_impl!(DataFrameF64);
ios_bindings_impl!(ArcDataFrameF64);