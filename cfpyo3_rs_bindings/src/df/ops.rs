use super::{meta::WithCore, DataFrameF64, OwnedDataFrameF64};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

pub trait Ops: WithCore {
    fn mean_axis1<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.to_core(py).mean_axis1(8).into_pyarray_bound(py)
    }

    fn corr_with_axis1<'py>(
        &'py self,
        py: Python<'py>,
        other: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let other = other.as_array();
        self.to_core(py)
            .corr_with_axis1(other, 8)
            .into_pyarray_bound(py)
    }
}

macro_rules! ops_bindings_impl {
    ($type:ty) => {
        impl Ops for $type {}

        #[pymethods]
        impl $type {
            fn mean_axis1<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                Ops::mean_axis1(self, py)
            }
            fn corr_with_axis1<'py>(
                &'py self,
                py: Python<'py>,
                other: PyReadonlyArray2<f64>,
            ) -> Bound<'py, PyArray1<f64>> {
                Ops::corr_with_axis1(self, py, other)
            }
        }
    };
}

ops_bindings_impl!(DataFrameF64);
ops_bindings_impl!(OwnedDataFrameF64);
