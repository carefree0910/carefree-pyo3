use super::{meta::WithCore, DataFrameF64, OwnedDataFrameF64};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

pub trait Ops: WithCore {
    fn nanmean_axis1<'py>(
        &'py self,
        py: Python<'py>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        self.to_core(py)
            .nanmean_axis1(num_threads)
            .into_pyarray_bound(py)
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
            .into_pyarray_bound(py)
    }
}

macro_rules! ops_bindings_impl {
    ($type:ty) => {
        impl Ops for $type {}

        #[pymethods]
        impl $type {
            fn nanmean_axis1<'py>(
                &'py self,
                py: Python<'py>,
                num_threads: Option<usize>,
            ) -> Bound<'py, PyArray1<f64>> {
                Ops::nanmean_axis1(self, py, num_threads)
            }
            fn nancorr_with_axis1<'py>(
                &'py self,
                py: Python<'py>,
                other: PyReadonlyArray2<f64>,
                num_threads: Option<usize>,
            ) -> Bound<'py, PyArray1<f64>> {
                Ops::nancorr_with_axis1(self, py, other, num_threads)
            }
        }
    };
}

ops_bindings_impl!(DataFrameF64);
ops_bindings_impl!(OwnedDataFrameF64);
