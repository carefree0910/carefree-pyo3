use cfpyo3_core::toolkit::array;
use numpy::{ndarray::ArrayView2, IntoPyArray, PyArray1};
use pyo3::prelude::*;

macro_rules! fast_concat_2d_axis0_impl {
    ($name:ident, $dtype:ty, $multiplier:expr) => {
        pub fn $name<'py>(
            py: Python<'py>,
            arrays: Vec<ArrayView2<$dtype>>,
        ) -> Bound<'py, PyArray1<$dtype>> {
            let num_rows: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
            let num_columns = arrays[0].shape()[1];
            for array in &arrays {
                if array.shape()[1] != num_columns {
                    panic!("all arrays should have same number of columns");
                }
            }
            let num_total_rows: usize = num_rows.iter().sum();
            let mut out: Vec<$dtype> = vec![0.; num_total_rows * num_columns];
            let out_slice = array::UnsafeSlice::new(out.as_mut_slice());
            array::fast_concat_2d_axis0(arrays, num_rows, num_columns, $multiplier, out_slice);
            out.into_pyarray_bound(py)
        }
    };
}
fast_concat_2d_axis0_impl!(fast_concat_2d_axis0_f32, f32, 1);
fast_concat_2d_axis0_impl!(fast_concat_2d_axis0_f64, f64, 2);
