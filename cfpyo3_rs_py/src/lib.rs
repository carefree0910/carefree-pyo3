use cfpyo3_bindings::register_submodule;
use numpy::{ndarray::ArrayView2, IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod toolkit;

#[pymodule]
fn cfpyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let rs_module = register_submodule!(m, "cfpyo3._rs");
    let df_module = register_submodule!(rs_module, "cfpyo3._rs.df");
    let toolkit_module = register_submodule!(rs_module, "cfpyo3._rs.toolkit");

    df_module.add("COLUMNS_NBYTES", cfpyo3_core::df::COLUMNS_NBYTES)?;
    df_module.add_class::<cfpyo3_bindings::df::DataFrameF64>()?;

    let misc_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.misc");
    misc_module.add_function(wrap_pyfunction!(toolkit::misc::hash_code, &misc_module)?)?;

    let array_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.array");

    macro_rules! unary_impl {
        ($name:ident, $type_str:ident, $dtype:ty, $func:expr) => {
            paste::item! {
                #[pyfunction]
                pub fn [< $name _ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    $func(&a, num_threads).into_pyarray(py)
                }
            }
        };
        ($name:ident, $type_str:ident, $dtype:ty, $func:expr, $default_num_threads:expr) => {
            paste::item! {
                #[pyfunction]
                pub fn [< $name _ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let num_threads = num_threads.unwrap_or($default_num_threads);
                    $func(&a, num_threads).into_pyarray(py)
                }
            }
        };
    }
    macro_rules! binary_impl {
        ($name:ident, $type_str:ident, $dtype:ty, $func:expr) => {
            paste::item! {
                #[pyfunction]
                pub fn [< $name _ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    b: PyReadonlyArray2<$dtype>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let b = b.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    $func(&a, &b, num_threads).into_pyarray(py)
                }
            }
        };
    }
    macro_rules! masked_unary_impl {
        ($name:ident, $type_str:ident, $dtype:ty, $func:expr) => {
            paste::item! {
                #[pyfunction]
                pub fn [< $name _ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    valid_mask: PyReadonlyArray2<bool>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let valid_mask = valid_mask.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    $func(&a, &valid_mask, num_threads).into_pyarray(py)
                }
            }
        };
    }
    macro_rules! masked_binary_impl {
        ($name:ident, $type_str:ident, $dtype:ty, $func:expr) => {
            paste::item! {
                #[pyfunction]
                pub fn [< $name _ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    b: PyReadonlyArray2<$dtype>,
                    valid_mask: PyReadonlyArray2<bool>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let b = b.as_array();
                    let valid_mask = valid_mask.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    $func(&a, &b, &valid_mask, num_threads).into_pyarray(py)
                }
            }
        };
    }

    macro_rules! array_ops_impl {
        ($type_str:ident, $dtype:ty) => {
            unary_impl!(sum_axis1, $type_str, $dtype, cfpyo3_core::toolkit::array::sum_axis1, 0);
            unary_impl!(mean_axis1, $type_str, $dtype, cfpyo3_core::toolkit::array::mean_axis1, 0);
            unary_impl!(nanmean_axis1, $type_str, $dtype, cfpyo3_core::toolkit::array::nanmean_axis1);
            masked_unary_impl!(masked_mean_axis1, $type_str, $dtype, cfpyo3_core::toolkit::array::masked_mean_axis1);
            binary_impl!(corr_axis1, $type_str, $dtype, cfpyo3_core::toolkit::array::corr_axis1);
            binary_impl!(nancorr_axis1, $type_str, $dtype, cfpyo3_core::toolkit::array::nancorr_axis1);
            masked_binary_impl!(masked_corr_axis1, $type_str, $dtype, cfpyo3_core::toolkit::array::masked_corr_axis1);
            paste::item! {
                #[pyfunction]
                pub fn [< coeff_axis1_ $type_str >]<'py>(
                    py: Python<'py>,
                    x: PyReadonlyArray2<$dtype>,
                    y: PyReadonlyArray2<$dtype>,
                    q: Option<$dtype>,
                    num_threads: Option<usize>,
                ) -> (Bound<'py, PyArray1<$dtype>>, Bound<'py, PyArray1<$dtype>>) {
                    let x = x.as_array();
                    let y = y.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    let (ws, bs) = cfpyo3_core::toolkit::array::coeff_axis1(&x, &y, q, num_threads);
                    (ws.into_pyarray(py), bs.into_pyarray(py))
                }
            }
            paste::item! {
                #[pyfunction]
                pub fn [< masked_coeff_axis1_ $type_str >]<'py>(
                    py: Python<'py>,
                    x: PyReadonlyArray2<$dtype>,
                    y: PyReadonlyArray2<$dtype>,
                    valid_mask: PyReadonlyArray2<bool>,
                    q: Option<$dtype>,
                    num_threads: Option<usize>,
                ) -> (Bound<'py, PyArray1<$dtype>>, Bound<'py, PyArray1<$dtype>>) {
                    let x = x.as_array();
                    let y = y.as_array();
                    let valid_mask = valid_mask.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    let (ws, bs) = cfpyo3_core::toolkit::array::masked_coeff_axis1(&x, &y, &valid_mask, q, num_threads);
                    (ws.into_pyarray(py), bs.into_pyarray(py))
                }
            }
            paste::item! {
                #[pyfunction]
                pub fn [< fast_concat_2d_axis0_ $type_str >]<'py>(
                    py: Python<'py>,
                    arrays: Vec<PyReadonlyArray2<$dtype>>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let arrays: Vec<ArrayView2<$dtype>> = arrays.iter().map(|x| x.as_array()).collect();
                    toolkit::array::[< fast_concat_2d_axis0_ $type_str >](py, arrays)
                }
            }
        };
    }
    array_ops_impl!(f32, f32);
    array_ops_impl!(f64, f64);

    macro_rules! register_functions {
        ($($func:ident),*) => {
            $(
                paste::item! {
                    array_module.add_function(wrap_pyfunction!([< $func _f32 >], &array_module)?)?;
                    array_module.add_function(wrap_pyfunction!([< $func _f64 >], &array_module)?)?;
                }
            )*
        };
    }
    register_functions!(
        sum_axis1,
        mean_axis1,
        nanmean_axis1,
        masked_mean_axis1,
        corr_axis1,
        nancorr_axis1,
        masked_corr_axis1,
        coeff_axis1,
        masked_coeff_axis1,
        fast_concat_2d_axis0
    );

    Ok(())
}
