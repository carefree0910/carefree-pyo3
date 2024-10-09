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
    df_module.add_class::<cfpyo3_bindings::df::OwnedDataFrameF64>()?;

    let misc_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.misc");
    misc_module.add_function(wrap_pyfunction!(toolkit::misc::hash_code, &misc_module)?)?;

    let array_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.array");
    macro_rules! array_ops_impl {
        ($type_str:ident, $dtype:ty) => {
            paste::item! {
                #[pyfunction]
                pub fn [< mean_axis1_ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    cfpyo3_core::toolkit::array::mean_axis1(&a, num_threads).into_pyarray_bound(py)
                }
            }
            paste::item! {
                #[pyfunction]
                pub fn [< masked_mean_axis1_ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    valid_mask: PyReadonlyArray2<bool>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let valid_mask = valid_mask.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    cfpyo3_core::toolkit::array::masked_mean_axis1(&a, &valid_mask, num_threads).into_pyarray_bound(py)
                }
            }
            paste::item! {
                #[pyfunction]
                pub fn [< corr_axis1_ $type_str >]<'py>(
                    py: Python<'py>,
                    a: PyReadonlyArray2<$dtype>,
                    b: PyReadonlyArray2<$dtype>,
                    num_threads: Option<usize>,
                ) -> Bound<'py, PyArray1<$dtype>> {
                    let a = a.as_array();
                    let b = b.as_array();
                    let num_threads = num_threads.unwrap_or(8);
                    cfpyo3_core::toolkit::array::corr_axis1(&a, &b, num_threads).into_pyarray_bound(py)
                }
            }
            paste::item! {
                #[pyfunction]
                pub fn [< masked_corr_axis1_ $type_str >]<'py>(
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
                    cfpyo3_core::toolkit::array::masked_corr_axis1(&a, &b, &valid_mask, num_threads).into_pyarray_bound(py)
                }
            }
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
                    (ws.into_pyarray_bound(py), bs.into_pyarray_bound(py))
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
                    (ws.into_pyarray_bound(py), bs.into_pyarray_bound(py))
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
    array_module.add_function(wrap_pyfunction!(mean_axis1_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(mean_axis1_f64, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(masked_mean_axis1_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(masked_mean_axis1_f64, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(corr_axis1_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(corr_axis1_f64, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(masked_corr_axis1_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(masked_corr_axis1_f64, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(coeff_axis1_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(coeff_axis1_f64, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(masked_coeff_axis1_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(masked_coeff_axis1_f64, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(fast_concat_2d_axis0_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(fast_concat_2d_axis0_f64, &array_module)?)?;

    Ok(())
}
