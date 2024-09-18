mod df;
mod toolkit;
use numpy::{ndarray::ArrayView2, PyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, py_run};

macro_rules! register_submodule {
    ($parent:expr, $hierarchy:expr) => {{
        let py = $parent.py();
        let module_name = $hierarchy.split('.').last().unwrap();
        let submodule = PyModule::new_bound(py, module_name)?;
        py_run!(
            py,
            submodule,
            concat!("import sys; sys.modules['", $hierarchy, "'] = submodule")
        );
        $parent.add_submodule(&submodule)?;
        submodule
    }};
}

#[pymodule]
fn cfpyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let rs_module = register_submodule!(m, "cfpyo3._rs");
    let df_module = register_submodule!(rs_module, "cfpyo3._rs.df");
    let toolkit_module = register_submodule!(rs_module, "cfpyo3._rs.toolkit");

    df_module.add("INDEX_CHAR_LEN", df::INDEX_CHAR_LEN)?;

    let frame_module = register_submodule!(df_module, "cfpyo3._rs.df.frame");
    frame_module.add_class::<df::frame::DataFrameF64>()?;

    let misc_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.misc");
    misc_module.add_function(wrap_pyfunction!(toolkit::misc::hash_code, &misc_module)?)?;

    let array_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.array");
    macro_rules! fast_concat_2d_axis0_impl {
        ($func:ident, $dtype:ty) => {
            #[pyfunction]
            pub fn $func<'py>(
                py: Python<'py>,
                arrays: Vec<PyReadonlyArray2<$dtype>>,
            ) -> Bound<'py, PyArray1<$dtype>> {
                let arrays: Vec<ArrayView2<$dtype>> = arrays.iter().map(|x| x.as_array()).collect();
                toolkit::array::$func(py, arrays)
            }
        };
    }
    fast_concat_2d_axis0_impl!(fast_concat_2d_axis0_f32, f32);
    fast_concat_2d_axis0_impl!(fast_concat_2d_axis0_f64, f64);
    array_module.add_function(wrap_pyfunction!(fast_concat_2d_axis0_f32, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(fast_concat_2d_axis0_f64, &array_module)?)?;

    Ok(())
}
