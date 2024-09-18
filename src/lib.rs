mod df;
mod toolkit;
use numpy::{ndarray::ArrayView2, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, py_run};

macro_rules! register_submodule {
    ($parent:expr, $hierarchy:expr, $module_name:expr) => {{
        let py = $parent.py();
        let submodule = PyModule::new_bound(py, $module_name)?;
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
    let rs_module = register_submodule!(m, "cfpyo3._rs", "_rs");
    let df_module = register_submodule!(rs_module, "cfpyo3._rs.df", "df");
    let toolkit_module = register_submodule!(rs_module, "cfpyo3._rs.toolkit", "toolkit");

    df_module.add("INDEX_CHAR_LEN", df::INDEX_CHAR_LEN)?;

    let frame_module = register_submodule!(df_module, "cfpyo3._rs.df.frame", "frame");
    frame_module.add_class::<df::frame::DataFrameF64>()?;
    #[pyfn(frame_module)]
    pub fn index<'py>(
        py: Python<'py>,
        df: &df::frame::DataFrameF64,
    ) -> Bound<'py, PyArray1<df::IndexDtype>> {
        df.index.to_pyarray_bound(py)
    }
    #[pyfn(frame_module)]
    pub fn columns<'py>(
        py: Python<'py>,
        df: &df::frame::DataFrameF64,
    ) -> Bound<'py, PyArray1<df::ColumnsDtype>> {
        df.columns.to_pyarray_bound(py)
    }
    #[pyfn(frame_module)]
    pub fn values<'py>(py: Python<'py>, df: &df::frame::DataFrameF64) -> Bound<'py, PyArray2<f64>> {
        df.data.to_pyarray_bound(py)
    }

    let misc_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.misc", "misc");
    misc_module.add_function(wrap_pyfunction!(toolkit::misc::hash_code, &misc_module)?)?;

    let array_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.array", "array");
    #[pyfn(array_module)]
    pub fn fast_concat_2d_axis0_f32<'py>(
        py: Python<'py>,
        arrays: Vec<PyReadonlyArray2<f32>>,
    ) -> Bound<'py, PyArray1<f32>> {
        let arrays: Vec<ArrayView2<f32>> = arrays.iter().map(|x| x.as_array()).collect();
        toolkit::array::fast_concat_2d_axis0_f32(py, arrays)
    }
    #[pyfn(array_module)]
    pub fn fast_concat_2d_axis0_f64<'py>(
        py: Python<'py>,
        arrays: Vec<PyReadonlyArray2<f64>>,
    ) -> Bound<'py, PyArray1<f64>> {
        let arrays: Vec<ArrayView2<f64>> = arrays.iter().map(|x| x.as_array()).collect();
        toolkit::array::fast_concat_2d_axis0_f64(py, arrays)
    }

    Ok(())
}
