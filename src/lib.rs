mod df;
mod toolkit;
use numpy::{PyArray1, PyArray2, ToPyArray};
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

    df_module.add_class::<df::DataFrameF64>()?;
    df_module.add("INDEX_CHAR_LEN", df::INDEX_CHAR_LEN)?;

    let frame_module = register_submodule!(df_module, "cfpyo3._rs.df.frame", "frame");
    frame_module.add_function(wrap_pyfunction!(df::frame::meta::new, &frame_module)?)?;
    frame_module.add_function(wrap_pyfunction!(df::frame::meta::shape, &frame_module)?)?;
    frame_module.add_function(wrap_pyfunction!(df::frame::indexing::rows, &frame_module)?)?;
    #[pyfn(frame_module)]
    pub fn index<'py>(
        py: Python<'py>,
        df: &df::DataFrameF64,
    ) -> Bound<'py, PyArray1<df::IndexDtype>> {
        df.index.to_pyarray_bound(py)
    }
    #[pyfn(frame_module)]
    pub fn columns<'py>(
        py: Python<'py>,
        df: &df::DataFrameF64,
    ) -> Bound<'py, PyArray1<df::ColumnsDtype>> {
        df.columns.to_pyarray_bound(py)
    }
    #[pyfn(frame_module)]
    pub fn values<'py>(py: Python<'py>, df: &df::DataFrameF64) -> Bound<'py, PyArray2<f64>> {
        df.data.to_pyarray_bound(py)
    }

    let misc_module = register_submodule!(toolkit_module, "cfpyo3._rs.toolkit.misc", "misc");
    misc_module.add_function(wrap_pyfunction!(toolkit::misc::hash_code, &misc_module)?)?;

    Ok(())
}
