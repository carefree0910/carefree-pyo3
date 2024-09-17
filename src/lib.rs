mod df;
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

    df_module.add_class::<df::DataFrameF64>()?;
    df_module.add("INDEX_CHAR_LEN", df::INDEX_CHAR_LEN)?;
    df_module.add_function(wrap_pyfunction!(df::meta::new, &df_module)?)?;
    df_module.add_function(wrap_pyfunction!(df::meta::shape, &df_module)?)?;
    df_module.add_function(wrap_pyfunction!(df::indexing::rows, &df_module)?)?;
    #[pyfn(df_module)]
    pub fn index<'py>(
        py: Python<'py>,
        df: &df::DataFrameF64,
    ) -> Bound<'py, PyArray1<df::IndexDtype>> {
        df.index.to_pyarray_bound(py)
    }
    #[pyfn(df_module)]
    pub fn columns<'py>(
        py: Python<'py>,
        df: &df::DataFrameF64,
    ) -> Bound<'py, PyArray1<df::ColumnsDtype>> {
        df.columns.to_pyarray_bound(py)
    }
    #[pyfn(df_module)]
    pub fn values<'py>(py: Python<'py>, df: &df::DataFrameF64) -> Bound<'py, PyArray2<f64>> {
        df.data.to_pyarray_bound(py)
    }

    Ok(())
}
