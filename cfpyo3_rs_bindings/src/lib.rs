pub mod df;

#[macro_export]
macro_rules! register_submodule {
    ($parent:expr, $hierarchy:expr) => {{
        use pyo3::{prelude::*, py_run};

        let py = $parent.py();
        let module_name = $hierarchy.split('.').last().unwrap();
        let submodule = PyModule::new(py, module_name)?;
        py_run!(
            py,
            submodule,
            concat!("import sys; sys.modules['", $hierarchy, "'] = submodule")
        );
        $parent.add_submodule(&submodule)?;
        submodule
    }};
}
