pub mod df;

#[macro_export]
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
