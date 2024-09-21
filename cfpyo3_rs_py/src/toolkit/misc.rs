use cfpyo3_core::toolkit::misc;
use pyo3::prelude::*;

#[pyfunction]
pub fn hash_code(code: &str) -> String {
    misc::hash_code(code)
}
