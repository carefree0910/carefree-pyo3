use md5::{Digest, Md5};

use pyo3::prelude::*;

#[pyfunction]
pub fn hash_code(code: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(code.as_bytes());
    format!("{:x}", hasher.finalize())
}
