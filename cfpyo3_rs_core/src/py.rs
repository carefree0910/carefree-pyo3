//! Some useful bindings for pyo3.
//!
//! # Design
//!
//! The initial design of this repo is to completely separate the Python bindings
//! from the Rust core. But as the project grows, we found that some bindings are
//! relatively common and have broad use cases.
//!
//! Therefore, we decide to extract some 'core' bindings to this module. With them,
//! people will find it much easier when they want to create their own Python bindings
//! on top of the Rust core (i.e., `cfpyo3_core`).

pub mod df;
