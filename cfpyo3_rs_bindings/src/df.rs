//! A DataFrame binding from [`cfpyo3_core::df::DataFrame`] to Python using [`pyo3`].
//!
//! # Design
//!
//! This module defines two classes: [`DataFrameF64`] and [`OwnedDataFrameF64`] for different purposes.
//!
//! ## [`DataFrameF64`]
//!
//! This class is supposed to 'interact' data with Python directly, so its fields are wrapped
//! with the [`Py`] smart pointers. Therefore it achieves zero-copy data exchange between Rust and Python.
//!
//! It however will suffer a little bit when performing calculations in Rust, because it requires GIL
//! to get the data from Python.
//!
//! ## [`OwnedDataFrameF64`]
//!
//! This is a pure Rust class that uses [`Array`]s to store the data. It is designed for almost only
//! one single purpose: to load data from an external source (e.g., fs, internet, etc.) and then perform
//! calculations directly in Rust. There should be no data interactions with Python in this workflow.
//!
//! > Supported operations of [`OwnedDataFrameF64`] are limited to the [`io::IOs`] + [`ops::Ops`] trait.
//!
//! With this design, it is useful in some performance-critical scenarios. To list a few:
//! - Read from a lot of files, do calculations, and write the results back to some other files.
//! - Read from a lot of requests, process them, and report the (aggregated) results to Python.
//!
//! If we need to acquire GIL everytime a single read is performed, it will be basically single threaded and
//! therefore impossible to utilize the modern hardware to its full potential.

use cfpyo3_core::df::{ColumnsDtype, IndexDtype};
use numpy::{
    ndarray::{Array1, Array2},
    PyArray1, PyArray2,
};
use pyo3::prelude::*;

pub mod io;
pub mod meta;
pub mod ops;

#[pyclass(frozen)]
pub struct DataFrameF64 {
    pub index: Py<PyArray1<IndexDtype>>,
    pub columns: Py<PyArray1<ColumnsDtype>>,
    pub values: Py<PyArray2<f64>>,
}

#[pyclass(frozen)]
pub struct OwnedDataFrameF64 {
    pub index: Array1<IndexDtype>,
    pub columns: Array1<ColumnsDtype>,
    pub values: Array2<f64>,
}
