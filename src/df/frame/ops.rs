use std::{borrow::Borrow, iter::zip};

use itertools::enumerate;
use numpy::{
    ndarray::{ArrayView1, Axis},
    IntoPyArray, PyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;

use crate::toolkit::array::UnsafeSlice;

use super::DataFrameF64;

fn mean(a: ArrayView1<f64>) -> f64 {
    let mut sum = 0.;
    let mut num = 0.;
    for &x in a.iter() {
        if x.is_nan() {
            continue;
        }
        sum += x;
        num += 1.;
    }
    if num == 0. {
        f64::NAN
    } else {
        sum / num
    }
}

fn corr(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let valid_indices: Vec<usize> = zip(a.iter(), b.iter())
        .enumerate()
        .filter_map(|(i, (&x, &y))| {
            if x.is_nan() || y.is_nan() {
                None
            } else {
                Some(i)
            }
        })
        .collect();
    if valid_indices.is_empty() {
        return f64::NAN;
    }
    let a = a.select(Axis(0), &valid_indices);
    let b = b.select(Axis(0), &valid_indices);
    let a_mean = a.mean().unwrap();
    let b_mean = b.mean().unwrap();
    let a = a - a_mean;
    let b = b - b_mean;
    let cov = a.dot(&b);
    let var1 = a.dot(&a);
    let var2 = b.dot(&b);
    cov / (var1.sqrt() * var2.sqrt())
}

#[pymethods]
impl DataFrameF64 {
    fn mean_axis1<'a>(&'a self, py: Python<'a>) -> Bound<'a, PyArray1<f64>> {
        let data = self.get_data_array(py);
        let data = data.borrow();
        let mut res: Vec<f64> = vec![0.; data.nrows()];
        let mut slice = UnsafeSlice::new(res.as_mut_slice());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        py.allow_threads(|| {
            pool.scope(|s| {
                enumerate(data.rows()).for_each(|(i, row)| {
                    s.spawn(move |_| slice.set(i, mean(row)));
                });
            });
        });
        res.into_pyarray_bound(py)
    }

    fn corr_with_axis1<'a>(
        &'a self,
        py: Python<'a>,
        other: PyReadonlyArray2<f64>,
    ) -> Bound<'a, PyArray1<f64>> {
        let data = self.get_data_array(py);
        let data = data.borrow();
        let other = other.as_array();
        let other = other.borrow();
        let mut res: Vec<f64> = vec![0.; data.nrows()];
        let mut slice = UnsafeSlice::new(res.as_mut_slice());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        py.allow_threads(|| {
            pool.scope(move |s| {
                zip(data.rows(), other.rows())
                    .enumerate()
                    .for_each(|(i, (a, b))| {
                        s.spawn(move |_| slice.set(i, corr(a, b)));
                    });
            });
        });
        res.into_pyarray_bound(py)
    }
}
