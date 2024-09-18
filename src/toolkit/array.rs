use std::{cell::UnsafeCell, mem, ptr, thread::available_parallelism};

use itertools::izip;
use numpy::{ndarray::ArrayView2, IntoPyArray, PyArray1};
use pyo3::prelude::*;

#[derive(Copy, Clone)]
pub struct UnsafeSlice<'a, T> {
    slice: &'a [UnsafeCell<T>],
}
unsafe impl<'a, T: Send + Sync> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for UnsafeSlice<'a, T> {}
impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }

    pub fn shadow(&mut self) -> Self {
        Self { slice: self.slice }
    }

    pub fn copy_from_slice(&mut self, i: usize, src: &[T])
    where
        T: Copy,
    {
        let ptr = self.slice[i].get();
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
        }
    }
}

const CONCAT_GROUP_LIMIT: usize = 4 * 239 * 5000;
type Task<'a, 'b, D> = (Vec<usize>, Vec<ArrayView2<'a, D>>, UnsafeSlice<'b, D>);
#[inline]
fn fill_concat<D: Copy>((offsets, arrays, mut out): Task<D>) {
    offsets.iter().enumerate().for_each(|(i, &offset)| {
        out.copy_from_slice(offset, arrays[i].as_slice().unwrap());
    });
}
fn fast_concat_2d_axis0<D: Copy + Send + Sync>(
    arrays: Vec<ArrayView2<D>>,
    num_rows: Vec<usize>,
    num_columns: usize,
    limit_multiplier: usize,
    mut out: UnsafeSlice<D>,
) {
    let mut cumsum: usize = 0;
    let mut offsets: Vec<usize> = vec![0; num_rows.len()];
    for i in 1..num_rows.len() {
        cumsum += num_rows[i - 1];
        offsets[i] = cumsum * num_columns;
    }

    let bumped_limit = CONCAT_GROUP_LIMIT * 16;
    let total_bytes = offsets.last().unwrap() + num_rows.last().unwrap() * num_columns;
    let (mut group_limit, mut tasks_divisor) = if total_bytes <= bumped_limit {
        (CONCAT_GROUP_LIMIT, 8)
    } else {
        (bumped_limit, 1)
    };
    group_limit *= limit_multiplier;

    let prior_num_tasks = total_bytes.div_ceil(group_limit);
    let prior_num_threads = prior_num_tasks / tasks_divisor;
    if prior_num_threads > 1 {
        group_limit = total_bytes.div_ceil(prior_num_threads);
        tasks_divisor = 1;
    }

    let nbytes = mem::size_of::<D>();

    let mut tasks: Vec<Task<D>> = Vec::new();
    let mut current_tasks: Option<Task<D>> = Some((Vec::new(), Vec::new(), out.shadow()));
    let mut nbytes_cumsum = 0;
    izip!(num_rows.iter(), offsets.into_iter(), arrays.into_iter()).for_each(
        |(&num_row, offset, array)| {
            nbytes_cumsum += nbytes * num_row * num_columns;
            if let Some(ref mut current_tasks) = current_tasks {
                current_tasks.0.push(offset);
                current_tasks.1.push(array);
            }
            if nbytes_cumsum >= group_limit {
                nbytes_cumsum = 0;
                if let Some(current_tasks) = current_tasks.take() {
                    tasks.push(current_tasks);
                }
                current_tasks = Some((Vec::new(), Vec::new(), out.shadow()));
            }
        },
    );
    if let Some(current_tasks) = current_tasks.take() {
        if current_tasks.0.len() > 0 {
            tasks.push(current_tasks);
        }
    }

    let max_threads = available_parallelism()
        .expect("failed to get available parallelism")
        .get();
    let num_threads = (tasks.len() / tasks_divisor).min(max_threads * 8).min(512);
    if num_threads <= 1 {
        tasks.into_iter().for_each(fill_concat);
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.scope(move |s| {
            tasks.into_iter().for_each(|task| {
                s.spawn(move |_| fill_concat(task));
            });
        });
    }
}

macro_rules! fast_concat_2d_axis0_impl {
    ($name:ident, $dtype:ty, $multiplier:expr) => {
        pub fn $name<'py>(
            py: Python<'py>,
            arrays: Vec<ArrayView2<$dtype>>,
        ) -> Bound<'py, PyArray1<$dtype>> {
            let num_rows: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
            let num_columns = arrays[0].shape()[1];
            for array in &arrays {
                if array.shape()[1] != num_columns {
                    panic!("all arrays should have same number of columns");
                }
            }
            let num_total_rows: usize = num_rows.iter().sum();
            let mut out: Vec<$dtype> = vec![0.; num_total_rows * num_columns];
            let out_slice = UnsafeSlice::new(out.as_mut_slice());
            fast_concat_2d_axis0(arrays, num_rows, num_columns, $multiplier, out_slice);
            out.into_pyarray_bound(py)
        }
    };
}

fast_concat_2d_axis0_impl!(fast_concat_2d_axis0_f32, f32, 1);
fast_concat_2d_axis0_impl!(fast_concat_2d_axis0_f64, f64, 2);
