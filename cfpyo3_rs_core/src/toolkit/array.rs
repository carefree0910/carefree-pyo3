use anyhow::Result;
use itertools::{enumerate, izip, Itertools};
use num_traits::{Float, FromPrimitive};
use numpy::ndarray::{stack, Array1, Array2, ArrayView1, ArrayView2, Axis, ScalarOperand};
use std::{
    cell::UnsafeCell,
    fmt::{Debug, Display},
    iter::zip,
    mem,
    ops::{AddAssign, MulAssign, SubAssign},
    ptr,
    thread::available_parallelism,
};

#[derive(Debug)]
pub struct ArrayError(String);
impl ArrayError {
    fn new(msg: &str) -> Self {
        Self(msg.to_string())
    }
    pub fn data_not_contiguous<T>() -> Result<T> {
        Err(ArrayError::new("data is not contiguous").into())
    }
}
impl std::error::Error for ArrayError {}
impl std::fmt::Display for ArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error occurred in `array` module: {}", self.0)
    }
}

#[macro_export]
macro_rules! as_data_slice_or_err {
    ($data:expr) => {
        match $data.as_slice() {
            Some(data) => data,
            None => return $crate::toolkit::array::ArrayError::data_not_contiguous(),
        }
    };
}

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

    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            slice: &self.slice[start..end],
        }
    }

    pub fn set(&mut self, i: usize, value: T) {
        let ptr = self.slice[i].get();
        unsafe {
            ptr::write(ptr, value);
        }
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
pub fn fast_concat_2d_axis0<D: Copy + Send + Sync>(
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
        if !current_tasks.0.is_empty() {
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

pub trait AFloat:
    Float
    + AddAssign
    + SubAssign
    + MulAssign
    + FromPrimitive
    + ScalarOperand
    + Send
    + Sync
    + Debug
    + Display
{
}
impl<T> AFloat for T where
    T: Float
        + AddAssign
        + SubAssign
        + MulAssign
        + FromPrimitive
        + ScalarOperand
        + Send
        + Sync
        + Debug
        + Display
{
}

// ops

#[inline]
fn get_valid_indices<T: AFloat>(a: ArrayView1<T>, b: ArrayView1<T>) -> Vec<usize> {
    zip(a.iter(), b.iter())
        .enumerate()
        .filter_map(|(i, (&x, &y))| {
            if x.is_nan() || y.is_nan() {
                None
            } else {
                Some(i)
            }
        })
        .collect()
}
#[inline]
fn convert_valid_indices(valid_mask: ArrayView1<bool>) -> Vec<usize> {
    valid_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &valid)| if valid { Some(i) } else { None })
        .collect()
}

#[inline]
fn sorted_quantile<T: AFloat>(a: &[&T], q: T) -> T {
    let n = a.len();
    if n == 0 {
        return T::nan();
    }
    let q = q * T::from_f64(n as f64).unwrap();
    let i = q.floor().to_usize().unwrap();
    if i == n - 1 {
        return *a[n - 1];
    }
    let q = q - T::from_usize(i).unwrap();
    *a[i] * (T::one() - q) + *a[i + 1] * q
}
#[inline]
fn sorted_median<T: AFloat>(a: &[&T]) -> T {
    sorted_quantile(a, T::from_f64(0.5).unwrap())
}

#[inline]
fn solve_2d<T: AFloat>(x: ArrayView2<T>, y: ArrayView1<T>) -> (T, T) {
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y);
    let xtx = xtx.into_raw_vec();
    let (a, b, c, d) = (xtx[0], xtx[1], xtx[2], xtx[3]);
    let xtx_inv = Array2::from_shape_vec((2, 2), vec![d, -b, -c, a]).unwrap();
    let xtx_inv = xtx_inv / (a * d - b * c);
    let solution = xtx_inv.dot(&xty);
    (solution[0], solution[1])
}

fn mean<T: AFloat>(a: ArrayView1<T>) -> T {
    let mut sum = T::zero();
    let mut num = T::zero();
    for &x in a.iter() {
        if x.is_nan() {
            continue;
        }
        sum += x;
        num += T::one();
    }
    if num.is_zero() {
        T::nan()
    } else {
        sum / num
    }
}
fn masked_mean<T: AFloat>(a: ArrayView1<T>, valid_mask: ArrayView1<bool>) -> T {
    let mut sum = T::zero();
    let mut num = T::zero();
    for (&x, &valid) in zip(a.iter(), valid_mask.iter()) {
        if !valid {
            continue;
        }
        sum += x;
        num += T::one();
    }
    if num.is_zero() {
        T::nan()
    } else {
        sum / num
    }
}

#[inline]
fn corr_with<T: AFloat>(a: ArrayView1<T>, b: ArrayView1<T>, valid_indices: Vec<usize>) -> T {
    if valid_indices.is_empty() {
        return T::nan();
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
fn corr<T: AFloat>(a: ArrayView1<T>, b: ArrayView1<T>) -> T {
    corr_with(a, b, get_valid_indices(a, b))
}
fn masked_corr<T: AFloat>(a: ArrayView1<T>, b: ArrayView1<T>, valid_mask: ArrayView1<bool>) -> T {
    corr_with(a, b, convert_valid_indices(valid_mask))
}

#[inline]
fn coeff_with<T: AFloat>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    valid_indices: Vec<usize>,
    q: Option<T>,
) -> (T, T) {
    if valid_indices.is_empty() {
        return (T::nan(), T::nan());
    }
    let x = x.select(Axis(0), &valid_indices);
    let mut y = y.select(Axis(0), &valid_indices);
    let x_sorted = x
        .iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .collect_vec();
    let x_med = sorted_median(&x_sorted);
    let x_mad = x_sorted.iter().map(|&x| (*x - x_med).abs()).collect_vec();
    let x_mad = sorted_median(&x_mad.iter().collect_vec());
    let hundred = T::from_f64(100.0).unwrap();
    let x_floor = x_med - hundred * x_mad;
    let x_ceil = x_med + hundred * x_mad;
    let x = Array1::from_iter(x.iter().map(|&x| x.max(x_floor).min(x_ceil)));
    let x_mean = x.mean().unwrap();
    let x_std = x.std(T::zero());
    let mut x = (x - x_mean) / x_std;
    if let Some(q) = q {
        if q > T::zero() {
            let q_floor = sorted_quantile(&x_sorted, q);
            let q_ceil = sorted_quantile(&x_sorted, T::one() - q);
            let picked_indices: Vec<usize> = x
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| {
                    if x <= q_floor && x >= q_ceil {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            x = x.select(Axis(0), &picked_indices);
            y = y.select(Axis(0), &picked_indices);
        }
    }
    let x = stack![Axis(1), x, Array1::ones(x.len())];
    solve_2d(x.view(), y.view())
}
fn coeff<T: AFloat>(x: ArrayView1<T>, y: ArrayView1<T>, q: Option<T>) -> (T, T) {
    coeff_with(x, y, get_valid_indices(x, y), q)
}
fn masked_coeff<T: AFloat>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    valid_mask: ArrayView1<bool>,
    q: Option<T>,
) -> (T, T) {
    coeff_with(x, y, convert_valid_indices(valid_mask), q)
}

// axis1 wrappers

pub fn mean_axis1<T: AFloat>(a: &ArrayView2<T>, num_threads: usize) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(res.as_mut_slice());
    if num_threads <= 1 {
        enumerate(a.rows()).for_each(|(i, row)| {
            slice.set(i, mean(row));
        });
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.scope(|s| {
            enumerate(a.rows()).for_each(|(i, row)| {
                s.spawn(move |_| slice.set(i, mean(row)));
            });
        });
    }
    res
}
pub fn masked_mean_axis1<T: AFloat>(
    a: &ArrayView2<T>,
    valid_mask: &ArrayView2<bool>,
    num_threads: usize,
) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(res.as_mut_slice());
    if num_threads <= 1 {
        enumerate(zip(a.rows(), valid_mask.rows())).for_each(|(i, (row, valid_mask))| {
            slice.set(i, masked_mean(row, valid_mask));
        });
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.scope(|s| {
            enumerate(zip(a.rows(), valid_mask.rows())).for_each(|(i, (row, valid_mask))| {
                s.spawn(move |_| slice.set(i, masked_mean(row, valid_mask)));
            });
        });
    }
    res
}

pub fn corr_axis1<T: AFloat>(a: &ArrayView2<T>, b: &ArrayView2<T>, num_threads: usize) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(res.as_mut_slice());
    if num_threads <= 1 {
        zip(a.rows(), b.rows()).enumerate().for_each(|(i, (a, b))| {
            slice.set(i, corr(a, b));
        });
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.scope(move |s| {
            zip(a.rows(), b.rows()).enumerate().for_each(|(i, (a, b))| {
                s.spawn(move |_| slice.set(i, corr(a, b)));
            });
        });
    }
    res
}
pub fn masked_corr_axis1<T: AFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    valid_mask: &ArrayView2<bool>,
    num_threads: usize,
) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(res.as_mut_slice());
    if num_threads <= 1 {
        izip!(a.rows(), b.rows(), valid_mask.rows())
            .enumerate()
            .for_each(|(i, (a, b, valid_mask))| {
                slice.set(i, masked_corr(a, b, valid_mask));
            });
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.scope(move |s| {
            izip!(a.rows(), b.rows(), valid_mask.rows())
                .enumerate()
                .for_each(|(i, (a, b, valid_mask))| {
                    s.spawn(move |_| slice.set(i, masked_corr(a, b, valid_mask)));
                });
        });
    }
    res
}

pub fn coeff_axis1<T: AFloat>(
    x: &ArrayView2<T>,
    y: &ArrayView2<T>,
    q: Option<T>,
    num_threads: usize,
) -> (Vec<T>, Vec<T>) {
    let mut ws: Vec<T> = vec![T::zero(); x.nrows()];
    let mut bs: Vec<T> = vec![T::zero(); x.nrows()];
    let mut slice0 = UnsafeSlice::new(ws.as_mut_slice());
    let mut slice1 = UnsafeSlice::new(bs.as_mut_slice());
    if num_threads <= 1 {
        izip!(x.rows(), y.rows())
            .enumerate()
            .for_each(|(i, (x, y))| {
                let (w, b) = coeff(x, y, q);
                slice0.set(i, w);
                slice1.set(i, b);
            });
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.scope(move |s| {
            izip!(x.rows(), y.rows())
                .enumerate()
                .for_each(|(i, (x, y))| {
                    s.spawn(move |_| {
                        let (w, b) = coeff(x, y, q);
                        slice0.set(i, w);
                        slice1.set(i, b);
                    });
                });
        });
    }
    (ws, bs)
}
pub fn masked_coeff_axis1<T: AFloat>(
    x: &ArrayView2<T>,
    y: &ArrayView2<T>,
    valid_mask: &ArrayView2<bool>,
    q: Option<T>,
    num_threads: usize,
) -> (Vec<T>, Vec<T>) {
    let mut ws: Vec<T> = vec![T::zero(); x.nrows()];
    let mut bs: Vec<T> = vec![T::zero(); x.nrows()];
    let mut slice0 = UnsafeSlice::new(ws.as_mut_slice());
    let mut slice1 = UnsafeSlice::new(bs.as_mut_slice());
    if num_threads <= 1 {
        izip!(x.rows(), y.rows(), valid_mask.rows())
            .enumerate()
            .for_each(|(i, (x, y, valid_mask))| {
                let (w, b) = masked_coeff(x, y, valid_mask, q);
                slice0.set(i, w);
                slice1.set(i, b);
            });
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.scope(move |s| {
            izip!(x.rows(), y.rows(), valid_mask.rows())
                .enumerate()
                .for_each(|(i, (x, y, valid_mask))| {
                    s.spawn(move |_| {
                        let (w, b) = masked_coeff(x, y, valid_mask, q);
                        slice0.set(i, w);
                        slice1.set(i, b);
                    });
                });
        });
    }
    (ws, bs)
}

// misc

pub fn searchsorted<T: Ord>(arr: &ArrayView1<T>, value: &T) -> usize {
    arr.as_slice()
        .unwrap()
        .binary_search(value)
        .unwrap_or_else(|x| x)
}

pub fn batch_searchsorted<T: Ord>(arr: &ArrayView1<T>, values: &ArrayView1<T>) -> Vec<usize> {
    values
        .iter()
        .map(|value| searchsorted(arr, value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_allclose<T: AFloat>(a: &[T], b: &[T]) {
        let atol = T::from_f64(1e-6).unwrap();
        let rtol = T::from_f64(1e-6).unwrap();
        a.iter().zip(b.iter()).for_each(|(&x, &y)| {
            assert!(
                (x - y).abs() <= atol + rtol * y.abs(),
                "not close - a: {:?}, b: {:?}",
                a,
                b,
            );
        });
    }

    macro_rules! test_fast_concat_2d_axis0 {
        ($dtype:ty) => {
            let array_2d_u = ArrayView2::<$dtype>::from_shape((1, 3), &[1., 2., 3.]).unwrap();
            let array_2d_l =
                ArrayView2::<$dtype>::from_shape((2, 3), &[4., 5., 6., 7., 8., 9.]).unwrap();
            let arrays = vec![array_2d_u, array_2d_l];
            let mut out: Vec<$dtype> = vec![0.; 3 * 3];
            let out_slice = UnsafeSlice::new(out.as_mut_slice());
            fast_concat_2d_axis0(arrays, vec![1, 2], 3, 1, out_slice);
            assert_eq!(out.as_slice(), &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        };
    }

    macro_rules! test_mean_axis1 {
        ($dtype:ty) => {
            let array =
                ArrayView2::<$dtype>::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
            let out = mean_axis1(&array, 1);
            assert_allclose(out.as_slice(), &[2., 5.]);
            let out = mean_axis1(&array, 2);
            assert_allclose(out.as_slice(), &[2., 5.]);
        };
    }

    macro_rules! test_corr_axis1 {
        ($dtype:ty) => {
            let array =
                ArrayView2::<$dtype>::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
            let out = corr_axis1(&array, &(&array + 1.).view(), 1);
            assert_allclose(out.as_slice(), &[1., 1.]);
            let out = corr_axis1(&array, &(&array + 1.).view(), 2);
            assert_allclose(out.as_slice(), &[1., 1.]);
        };
    }

    #[test]
    fn test_fast_concat_2d_axis0_f32() {
        test_fast_concat_2d_axis0!(f32);
    }
    #[test]
    fn test_fast_concat_2d_axis0_f64() {
        test_fast_concat_2d_axis0!(f64);
    }

    #[test]
    fn test_mean_axis1_f32() {
        test_mean_axis1!(f32);
    }
    #[test]
    fn test_mean_axis1_f64() {
        test_mean_axis1!(f64);
    }

    #[test]
    fn test_corr_axis1_f32() {
        test_corr_axis1!(f32);
    }
    #[test]
    fn test_corr_axis1_f64() {
        test_corr_axis1!(f64);
    }

    #[test]
    fn test_coeff_axis1() {
        let x = ArrayView2::<f64>::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
        let y = ArrayView2::<f64>::from_shape((2, 3), &[2., 4., 6., 8., 10., 12.]).unwrap();
        let scale = 2. * (2. / 3.).sqrt();
        let (ws, bs) = coeff_axis1(&x, &y, None, 1);
        assert_allclose(ws.as_slice(), &[scale, scale]);
        assert_allclose(bs.as_slice(), &[4., 10.]);
        let (ws, bs) = coeff_axis1(&x, &y, None, 2);
        assert_allclose(ws.as_slice(), &[scale, scale]);
        assert_allclose(bs.as_slice(), &[4., 10.]);
    }

    #[test]
    fn test_searchsorted() {
        let array = ArrayView1::<i64>::from_shape(5, &[1, 2, 3, 5, 6]).unwrap();
        assert_eq!(searchsorted(&array, &0), 0);
        assert_eq!(searchsorted(&array, &1), 0);
        assert_eq!(searchsorted(&array, &3), 2);
        assert_eq!(searchsorted(&array, &4), 3);
        assert_eq!(searchsorted(&array, &5), 3);
        assert_eq!(searchsorted(&array, &6), 4);
        assert_eq!(searchsorted(&array, &7), 5);
        assert_eq!(batch_searchsorted(&array, &array), vec![0, 1, 2, 3, 4]);
    }
}
