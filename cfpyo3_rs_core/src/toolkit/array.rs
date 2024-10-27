use anyhow::{Ok, Result};
use core::{mem, ptr, slice};
use itertools::{izip, Itertools};
use memmap2::{Mmap, MmapOptions};
use num_traits::{Float, FromPrimitive};
use numpy::{
    ndarray::{stack, Array1, Array2, ArrayView1, ArrayView2, Axis, ScalarOperand},
    Element,
};
use std::{
    cell::UnsafeCell,
    cmp::Ordering,
    collections::HashMap,
    fmt::{Debug, Display},
    fs::File,
    iter::zip,
    marker::PhantomData,
    ops::{AddAssign, MulAssign, SubAssign},
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

pub struct MmapArray1<T: Element>(Mmap, usize, PhantomData<T>);
impl<T: Element> MmapArray1<T> {
    /// # Safety
    ///
    /// The use of `mmap` is unsafe, see the documentation of [`MmapOptions`] for more details.
    pub unsafe fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let len = mmap.len() / mem::size_of::<T>();
        Ok(Self(mmap, len, PhantomData))
    }

    pub fn len(&self) -> usize {
        self.1
    }
    pub fn is_empty(&self) -> bool {
        self.1 == 0
    }

    /// # Safety
    ///
    /// The use of [`slice::from_raw_parts`] is unsafe, see its documentation for more details.
    pub unsafe fn as_slice(&self) -> &[T] {
        slice::from_raw_parts(self.0.as_ptr() as *const T, self.1)
    }

    /// # Safety
    ///
    /// The use of [`ArrayView1::from_shape_ptr`] is unsafe, see its documentation for more details.
    pub unsafe fn as_array_view(&self) -> ArrayView1<T> {
        ArrayView1::from_shape_ptr((self.1,), self.0.as_ptr() as *const T)
    }
}

// float ops

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

// simd ops

const LANES: usize = 16;

macro_rules! simd_unary_reduce {
    ($a:expr, $a_dtype:ty, $func:expr) => {{
        let chunks = $a.chunks_exact(LANES);
        let remainder = chunks.remainder();

        let sum = chunks.fold([T::zero(); LANES], |mut acc, chunk| {
            let chunk: [$a_dtype; LANES] = chunk.try_into().unwrap();
            (0..LANES).for_each(|i| acc[i] += $func(chunk[i]));
            acc
        });

        let mut reduced = T::zero();
        sum.iter().for_each(|&x| reduced += x);
        remainder.iter().for_each(|&x| reduced += $func(x));
        reduced
    }};
    ($a:expr, $func:expr) => {{
        simd_unary_reduce!($a, T, $func)
    }};
}
macro_rules! simd_binary_reduce {
    ($a:expr, $b:expr, $b_dtype:ty, $func:expr) => {{
        let a_chunks = $a.chunks_exact(LANES);
        let b_chunks = $b.chunks_exact(LANES);
        let remainder_a = a_chunks.remainder();
        let remainder_b = b_chunks.remainder();
        let zip_chunks = zip(a_chunks, b_chunks);

        let sum = zip_chunks.fold([T::zero(); LANES], |mut acc, (a_chunk, b_chunk)| {
            let a_chunk: [T; LANES] = a_chunk.try_into().unwrap();
            let b_chunk: [$b_dtype; LANES] = b_chunk.try_into().unwrap();
            (0..LANES).for_each(|i| acc[i] += $func(a_chunk[i], b_chunk[i]));
            acc
        });

        let mut reduced = T::zero();
        sum.iter().for_each(|&x| reduced += x);
        zip(remainder_a, remainder_b).for_each(|(&x, &y)| reduced += $func(x, y));

        reduced
    }};
    ($a:expr, $b:expr, $func:expr) => {{
        simd_binary_reduce!($a, $b, T, $func)
    }};
}
macro_rules! simd_ternary_reduce {
    ($a:expr, $b:expr, $c:expr, $c_dtype:ty, $func:expr) => {{
        let a_chunks = $a.chunks_exact(LANES);
        let b_chunks = $b.chunks_exact(LANES);
        let c_chunks = $c.chunks_exact(LANES);
        let remainder_a = a_chunks.remainder();
        let remainder_b = b_chunks.remainder();
        let remainder_c = c_chunks.remainder();
        let zip_chunks = izip!(a_chunks, b_chunks, c_chunks);

        let sum = zip_chunks.fold(
            [T::zero(); LANES],
            |mut acc, (a_chunk, b_chunk, c_chunk)| {
                let a_chunk: [T; LANES] = a_chunk.try_into().unwrap();
                let b_chunk: [T; LANES] = b_chunk.try_into().unwrap();
                let c_chunk: [$c_dtype; LANES] = c_chunk.try_into().unwrap();
                (0..LANES).for_each(|i| acc[i] += $func(a_chunk[i], b_chunk[i], c_chunk[i]));
                acc
            },
        );

        let mut reduced = T::zero();
        sum.iter().for_each(|&x| reduced += x);
        izip!(remainder_a, remainder_b, remainder_c).for_each(|(&x, &y, &z)| {
            reduced += $func(x, y, z);
        });

        reduced
    }};
    ($a:expr, $b:expr, $c:expr, $func:expr) => {{
        simd_ternary_reduce!($a, $b, $c, T, $func)
    }};
}

pub fn simd_sum<T: AFloat>(a: &[T]) -> T {
    simd_unary_reduce!(a, |x| x)
}
pub fn simd_mean<T: AFloat>(a: &[T]) -> T {
    simd_sum(a) / T::from_usize(a.len()).unwrap()
}
pub fn simd_nanmean<T: AFloat>(a: &[T]) -> T {
    let sum = simd_unary_reduce!(a, |x: T| if x.is_nan() { T::zero() } else { x });
    let num = simd_unary_reduce!(a, |x: T| if x.is_nan() { T::zero() } else { T::one() });
    sum / num
}
pub fn simd_masked_mean<T: AFloat>(a: &[T], valid_mask: &[bool]) -> T {
    let sum = simd_binary_reduce!(a, valid_mask, bool, |x, y| if y { x } else { T::zero() });
    let num = simd_unary_reduce!(valid_mask, bool, |x| if x { T::one() } else { T::zero() });
    sum / num
}
pub fn simd_subtract<T: AFloat>(a: &[T], n: T) -> Vec<T> {
    a.iter().map(|&x| x - n).collect()
}
pub fn simd_dot<T: AFloat>(a: &[T], b: &[T]) -> T {
    simd_binary_reduce!(a, b, |x, y| x * y)
}
pub fn simd_inner<T: AFloat>(a: &[T]) -> T {
    simd_unary_reduce!(a, |x| x * x)
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
pub fn to_valid_indices(valid_mask: ArrayView1<bool>) -> Vec<usize> {
    valid_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &valid)| if valid { Some(i) } else { None })
        .collect()
}

#[inline]
/// this function will put `NaN` at the end
fn sorted<T: AFloat>(a: &[T]) -> Vec<&T> {
    a.iter()
        .sorted_by(|a, b| {
            if a.is_nan() {
                if b.is_nan() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            } else if b.is_nan() {
                Ordering::Less
            } else {
                a.partial_cmp(b).unwrap()
            }
        })
        .collect_vec()
}
#[inline]
fn sorted_quantile<T: AFloat>(a: &[&T], q: T) -> T {
    if a.is_empty() {
        return T::nan();
    }
    let n = a.len() - 1;
    let q = q * T::from_f64(n as f64).unwrap();
    let i = q.floor().to_usize().unwrap();
    if i == n {
        return *a[n];
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
    let solution = xtx_inv.dot(&xty);
    let solution = solution / (a * d - b * c).max(T::epsilon());
    (solution[0], solution[1])
}

fn simd_corr<T: AFloat>(a: &[T], b: &[T]) -> T {
    let a_mean = simd_mean(a);
    let b_mean = simd_mean(b);
    let a = simd_subtract(a, a_mean);
    let b = simd_subtract(b, b_mean);
    let a = a.as_slice();
    let b = b.as_slice();
    let cov = simd_dot(a, b);
    let var1 = simd_inner(a);
    let var2 = simd_inner(b);
    cov / (var1.sqrt() * var2.sqrt())
}
fn simd_nancorr<T: AFloat>(a: &[T], b: &[T]) -> T {
    let num = simd_binary_reduce!(a, b, |x: T, y: T| if x.is_nan() || y.is_nan() {
        T::zero()
    } else {
        T::one()
    });
    if num == T::zero() || num == T::one() {
        return T::nan();
    }
    let a_sum = simd_binary_reduce!(a, b, |x: T, y: T| if x.is_nan() || y.is_nan() {
        T::zero()
    } else {
        x
    });
    let b_sum = simd_binary_reduce!(a, b, |x: T, y: T| if x.is_nan() || y.is_nan() {
        T::zero()
    } else {
        y
    });
    let a_mean = a_sum / num;
    let b_mean = b_sum / num;
    let a = simd_subtract(a, a_mean);
    let b = simd_subtract(b, b_mean);
    let a = a.as_slice();
    let b = b.as_slice();
    let cov = simd_binary_reduce!(a, b, |x: T, y: T| if x.is_nan() || y.is_nan() {
        T::zero()
    } else {
        x * y
    });
    let var1 = simd_binary_reduce!(a, b, |x: T, y: T| if x.is_nan() || y.is_nan() {
        T::zero()
    } else {
        x * x
    });
    let var2 = simd_binary_reduce!(a, b, |x: T, y: T| if x.is_nan() || y.is_nan() {
        T::zero()
    } else {
        y * y
    });
    cov / (var1.sqrt() * var2.sqrt())
}
fn simd_masked_corr<T: AFloat>(a: &[T], b: &[T], valid_mask: &[bool]) -> T {
    let num = simd_unary_reduce!(valid_mask, bool, |x| if x { T::one() } else { T::zero() });
    if num == T::zero() || num == T::one() {
        return T::nan();
    }
    let a_sum = simd_binary_reduce!(a, valid_mask, bool, |x, y| if y { x } else { T::zero() });
    let b_sum = simd_binary_reduce!(b, valid_mask, bool, |x, y| if y { x } else { T::zero() });
    let a_mean = a_sum / num;
    let b_mean = b_sum / num;
    let a = simd_subtract(a, a_mean);
    let b = simd_subtract(b, b_mean);
    let a = a.as_slice();
    let b = b.as_slice();
    let cov = simd_ternary_reduce!(a, b, valid_mask, bool, |x, y, z| if z {
        x * y
    } else {
        T::zero()
    });
    let var1 = simd_binary_reduce!(a, valid_mask, bool, |x, y| if y {
        x * x
    } else {
        T::zero()
    });
    let var2 = simd_binary_reduce!(b, valid_mask, bool, |x, y| if y {
        x * x
    } else {
        T::zero()
    });
    cov / (var1.sqrt() * var2.sqrt())
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
    let x_sorted = sorted(x.as_slice().unwrap());
    let x_med = sorted_median(&x_sorted);
    let x_mad = x_sorted.iter().map(|&x| (*x - x_med).abs()).collect_vec();
    let x_mad = sorted_median(&sorted(&x_mad));
    let hundred = T::from_f64(100.0).unwrap();
    let x_floor = x_med - hundred * x_mad;
    let x_ceil = x_med + hundred * x_mad;
    let x = Array1::from_iter(x.iter().map(|&x| x.max(x_floor).min(x_ceil)));
    let x_mean = x.mean().unwrap();
    let x_std = x.std(T::zero()).max(T::epsilon());
    let mut x = (x - x_mean) / x_std;
    if let Some(q) = q {
        if q > T::zero() {
            let x_sorted = sorted(x.as_slice().unwrap());
            let q_floor = sorted_quantile(&x_sorted, q);
            let q_ceil = sorted_quantile(&x_sorted, T::one() - q);
            let picked_indices: Vec<usize> = x
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| {
                    if x <= q_floor || x >= q_ceil {
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
    coeff_with(x, y, to_valid_indices(valid_mask), q)
}

// macros

macro_rules! parallel_apply {
    ($func:expr, $iter:expr, $slice:expr, $num_threads:expr) => {{
        if $num_threads <= 1 {
            $iter.enumerate().for_each(|(i, args)| {
                $slice.set(i, $func(args));
            });
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads($num_threads)
                .build()
                .unwrap();
            pool.scope(|s| {
                $iter.enumerate().for_each(|(i, args)| {
                    s.spawn(move |_| $slice.set(i, $func(args)));
                });
            });
        }
    }};
}

// axis1 wrappers

pub fn sum_axis1<T: AFloat>(a: &ArrayView2<T>, num_threads: usize) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(&mut res);
    parallel_apply!(
        |row: ArrayView1<T>| simd_sum(row.as_slice().unwrap()),
        a.rows().into_iter(),
        slice,
        num_threads
    );
    res
}
pub fn mean_axis1<T: AFloat>(a: &ArrayView2<T>, num_threads: usize) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(&mut res);
    parallel_apply!(
        |row: ArrayView1<T>| simd_mean(row.as_slice().unwrap()),
        a.rows().into_iter(),
        slice,
        num_threads
    );
    res
}
pub fn nanmean_axis1<T: AFloat>(a: &ArrayView2<T>, num_threads: usize) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(&mut res);
    parallel_apply!(
        |row: ArrayView1<T>| simd_nanmean(row.as_slice().unwrap()),
        a.rows().into_iter(),
        slice,
        num_threads
    );
    res
}
pub fn masked_mean_axis1<T: AFloat>(
    a: &ArrayView2<T>,
    valid_mask: &ArrayView2<bool>,
    num_threads: usize,
) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(&mut res);
    parallel_apply!(
        |(row, valid_mask): (ArrayView1<T>, ArrayView1<bool>)| simd_masked_mean(
            row.as_slice().unwrap(),
            valid_mask.as_slice().unwrap()
        ),
        zip(a.rows(), valid_mask.rows()),
        slice,
        num_threads
    );
    res
}

pub fn corr_axis1<T: AFloat>(a: &ArrayView2<T>, b: &ArrayView2<T>, num_threads: usize) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(&mut res);
    parallel_apply!(
        |(a, b): (ArrayView1<T>, ArrayView1<T>)| simd_corr(
            a.as_slice().unwrap(),
            b.as_slice().unwrap()
        ),
        zip(a.rows(), b.rows()),
        slice,
        num_threads
    );
    res
}
pub fn nancorr_axis1<T: AFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    num_threads: usize,
) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(&mut res);
    parallel_apply!(
        |(a, b): (ArrayView1<T>, ArrayView1<T>)| simd_nancorr(
            a.as_slice().unwrap(),
            b.as_slice().unwrap()
        ),
        zip(a.rows(), b.rows()),
        slice,
        num_threads
    );
    res
}
pub fn masked_corr_axis1<T: AFloat>(
    a: &ArrayView2<T>,
    b: &ArrayView2<T>,
    valid_mask: &ArrayView2<bool>,
    num_threads: usize,
) -> Vec<T> {
    let mut res: Vec<T> = vec![T::zero(); a.nrows()];
    let mut slice = UnsafeSlice::new(&mut res);
    parallel_apply!(
        |(a, b, valid_mask): (ArrayView1<T>, ArrayView1<T>, ArrayView1<bool>)| simd_masked_corr(
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
            valid_mask.as_slice().unwrap()
        ),
        izip!(a.rows(), b.rows(), valid_mask.rows()),
        slice,
        num_threads
    );
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
    let mut slice0 = UnsafeSlice::new(&mut ws);
    let mut slice1 = UnsafeSlice::new(&mut bs);
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
    let mut slice0 = UnsafeSlice::new(&mut ws);
    let mut slice1 = UnsafeSlice::new(&mut bs);
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

pub fn unique(arr: &[i64]) -> (Array1<i64>, Array1<i64>) {
    let mut counts = HashMap::new();

    for &value in arr.iter() {
        *counts.entry(value).or_insert(0) += 1;
    }

    let mut unique_values: Vec<i64> = counts.keys().cloned().collect();
    unique_values.sort();

    let counts: Vec<i64> = unique_values.iter().map(|&value| counts[&value]).collect();

    (Array1::from(unique_values), Array1::from(counts))
}

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

const CONCAT_GROUP_LIMIT: usize = 4 * 239 * 5000;
type ConcatTask<'a, 'b, D> = (Vec<usize>, Vec<ArrayView2<'a, D>>, UnsafeSlice<'b, D>);
#[inline]
fn fill_concat<D: Copy>((offsets, arrays, mut out): ConcatTask<D>) {
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

    let mut tasks: Vec<ConcatTask<D>> = Vec::new();
    let mut current_tasks: Option<ConcatTask<D>> = Some((Vec::new(), Vec::new(), out.shadow()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolkit::convert::to_bytes;
    use std::io::Write;
    use tempfile::tempdir;

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

    #[test]
    fn test_mmap() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.cfy");
        let array = Array1::<f32>::from_shape_vec(3, vec![1., 2., 3.]).unwrap();
        let bytes = unsafe { to_bytes(array.as_slice().unwrap()) };
        let mut file = File::create(&file_path).unwrap();
        file.write_all(bytes).unwrap();
        let file_path = file_path.to_str().unwrap();
        let mmap_array = unsafe { MmapArray1::<f32>::new(file_path).unwrap() };
        assert_eq!(array.len(), mmap_array.len());
        assert_allclose(array.as_slice().unwrap(), unsafe { mmap_array.as_slice() });
        assert_allclose(
            array.as_slice().unwrap(),
            unsafe { mmap_array.as_array_view() }.as_slice().unwrap(),
        );
    }

    macro_rules! test_fast_concat_2d_axis0 {
        ($dtype:ty) => {
            let array_2d_u = ArrayView2::<$dtype>::from_shape((1, 3), &[1., 2., 3.]).unwrap();
            let array_2d_l =
                ArrayView2::<$dtype>::from_shape((2, 3), &[4., 5., 6., 7., 8., 9.]).unwrap();
            let arrays = vec![array_2d_u, array_2d_l];
            let mut out: Vec<$dtype> = vec![0.; 3 * 3];
            let out_slice = UnsafeSlice::new(&mut out);
            fast_concat_2d_axis0(arrays, vec![1, 2], 3, 1, out_slice);
            assert_eq!(out.as_slice(), &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        };
    }

    macro_rules! test_mean_axis1 {
        ($dtype:ty) => {
            let array =
                ArrayView2::<$dtype>::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
            let out = nanmean_axis1(&array, 1);
            assert_allclose(out.as_slice(), &[2., 5.]);
            let out = nanmean_axis1(&array, 2);
            assert_allclose(out.as_slice(), &[2., 5.]);
        };
    }

    macro_rules! test_corr_axis1 {
        ($dtype:ty) => {
            let array =
                ArrayView2::<$dtype>::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
            let out = nancorr_axis1(&array, &(&array + 1.).view(), 1);
            assert_allclose(out.as_slice(), &[1., 1.]);
            let out = nancorr_axis1(&array, &(&array + 1.).view(), 2);
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
        let x = ArrayView2::<f64>::from_shape((2, 3), &[2., 1., 3., 6., 4., 5.]).unwrap();
        let y = ArrayView2::<f64>::from_shape((2, 3), &[4., 2., 6., 12., 8., 10.]).unwrap();
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
