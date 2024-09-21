use cfpyo3_core::toolkit::array;
use numpy::ndarray::ArrayView2;

fn assert_allclose<T: array::AFloat>(a: &[T], b: &[T]) {
    let atol = T::from_f64(1e-6).unwrap();
    let rtol = T::from_f64(1e-6).unwrap();
    a.iter().zip(b.iter()).for_each(|(&x, &y)| {
        assert!(
            (x - y).abs() <= atol + rtol * y.abs(),
            "not close - x: {:?}, y: {:?}",
            x,
            y
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
        let out_slice = array::UnsafeSlice::new(out.as_mut_slice());
        array::fast_concat_2d_axis0(arrays, vec![1, 2], 3, 1, out_slice);
        assert_eq!(out.as_slice(), &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    };
}

macro_rules! test_mean_axis1 {
    ($dtype:ty) => {
        let array = ArrayView2::<$dtype>::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
        let out = array::mean_axis1(&array, 1);
        assert_allclose(out.as_slice(), &[2., 5.]);
    };
}

macro_rules! test_corr_axis1 {
    ($dtype:ty) => {
        let array = ArrayView2::<$dtype>::from_shape((2, 3), &[1., 2., 3., 4., 5., 6.]).unwrap();
        let out = array::corr_axis1(&array, &(&array + 1.).view(), 1);
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
