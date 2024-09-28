use cfpyo3_core::toolkit::array::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::ndarray::Array2;

macro_rules! bench_mean_axis1 {
    ($c:expr, $multiplier:expr, $nthreads:expr, $a32:expr, $a64:expr) => {{
        let name_f32 = format!("mean_axis1 (f32) (x{}, {} threads)", $multiplier, $nthreads);
        let name_f64 = format!("mean_axis1 (f64) (x{}, {} threads)", $multiplier, $nthreads);
        $c.bench_function(&name_f32, |b| {
            b.iter(|| mean_axis1(black_box($a32), black_box($nthreads)))
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| mean_axis1(black_box($a64), black_box($nthreads)))
        });
    }};
}
macro_rules! bench_mean_axis1_full {
    ($c:expr, $multiplier:expr) => {
        let array_f32 = Array2::<f32>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let array_f64 = Array2::<f64>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let array_f32 = &array_f32.view();
        let array_f64 = &array_f64.view();
        bench_mean_axis1!($c, $multiplier, 1, array_f32, array_f64);
        // bench_mean_axis1!($c, $multiplier, 2, array_f32, array_f64);
        // bench_mean_axis1!($c, $multiplier, 4, array_f32, array_f64);
        // bench_mean_axis1!($c, $multiplier, 8, array_f32, array_f64);
    };
}
macro_rules! bench_corr_axis1 {
    ($c:expr, $multiplier:expr, $nthreads:expr, $a32:expr, $a64:expr) => {{
        let name_f32 = format!("corr_axis1 (f32) (x{}, {} threads)", $multiplier, $nthreads);
        let name_f64 = format!("corr_axis1 (f64) (x{}, {} threads)", $multiplier, $nthreads);
        $c.bench_function(&name_f32, |b| {
            b.iter(|| corr_axis1(black_box($a32), black_box($a32), black_box($nthreads)))
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| corr_axis1(black_box($a64), black_box($a64), black_box($nthreads)))
        });
    }};
}
macro_rules! bench_corr_axis1_full {
    ($c:expr, $multiplier:expr) => {
        let array_f32 = Array2::<f32>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let array_f64 = Array2::<f64>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let array_f32 = &array_f32.view();
        let array_f64 = &array_f64.view();
        bench_corr_axis1!($c, $multiplier, 1, array_f32, array_f64);
        bench_corr_axis1!($c, $multiplier, 2, array_f32, array_f64);
        bench_corr_axis1!($c, $multiplier, 4, array_f32, array_f64);
        bench_corr_axis1!($c, $multiplier, 8, array_f32, array_f64);
    };
}

pub fn bench_axis1_ops(c: &mut Criterion) {
    bench_mean_axis1_full!(c, 1);
    bench_mean_axis1_full!(c, 2);
    bench_mean_axis1_full!(c, 4);
    bench_mean_axis1_full!(c, 8);
    bench_corr_axis1_full!(c, 1);
    bench_corr_axis1_full!(c, 2);
    bench_corr_axis1_full!(c, 4);
    bench_corr_axis1_full!(c, 8);
}

criterion_group!(benches, bench_axis1_ops);
criterion_main!(benches);
