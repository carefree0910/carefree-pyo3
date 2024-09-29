use cfpyo3_core::toolkit::array::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray_rand::RandomExt;
use ndarray_rand::{rand::seq::SliceRandom, rand_distr::Uniform};
use numpy::ndarray::{Array1, Array2};

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
        bench_mean_axis1!($c, $multiplier, 2, array_f32, array_f64);
        bench_mean_axis1!($c, $multiplier, 4, array_f32, array_f64);
        bench_mean_axis1!($c, $multiplier, 8, array_f32, array_f64);
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

fn bench_axis1_ops(c: &mut Criterion) {
    bench_mean_axis1_full!(c, 1);
    bench_mean_axis1_full!(c, 2);
    bench_mean_axis1_full!(c, 4);
    bench_mean_axis1_full!(c, 8);
    bench_corr_axis1_full!(c, 1);
    bench_corr_axis1_full!(c, 2);
    bench_corr_axis1_full!(c, 4);
    bench_corr_axis1_full!(c, 8);
}

fn bench_searchsorted(c: &mut Criterion) {
    let total = 5000;
    let array_i64 = Array1::<i64>::from_iter(0..total);
    let array_i64 = &array_i64.view();
    for amount in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048].iter() {
        let random_picked: Vec<i64> = array_i64
            .as_slice()
            .unwrap()
            .choose_multiple(&mut rand::thread_rng(), *amount)
            .cloned()
            .collect();
        let random_picked = Array1::<i64>::from(random_picked);
        let random_picked = &random_picked.view();
        c.bench_function(&format!("searchsorted ({} / {})", amount, total), |b| {
            b.iter(|| batch_searchsorted(black_box(array_i64), black_box(random_picked)))
        });
    }
}

criterion_group!(benches, bench_axis1_ops, bench_searchsorted);
criterion_main!(benches);
