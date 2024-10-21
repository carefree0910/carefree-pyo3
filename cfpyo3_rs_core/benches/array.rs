use cfpyo3_core::toolkit::array::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray_rand::RandomExt;
use ndarray_rand::{
    rand::seq::SliceRandom,
    rand_distr::{Bernoulli, Uniform},
};
use numpy::ndarray::{Array1, Array2};

macro_rules! bench_mean_axis1 {
    ($c:expr, $multiplier:expr, $nthreads:expr, $a32:expr, $a64:expr, $mask:expr) => {{
        let name_f32 = format!("mean_axis1 (f32) (x{}, {} threads)", $multiplier, $nthreads);
        let name_f64 = format!("mean_axis1 (f64) (x{}, {} threads)", $multiplier, $nthreads);
        $c.bench_function(&name_f32, |b| {
            b.iter(|| mean_axis1(black_box($a32), black_box($nthreads)))
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| mean_axis1(black_box($a64), black_box($nthreads)))
        });
        let name_f32 = format!(
            "nanmean_axis1 (f32) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        let name_f64 = format!(
            "nanmean_axis1 (f64) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        $c.bench_function(&name_f32, |b| {
            b.iter(|| nanmean_axis1(black_box($a32), black_box($nthreads)))
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| nanmean_axis1(black_box($a64), black_box($nthreads)))
        });
        let name_f32 = format!(
            "masked_mean_axis1 (f32) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        let name_f64 = format!(
            "masked_mean_axis1 (f64) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        $c.bench_function(&name_f32, |b| {
            b.iter(|| masked_mean_axis1(black_box($a32), black_box($mask), black_box($nthreads)))
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| masked_mean_axis1(black_box($a64), black_box($mask), black_box($nthreads)))
        });
    }};
}
macro_rules! bench_mean_axis1_full {
    ($c:expr, $multiplier:expr) => {
        let array_f32 = Array2::<f32>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let array_f64 = Array2::<f64>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let mask = Array2::<bool>::random((239 * $multiplier, 5000), Bernoulli::new(0.5).unwrap());
        let array_f32 = &array_f32.view();
        let array_f64 = &array_f64.view();
        let mask = &mask.view();
        bench_mean_axis1!($c, $multiplier, 1, array_f32, array_f64, mask);
        bench_mean_axis1!($c, $multiplier, 2, array_f32, array_f64, mask);
        bench_mean_axis1!($c, $multiplier, 4, array_f32, array_f64, mask);
    };
}
macro_rules! bench_corr_axis1 {
    ($c:expr, $multiplier:expr, $nthreads:expr, $a32:expr, $a64:expr, $mask:expr) => {{
        let name_f32 = format!("corr_axis1 (f32) (x{}, {} threads)", $multiplier, $nthreads);
        let name_f64 = format!("corr_axis1 (f64) (x{}, {} threads)", $multiplier, $nthreads);
        $c.bench_function(&name_f32, |b| {
            b.iter(|| corr_axis1(black_box($a32), black_box($a32), black_box($nthreads)))
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| corr_axis1(black_box($a64), black_box($a64), black_box($nthreads)))
        });
        let name_f32 = format!(
            "nancorr_axis1 (f32) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        let name_f64 = format!(
            "nancorr_axis1 (f64) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        $c.bench_function(&name_f32, |b| {
            b.iter(|| nancorr_axis1(black_box($a32), black_box($a32), black_box($nthreads)))
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| nancorr_axis1(black_box($a64), black_box($a64), black_box($nthreads)))
        });
        let name_f32 = format!(
            "masked_corr_axis1 (f32) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        let name_f64 = format!(
            "masked_corr_axis1 (f64) (x{}, {} threads)",
            $multiplier, $nthreads
        );
        $c.bench_function(&name_f32, |b| {
            b.iter(|| {
                masked_corr_axis1(
                    black_box($a32),
                    black_box($a32),
                    black_box($mask),
                    black_box($nthreads),
                )
            })
        });
        $c.bench_function(&name_f64, |b| {
            b.iter(|| {
                masked_corr_axis1(
                    black_box($a64),
                    black_box($a64),
                    black_box($mask),
                    black_box($nthreads),
                )
            })
        });
    }};
}
macro_rules! bench_corr_axis1_full {
    ($c:expr, $multiplier:expr) => {
        let array_f32 = Array2::<f32>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let array_f64 = Array2::<f64>::random((239 * $multiplier, 5000), Uniform::new(0., 1.));
        let mask = Array2::<bool>::random((239 * $multiplier, 5000), Bernoulli::new(0.5).unwrap());
        let array_f32 = &array_f32.view();
        let array_f64 = &array_f64.view();
        let mask = &mask.view();
        bench_corr_axis1!($c, $multiplier, 1, array_f32, array_f64, mask);
        bench_corr_axis1!($c, $multiplier, 2, array_f32, array_f64, mask);
        bench_corr_axis1!($c, $multiplier, 4, array_f32, array_f64, mask);
    };
}

fn bench_simd_ops(c: &mut Criterion) {
    let array_f32 = Array1::<f32>::random(239 * 5000, Uniform::new(0., 1.));
    let array_f64 = Array1::<f64>::random(239 * 5000, Uniform::new(0., 1.));
    let mask = Array1::<bool>::random(239 * 5000, Bernoulli::new(0.5).unwrap());
    let array_f32 = array_f32.as_slice().unwrap();
    let array_f64 = array_f64.as_slice().unwrap();
    let mask = mask.as_slice().unwrap();
    c.bench_function("sum (f32)", |b| b.iter(|| simd_sum(black_box(array_f32))));
    c.bench_function("sum (f64)", |b| b.iter(|| simd_sum(black_box(array_f64))));
    c.bench_function("nanmean (f32)", |b| {
        b.iter(|| simd_nanmean(black_box(array_f32)))
    });
    c.bench_function("nanmean (f64)", |b| {
        b.iter(|| simd_nanmean(black_box(array_f64)))
    });
    c.bench_function("masked_mean (f32)", |b| {
        b.iter(|| simd_masked_mean(black_box(array_f32), black_box(mask)))
    });
    c.bench_function("masked_mean (f64)", |b| {
        b.iter(|| simd_masked_mean(black_box(array_f64), black_box(mask)))
    });
    c.bench_function("subtract (f32)", |b| {
        b.iter(|| simd_subtract(black_box(array_f32), black_box(0.123)))
    });
    c.bench_function("subtract (f64)", |b| {
        b.iter(|| simd_subtract(black_box(array_f64), black_box(0.123)))
    });
    c.bench_function("dot (f32)", |b| {
        b.iter(|| simd_dot(black_box(array_f32), black_box(array_f32)))
    });
    c.bench_function("dot (f64)", |b| {
        b.iter(|| simd_dot(black_box(array_f64), black_box(array_f64)))
    });
    c.bench_function("inner (f32)", |b| {
        b.iter(|| simd_inner(black_box(array_f32)))
    });
    c.bench_function("inner (f64)", |b| {
        b.iter(|| simd_inner(black_box(array_f64)))
    });
}

fn bench_to_valid_indices(c: &mut Criterion) {
    let mask = Array1::<bool>::random(239 * 5000, Bernoulli::new(0.5).unwrap());
    let mask = mask.view();
    c.bench_function("to_valid_indices", |b| {
        b.iter(|| to_valid_indices(black_box(mask)))
    });
}

fn bench_axis1_ops(c: &mut Criterion) {
    bench_mean_axis1_full!(c, 1);
    bench_mean_axis1_full!(c, 2);
    bench_mean_axis1_full!(c, 4);
    bench_corr_axis1_full!(c, 1);
    bench_corr_axis1_full!(c, 2);
    bench_corr_axis1_full!(c, 4);
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

fn bench_unsafe_slice(c: &mut Criterion) {
    let num_total = 239 * 5000;
    let array_f32 = Array1::<f32>::random(num_total, Uniform::new(0., 1.)).to_vec();
    let array_f64 = Array1::<f64>::random(num_total, Uniform::new(0., 1.)).to_vec();
    let array_f32_slice = array_f32.as_slice();
    let array_f64_slice = array_f64.as_slice();
    c.bench_function("array clone (f32)", |b| {
        b.iter(|| black_box(&array_f32).clone())
    });
    c.bench_function("array clone (f64)", |b| {
        b.iter(|| black_box(&array_f64).clone())
    });
    c.bench_function("extend_from_slice (f32)", |b| {
        b.iter(|| {
            let empty: Vec<f32> = black_box(Vec::with_capacity(num_total));
            black_box(empty).extend_from_slice(array_f32_slice);
        })
    });
    c.bench_function("extend_from_slice (f64)", |b| {
        b.iter(|| {
            let empty: Vec<f64> = black_box(Vec::with_capacity(num_total));
            black_box(empty).extend_from_slice(array_f64_slice);
        })
    });
    c.bench_function("unsafe_slice copy (f32)", |b| {
        b.iter(|| {
            let mut empty: Vec<f32> = black_box(vec![0.0; num_total]);
            let empty = black_box(UnsafeSlice::new(&mut empty));
            black_box(empty).copy_from_slice(0, array_f32_slice)
        })
    });
    c.bench_function("unsafe_slice copy (f64)", |b| {
        b.iter(|| {
            let mut empty: Vec<f64> = black_box(vec![0.0; num_total]);
            let empty = black_box(UnsafeSlice::new(&mut empty));
            black_box(empty).copy_from_slice(0, array_f64_slice)
        })
    });
}

criterion_group!(
    benches,
    bench_simd_ops,
    bench_to_valid_indices,
    bench_axis1_ops,
    bench_searchsorted,
    bench_unsafe_slice,
);
criterion_main!(benches);
