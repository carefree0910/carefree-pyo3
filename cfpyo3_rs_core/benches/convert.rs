use cfpyo3_core::toolkit::convert::*;
use core::mem;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::ndarray::Array1;

fn bench_convert(c: &mut Criterion) {
    let num_total = 239 * 5000;
    let array_f32 = Array1::<f32>::random(num_total, Uniform::new(0., 1.)).to_vec();
    let array_f64 = Array1::<f64>::random(num_total, Uniform::new(0., 1.)).to_vec();
    let array_f32_slice = array_f32.as_slice();
    let array_f64_slice = array_f64.as_slice();
    let array_f32_ptr = unsafe { to_bytes(&array_f32).to_vec().as_ptr() };
    let array_f64_ptr = unsafe { to_bytes(&array_f64).to_vec().as_ptr() };
    c.bench_function("to_bytes (f32)", |b| {
        b.iter(|| unsafe { to_bytes(black_box(array_f32_slice)) })
    });
    c.bench_function("to_bytes (f64)", |b| {
        b.iter(|| unsafe { to_bytes(black_box(array_f64_slice)) })
    });
    c.bench_function("from_bytes (f32)", |b| {
        b.iter(|| unsafe {
            let rv = black_box(from_ptr::<f32, u8>(array_f32_ptr, num_total));
            mem::forget(black_box(rv));
        })
    });
    c.bench_function("from_bytes (f64)", |b| {
        b.iter(|| unsafe {
            let rv = black_box(from_ptr::<f64, u8>(array_f64_ptr, num_total));
            mem::forget(black_box(rv));
        })
    });
}

criterion_group!(benches, bench_convert);
criterion_main!(benches);
