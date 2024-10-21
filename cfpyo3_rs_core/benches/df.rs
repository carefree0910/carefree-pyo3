use cfpyo3_core::toolkit::convert::from_vec;
use cfpyo3_core::{df::*, toolkit::convert::to_nbytes};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn get_test_df<'a>(nindex: usize, ncolumns: usize) -> DataFrame<'a, f32> {
    let mut index_vec: Vec<u8> = vec![0; INDEX_NBYTES * nindex];
    index_vec[0] = 1;
    let index_vec = unsafe { from_vec(index_vec) };
    let mut columns_vec: Vec<u8> = vec![0; COLUMNS_NBYTES * ncolumns];
    columns_vec[0] = 2;
    let columns_vec = unsafe { from_vec(columns_vec) };
    let mut values_vec: Vec<u8> = vec![0; to_nbytes::<f32>(nindex * ncolumns)];
    values_vec[0] = 3;
    let values_vec = unsafe { from_vec(values_vec) };

    DataFrame::<f32>::from_vec(index_vec, columns_vec, values_vec).unwrap()
}

fn bench_bytes_io(c: &mut Criterion) {
    let df = get_test_df(239, 5000);
    let df_bytes = df.to_bytes().unwrap();
    c.bench_function("df_io_to_bytes (f32)", |b| {
        b.iter(|| black_box(&df).to_bytes().unwrap())
    });
    c.bench_function("df_io_from_bytes (f32)", |b| {
        b.iter(|| unsafe { black_box(DataFrame::<f32>::from_bytes(&df_bytes).unwrap()) })
    });
    c.bench_function("df_io_from_buffer (f32)", |b| {
        b.iter(|| {
            let buf = black_box(&df_bytes).as_slice();
            black_box(DataFrame::<f32>::from_buffer(buf).unwrap())
        })
    });
}

criterion_group!(benches, bench_bytes_io);
criterion_main!(benches);
