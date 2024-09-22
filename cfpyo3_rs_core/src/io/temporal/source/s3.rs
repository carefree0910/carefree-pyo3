use super::Source;
use crate::df::frame::DataFrame;
use crate::df::{ColumnsDtype, IndexDtype, COLUMNS_NBYTES};
use crate::toolkit::array::AFloat;
use crate::toolkit::convert::{from_bytes, to_bytes, to_nbytes};
use bytes::{Buf, BufMut};
use numpy::ndarray::{Array1, Array2};
use opendal::services::S3;
use opendal::Result;
use opendal::{Buffer, Operator};
use std::marker::PhantomData;

// core implementations

struct S3Client<T: AFloat> {
    op: Operator,
    phantom: PhantomData<T>,
}

fn extract_vec<T: Sized>(rv: &mut Buffer, nbytes: usize) -> Vec<T> {
    let vec_u8 = rv.slice(..nbytes).to_vec();
    let vec = unsafe { from_bytes(vec_u8) };
    rv.advance(nbytes);
    vec
}

impl<T: AFloat> S3Client<T> {
    pub async fn read(&self, key: &str) -> Result<DataFrame<T>> {
        let mut rv = self.op.read(key).await?;
        let index_nbytes = rv.get_i64() as usize;
        let columns_nbytes = rv.get_i64() as usize;

        let index_shape = index_nbytes / 8;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;

        let index_array = Array1::<IndexDtype>::from_shape_vec(
            (index_shape,),
            extract_vec::<IndexDtype>(&mut rv, index_nbytes),
        )
        .unwrap();
        let columns_array = Array1::<ColumnsDtype>::from_shape_vec(
            (columns_shape,),
            extract_vec::<ColumnsDtype>(&mut rv, columns_nbytes),
        )
        .unwrap();
        let values_nbytes = rv.len();
        let values_array = Array2::<T>::from_shape_vec(
            (index_shape, columns_shape),
            extract_vec::<T>(&mut rv, values_nbytes),
        )
        .unwrap();

        Ok(DataFrame::new(
            index_array.into(),
            columns_array.into(),
            values_array.into(),
        ))
    }

    pub async fn write(&self, key: &str, df: &DataFrame<'_, T>) -> Result<()> {
        let index = &df.index;
        let columns = &df.columns;
        let values = &df.data;
        let index_nbytes = to_nbytes::<IndexDtype>(index.len());
        let columns_nbytes = to_nbytes::<ColumnsDtype>(columns.len());
        let total_nbytes = index_nbytes + columns_nbytes + to_nbytes::<T>(values.len());
        let mut bytes: Vec<u8> = Vec::with_capacity(total_nbytes + 16);
        bytes.put_i64(index_nbytes as i64);
        bytes.put_i64(columns_nbytes as i64);
        unsafe {
            bytes.put_slice(to_bytes(index.as_slice().unwrap()));
            bytes.put_slice(to_bytes(columns.as_slice().unwrap()));
            bytes.put_slice(to_bytes(values.as_slice().unwrap()));
        }
        self.op.write(key, bytes).await
    }
}

// public interface

pub struct S3Source<T: AFloat>(S3Client<T>);

impl<T: AFloat> S3Source<T> {
    pub fn new(bucket: &str, endpoint: &str) -> Self {
        let builder = S3::default()
            .region("auto")
            .bucket(bucket)
            .endpoint(endpoint);
        let op = Operator::new(builder)
            .expect("failed to initialize s3 client")
            .finish();

        Self(S3Client {
            op,
            phantom: PhantomData,
        })
    }
    #[inline]
    fn to_s3_key(&self, date: &str, key: &str) -> String {
        format!("{}/{}", key, date)
    }
}

impl<T: AFloat> Source<T> for S3Source<T> {
    async fn read(&self, date: &str, key: &str) -> DataFrame<T> {
        self.0
            .read(self.to_s3_key(date, key).as_str())
            .await
            .unwrap()
    }
    async fn write(&self, date: &str, key: &str, df: &DataFrame<'_, T>) {
        self.0
            .write(self.to_s3_key(date, key).as_str(), df)
            .await
            .unwrap()
    }
}
