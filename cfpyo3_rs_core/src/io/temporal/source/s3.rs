use super::Source;
use crate::df::frame::DataFrame;
use crate::df::{ColumnsDtype, IndexDtype, COLUMNS_NBYTES};
use crate::toolkit::array::AFloat;
use crate::toolkit::convert::{from_bytes, to_bytes};
use bytes::{Buf, BufMut};
use numpy::ndarray::{Array1, Array2};
use opendal::services::S3;
use opendal::Result;
use opendal::{Buffer, Operator};
use std::marker::PhantomData;

/// core implementations

struct S3Client<T: AFloat> {
    op: Operator,
    phantom: PhantomData<T>,
}

fn extract_vec<T: Sized>(rv: &mut Buffer, len: usize) -> Vec<T> {
    let vec_u8 = rv.slice(..len).to_vec();
    let vec = from_bytes(vec_u8);
    rv.advance(len);
    vec
}

impl<T: AFloat> S3Client<T> {
    pub async fn read(&self, key: &str) -> Result<DataFrame<T>> {
        let mut rv = self.op.read(key).await?;
        let index_len = rv.get_i64() as usize;
        let columns_len = rv.get_i64() as usize;

        let index_shape = index_len / 8;
        let columns_shape = columns_len / COLUMNS_NBYTES;

        let index_array = Array1::<IndexDtype>::from_shape_vec(
            (index_shape,),
            extract_vec::<IndexDtype>(&mut rv, index_len),
        )
        .unwrap();
        let columns_array = Array1::<ColumnsDtype>::from_shape_vec(
            (columns_shape,),
            extract_vec::<ColumnsDtype>(&mut rv, columns_len),
        )
        .unwrap();
        let values_len = rv.len();
        let values_array = Array2::<T>::from_shape_vec(
            (index_shape, columns_shape),
            extract_vec::<T>(&mut rv, values_len),
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
        let index_len = index.len() * 8;
        let columns_len = columns.len() * COLUMNS_NBYTES;
        let total_bytes = index_len + columns_len + values.len() * 8;
        let mut bytes: Vec<u8> = Vec::with_capacity(total_bytes + 16);
        bytes.put_i64(index_len as i64);
        bytes.put_i64(columns_len as i64);
        bytes.put_slice(to_bytes(index.as_slice().unwrap()));
        bytes.put_slice(to_bytes(columns.as_slice().unwrap()));
        bytes.put_slice(to_bytes(values.as_slice().unwrap()));
        self.op.write(key, bytes).await
    }
}

/// public interface

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
