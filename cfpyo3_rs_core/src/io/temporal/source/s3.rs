use super::Source;
use crate::df::frame::DataFrame;
use crate::toolkit::array::AFloat;
use opendal::services::S3;
use opendal::Operator;
use opendal::Result;
use std::marker::PhantomData;

// core implementations

struct S3Client<T: AFloat> {
    op: Operator,
    phantom: PhantomData<T>,
}

impl<T: AFloat> S3Client<T> {
    pub async fn read(&self, key: &str) -> Result<DataFrame<T>> {
        let bytes = self.op.read(key).await?.to_vec();
        let df = unsafe { DataFrame::from_bytes(bytes) };
        Ok(df)
    }

    pub async fn write(&self, key: &str, df: &DataFrame<'_, T>) -> Result<()> {
        self.op.write(key, df.to_bytes()).await
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
