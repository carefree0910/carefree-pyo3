use crate::{
    df::{frame::DataFrame, ColumnsDtype, IndexDtype, COLUMNS_NBYTES},
    toolkit::{
        array::AFloat,
        convert::{to_bytes, to_nbytes},
    },
};
use bytes::BufMut;
use numpy::ndarray::{ArrayView1, ArrayView2};

fn extract_usize(bytes: &[u8]) -> (&[u8], usize) {
    let (target, remain) = bytes.split_at(to_nbytes::<i64>(1));
    let value = i64::from_be_bytes(target.try_into().unwrap());
    (remain, value as usize)
}
fn extract_vec<T: Sized>(bytes: &[u8], nbytes: usize) -> (&[u8], *const T) {
    let (target, remain) = bytes.split_at(nbytes);
    (remain, target.as_ptr() as *const u8 as *const T)
}

impl<'a, T: AFloat> DataFrame<'a, T> {
    /// # Safety
    ///
    /// This method returns a [`Vec<u8>`], but the underlying data is shared with the [`DataFrame`].
    ///
    /// - Its purpose is to provide a convenient interface for you to serialize the [`DataFrame`] to bytes
    /// without copying the data, so you can further write them to a file or send them over the network. It
    /// is generally NOT supposed to be used for any other purpose.
    /// - Since [`Vec<u8>`] is a owned type, double free is possible if you drop both the returned [`Vec<u8>`]
    /// and the underlying [`DataFrame`]. Please make sure that the returned [`Vec<u8>`] is consumed, or call
    /// [`core::mem::forget`] on it when necessary.
    pub unsafe fn to_bytes(&self) -> Vec<u8> {
        let index = &self.index;
        let columns = &self.columns;
        let values = &self.data;
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
        };
        bytes
    }

    /// # Safety
    ///
    /// Please ensure that the `bytes` is returned from the [`DataFrame::to_bytes`] method.
    pub unsafe fn from_bytes(bytes: Vec<u8>) -> Self {
        let bytes = bytes.leak();
        let (bytes, index_nbytes) = extract_usize(bytes);
        let (bytes, columns_nbytes) = extract_usize(bytes);
        let index_shape = index_nbytes / 8;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;

        let (bytes, index_ptr) = extract_vec::<IndexDtype>(bytes, index_nbytes);
        let index_array = ArrayView1::<IndexDtype>::from_shape_ptr((index_shape,), index_ptr);
        let (bytes, columns_ptr) = extract_vec::<ColumnsDtype>(bytes, columns_nbytes);
        let columns_array =
            ArrayView1::<ColumnsDtype>::from_shape_ptr((columns_shape,), columns_ptr);
        let values_nbytes = to_nbytes::<T>(index_shape * columns_shape);
        let (_, values_ptr) = extract_vec::<T>(bytes, values_nbytes);
        let values_array =
            ArrayView2::<T>::from_shape_ptr((index_shape, columns_shape), values_ptr);

        DataFrame::new(
            index_array.into(),
            columns_array.into(),
            values_array.into(),
        )
    }
}
