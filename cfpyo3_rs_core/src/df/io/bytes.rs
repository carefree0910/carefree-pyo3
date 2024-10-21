use crate::{
    as_data_slice_or_err,
    df::{
        meta::{align_nbytes, DF_ALIGN},
        ColumnsDtype, DataFrame, IndexDtype, COLUMNS_NBYTES, INDEX_NBYTES,
    },
    toolkit::{
        array::AFloat,
        convert::{to_bytes, to_nbytes},
    },
};
use anyhow::Result;
use bytes::BufMut;
use core::mem;

fn extract_usize(bytes: &[u8]) -> Result<(&[u8], usize)> {
    let (target, remain) = bytes.split_at(to_nbytes::<i64>(1));
    let value = i64::from_le_bytes(target.try_into()?);
    Ok((remain, value as usize))
}
fn extract_ptr(bytes: &[u8], nbytes: usize) -> (&[u8], *const u8) {
    let (target, remain) = bytes.split_at(nbytes);
    (remain, target.as_ptr())
}

fn put_aligned_slice(bytes: &mut Vec<u8>, slice: &[u8]) {
    bytes.put_slice(slice);
    let remainder = slice.len() % DF_ALIGN;
    if remainder != 0 {
        let padding = DF_ALIGN - remainder;
        bytes.put_slice(&[0u8; DF_ALIGN][..padding]);
    }
}

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let index = self.index();
        let columns = self.columns();
        let values = self.values();
        let index_nbytes = to_nbytes::<IndexDtype>(index.len());
        let columns_nbytes = to_nbytes::<ColumnsDtype>(columns.len());
        let values_nbytes = to_nbytes::<T>(values.len());

        let index_aligned_nbytes = align_nbytes(index_nbytes);
        let columns_aligned_nbytes = align_nbytes(columns_nbytes);
        let values_aligned_nbytes = align_nbytes(values_nbytes);
        let total_aligned_nbytes =
            index_aligned_nbytes + columns_aligned_nbytes + values_aligned_nbytes + 16;
        let aligned_bytes: Vec<[u8; DF_ALIGN]> =
            Vec::with_capacity(total_aligned_nbytes / DF_ALIGN);
        let mut bytes: Vec<u8> = unsafe {
            Vec::from_raw_parts(aligned_bytes.as_ptr() as *mut u8, 0, total_aligned_nbytes)
        };
        mem::forget(aligned_bytes);

        bytes.put_i64_le(index_nbytes as i64);
        bytes.put_i64_le(columns_nbytes as i64);
        unsafe {
            put_aligned_slice(&mut bytes, to_bytes(as_data_slice_or_err!(index)));
            put_aligned_slice(&mut bytes, to_bytes(as_data_slice_or_err!(columns)));
            put_aligned_slice(&mut bytes, to_bytes(as_data_slice_or_err!(values)));
        };
        Ok(bytes)
    }

    /// Create a [`DataFrame`] from a compact bytes slice, which is usually created by the [`DataFrame::to_bytes`] method.
    ///
    /// The intention of this method is to create a 'read-only' [`DataFrame`] view. If you need
    /// an owned [`DataFrame`], you may call [`DataFrame::to_owned`] method on the returned [`DataFrame`].
    ///
    /// # Safety
    ///
    /// The safety concern only comes from whether the `bytes` is of the desired memory layout,
    /// the mutabilities are safe because we borrow the `bytes` immutably and specify the lifetime.
    pub unsafe fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let (bytes, index_nbytes) = extract_usize(bytes)?;
        let (bytes, columns_nbytes) = extract_usize(bytes)?;

        let index_shape = index_nbytes / INDEX_NBYTES;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;

        let (bytes, index_ptr) = extract_ptr(bytes, index_nbytes);
        let (bytes, columns_ptr) = extract_ptr(bytes, columns_nbytes);
        let values_nbytes = to_nbytes::<T>(index_shape * columns_shape);
        let (_, values_ptr) = extract_ptr(bytes, values_nbytes);

        Ok(DataFrame::from_ptr(
            index_ptr,
            index_shape,
            columns_ptr,
            columns_shape,
            values_ptr,
        ))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::toolkit::convert::from_vec;

    pub fn get_test_df<'a>() -> DataFrame<'a, f32> {
        let mut index_vec: Vec<u8> = vec![0; INDEX_NBYTES];
        index_vec[0] = 1;
        let index_vec = unsafe { from_vec(index_vec) };
        let mut columns_vec: Vec<u8> = vec![0; COLUMNS_NBYTES];
        columns_vec[0] = 2;
        let columns_vec = unsafe { from_vec(columns_vec) };
        let mut values_vec: Vec<u8> = vec![0; mem::size_of::<f32>()];
        values_vec[0] = 3;
        let values_vec = unsafe { from_vec(values_vec) };

        DataFrame::<f32>::from_vec(index_vec, columns_vec, values_vec).unwrap()
    }

    #[test]
    fn test_bytes_io() {
        let df = get_test_df();
        let bytes = df.to_bytes().unwrap();
        #[rustfmt::skip]
        {
            assert_eq!(
                bytes,
                [
                    8, 0, 0, 0, 0, 0, 0, 0,
                    32, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0,
                    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    3, 0, 0, 0, 0, 0, 0, 0,
                ]
            );
        };
        let loaded = unsafe { DataFrame::<f32>::from_bytes(&bytes).unwrap() };
        assert_eq!(df.index(), loaded.index());
        assert_eq!(df.columns(), loaded.columns());
        assert_eq!(df.values(), loaded.values());
    }
}
