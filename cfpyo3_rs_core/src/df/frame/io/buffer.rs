use crate::df::frame::meta::align_nbytes;
use crate::df::frame::DataFrame;
use crate::df::{COLUMNS_NBYTES, INDEX_NBYTES};
use crate::toolkit::array::AFloat;
use crate::toolkit::convert::to_nbytes;
use bytes::{Buf, Bytes};

fn extract_bytes(buf: &mut impl Buf, nbytes: usize) -> Bytes {
    // `advance` will happen inside `copy_to_bytes`
    buf.copy_to_bytes(align_nbytes(nbytes))
}

impl<'a, T: AFloat> DataFrame<'a, T> {
    /// # Safety
    ///
    /// Please ensure that the `buf` represents bytes returned from the [`DataFrame::to_bytes`] method.
    pub unsafe fn from_buffer(buf: &mut impl Buf) -> Self {
        let index_nbytes = buf.get_i64_le() as usize;
        let columns_nbytes = buf.get_i64_le() as usize;

        let index_shape = index_nbytes / INDEX_NBYTES;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;

        let index_bytes = extract_bytes(buf, index_nbytes);
        let index_ptr = index_bytes.as_ptr();
        let columns_bytes = extract_bytes(buf, columns_nbytes);
        let columns_ptr = columns_bytes.as_ptr();
        let values_nbytes = to_nbytes::<T>(index_shape * columns_shape);
        let values_bytes = extract_bytes(buf, values_nbytes);
        let values_ptr = values_bytes.as_ptr();

        core::mem::forget(index_bytes);
        core::mem::forget(columns_bytes);
        core::mem::forget(values_bytes);

        DataFrame::from_ptr(
            index_ptr,
            index_shape,
            columns_ptr,
            columns_shape,
            values_ptr,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df::frame::io::bytes::tests::get_test_df;

    #[test]
    fn test_buffer_io() {
        let df = get_test_df();
        let bytes = df.to_bytes();
        let buf = &mut bytes.as_slice();
        let loaded = unsafe { DataFrame::<f32>::from_buffer(buf) };
        assert_eq!(df.index, loaded.index);
        assert_eq!(df.columns, loaded.columns);
        assert_eq!(df.values, loaded.values);
    }
}
