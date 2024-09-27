use crate::df::frame::DataFrame;
use crate::df::{COLUMNS_NBYTES, INDEX_NBYTES};
use crate::toolkit::array::AFloat;
use crate::toolkit::convert::to_nbytes;
use bytes::Buf;

fn extract_vec(buf: &mut impl Buf, nbytes: usize) -> *const u8 {
    let ptr = buf.copy_to_bytes(nbytes).as_ptr();
    buf.advance(nbytes);
    ptr
}

impl<'a, T: AFloat> DataFrame<'a, T> {
    /// # Safety
    ///
    /// Please ensure that the `buf` represents bytes returned from the [`DataFrame::to_bytes`] method.
    pub unsafe fn from_buffer(buf: &mut impl Buf) -> Self {
        let index_nbytes = buf.get_i64() as usize;
        let columns_nbytes = buf.get_i64() as usize;

        let index_shape = index_nbytes / INDEX_NBYTES;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;

        let index_ptr = extract_vec(buf, index_nbytes);
        let columns_ptr = extract_vec(buf, columns_nbytes);
        let values_nbytes = to_nbytes::<T>(index_shape * columns_shape);
        let values_ptr = extract_vec(buf, values_nbytes);

        DataFrame::from(
            index_ptr,
            index_shape,
            columns_ptr,
            columns_shape,
            values_ptr,
        )
    }
}
