use crate::df::meta::align_nbytes;
use crate::df::DataFrame;
use crate::df::{COLUMNS_NBYTES, INDEX_NBYTES};
use crate::toolkit::array::AFloat;
use crate::toolkit::convert::{from_vec, to_nbytes};
use anyhow::Result;
use bytes::Buf;

fn extract_bytes<T: Sized>(buf: &mut impl Buf, nbytes: usize) -> Vec<T> {
    // `advance` will happen inside `copy_to_bytes`
    let mut vec_u8 = vec![0; nbytes];
    buf.copy_to_slice(&mut vec_u8);
    let remainder = align_nbytes(nbytes) - nbytes;
    if remainder != 0 {
        buf.advance(remainder);
    }
    unsafe { from_vec(vec_u8) }
}

impl<'a, T: AFloat> DataFrame<'a, T> {
    /// Create an **owned** [`DataFrame`] from a [`Buf`], whose underlying bytes are returned from the [`DataFrame::to_bytes`] method.
    ///
    /// Since the returned [`DataFrame`] is owned:
    /// - it is safe to use it anyway you like.
    /// - copying is occurred during this method.
    ///
    /// If you want a zero-copy loading, you can try to use the [`DataFrame::from_bytes`] method with your [`Buf`].
    ///
    /// # Panics
    ///
    /// This method will panic if:
    /// - the bytes behind the `buf` is not of the desired memory layout.
    /// - the `buf` is not fully consumed after loading.
    pub fn from_buffer(mut buf: impl Buf) -> Result<Self> {
        let index_nbytes = buf.get_i64_le() as usize;
        let columns_nbytes = buf.get_i64_le() as usize;

        let index_shape = index_nbytes / INDEX_NBYTES;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;

        let index_vec = extract_bytes(&mut buf, index_nbytes);
        let columns_vec = extract_bytes(&mut buf, columns_nbytes);
        let values_nbytes = to_nbytes::<T>(index_shape * columns_shape);
        let values_vec = extract_bytes(&mut buf, values_nbytes);

        assert!(
            buf.remaining() == 0,
            "internal error: buffer not fully consumed"
        );

        DataFrame::from_vec(index_vec, columns_vec, values_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df::io::bytes::tests::get_test_df;

    #[test]
    fn test_buffer_io() {
        let df = get_test_df();
        let bytes = df.to_bytes().unwrap();
        let loaded = DataFrame::<f32>::from_buffer(bytes.as_slice()).unwrap();
        assert_eq!(df.index(), loaded.index());
        assert_eq!(df.columns(), loaded.columns());
        assert_eq!(df.values(), loaded.values());
    }
}
