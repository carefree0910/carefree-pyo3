use crate::df::frame::DataFrame;
use crate::df::{ColumnsDtype, IndexDtype, COLUMNS_NBYTES};
use crate::toolkit::array::AFloat;
use crate::toolkit::convert::from_bytes;
use bytes::Buf;
use numpy::ndarray::{Array1, Array2};
use opendal::Buffer;

fn extract_vec<T: Sized>(rv: &mut Buffer, nbytes: usize) -> Vec<T> {
    let vec_u8 = rv.slice(..nbytes).to_vec();
    let vec = unsafe { from_bytes(vec_u8) };
    rv.advance(nbytes);
    vec
}

impl<'a, T: AFloat> DataFrame<'a, T> {
    /// # Safety
    ///
    /// Please ensure that the `rv` represents bytes returned from the [`DataFrame::to_bytes`] method.
    pub unsafe fn from_opendal(rv: &mut Buffer) -> Self {
        let index_nbytes = rv.get_i64() as usize;
        let columns_nbytes = rv.get_i64() as usize;

        let index_shape = index_nbytes / 8;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;

        let index_array = Array1::<IndexDtype>::from_shape_vec(
            (index_shape,),
            extract_vec::<IndexDtype>(rv, index_nbytes),
        )
        .unwrap();
        let columns_array = Array1::<ColumnsDtype>::from_shape_vec(
            (columns_shape,),
            extract_vec::<ColumnsDtype>(rv, columns_nbytes),
        )
        .unwrap();
        let values_nbytes = rv.len();
        let values_array = Array2::<T>::from_shape_vec(
            (index_shape, columns_shape),
            extract_vec::<T>(rv, values_nbytes),
        )
        .unwrap();

        DataFrame::new(
            index_array.into(),
            columns_array.into(),
            values_array.into(),
        )
    }
}
