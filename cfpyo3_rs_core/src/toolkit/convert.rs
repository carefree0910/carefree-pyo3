use core::mem::size_of;

/// get a bytes representation of <T> values, in a complete zero-copy way
///
/// # Safety
///
/// The caller must ensure that the `values` lives longer than the returned
/// bytes representation, otherwise it will cause undefined behavior.
#[inline]
pub unsafe fn to_bytes<T: Sized>(values: &[T]) -> &[u8] {
    let nbytes = values.len() * size_of::<T>();
    unsafe { core::slice::from_raw_parts(values.as_ptr() as *mut u8, nbytes) }
}

/// convert bytes into <T> values, in a complete zero-copy way
///
/// # Safety
///
/// The caller must ensure that `bytes` is a valid slice of `T`, that is
/// - the bytes are representing valid `T` values
/// - the length of `bytes` is a multiple of the size of `T`
#[inline]
pub unsafe fn from_bytes<T: Sized>(bytes: Vec<u8>) -> Vec<T> {
    let values_len = bytes.len() / size_of::<T>();
    unsafe {
        let vec: Vec<T> = Vec::from_raw_parts(bytes.as_ptr() as *mut T, values_len, values_len);
        core::mem::forget(bytes);
        vec
    }
}

#[inline]
pub fn to_nbytes<T: Sized>(values_len: usize) -> usize {
    values_len * size_of::<T>()
}
