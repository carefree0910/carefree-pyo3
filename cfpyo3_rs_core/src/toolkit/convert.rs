use core::mem::size_of;

#[inline]
pub fn to_bytes<T: Sized>(values: &[T]) -> &[u8] {
    let nbytes = values.len() * size_of::<T>();
    unsafe { core::slice::from_raw_parts(values.as_ptr() as *mut u8, nbytes) }
}

#[inline]
pub fn from_bytes<T: Sized>(bytes: Vec<u8>) -> Vec<T> {
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
