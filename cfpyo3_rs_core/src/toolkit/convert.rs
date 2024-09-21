pub fn to_bytes<T: Sized>(bytes: &[T]) -> &[u8] {
    let len_u8 = bytes.len() * core::mem::size_of::<T>();
    unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *mut u8, len_u8) }
}

pub fn from_bytes<T: Sized>(bytes: Vec<u8>) -> Vec<T> {
    let len = bytes.len() / core::mem::size_of::<T>();
    unsafe {
        let vec: Vec<T> = Vec::from_raw_parts(bytes.as_ptr() as *mut T, len, len);
        core::mem::forget(bytes);
        vec
    }
}
