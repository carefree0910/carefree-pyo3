use core::mem::size_of;

/// get a bytes representation of <T> values, in a complete zero-copy way
///
/// # Safety
///
/// This is basically a shortcut of `slice::align_to`, but the caller must ensure
/// that `values` is 'exactly' a slice of `T`, because we only take the second part
/// of the tuple returned by `align_to` (that is, assuming the 'prefix' and 'suffix'
/// are both empty).
#[inline]
pub unsafe fn to_bytes<T: Sized>(values: &[T]) -> &[u8] {
    unsafe { values.align_to().1 }
}

/// convert bytes into <T> values, in a complete zero-copy way
///
/// # Safety
///
/// The caller must ensure to check the [`Vec::from_raw_parts`] contract, and that
/// `bytes` is a valid slice of `T`, which means:
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
pub const fn to_nbytes<T: Sized>(values_len: usize) -> usize {
    values_len * size_of::<T>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_bytes() {
        let values: Vec<i32> = vec![1, 2, 3, 4, 5];
        let bytes = unsafe { to_bytes(&values) };
        assert_eq!(
            bytes,
            &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0]
        );
    }

    #[test]
    fn test_from_bytes() {
        let bytes = vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0];
        let values: Vec<i32> = unsafe { from_bytes(bytes) };
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_to_nbytes() {
        assert_eq!(to_nbytes::<i32>(5), 20);
    }
}
