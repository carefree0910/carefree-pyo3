use core::mem;

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

/// convert an arbitrary [`Vec`] into <T> values, in a complete zero-copy way
///
/// # Safety
///
/// The caller must ensure to check the [`Vec::from_raw_parts`] contract, and that
/// `vec` is a valid slice of `T`, which means:
/// - the underlying bytes are representing valid `T` values.
/// - `size_of::<U>() * vec.len()` is a multiple of `size_of::<T>()`.
#[inline]
pub unsafe fn from_vec<T: Sized, U>(vec: Vec<U>) -> Vec<T> {
    let values_len = mem::size_of::<U>() * vec.len() / mem::size_of::<T>();
    let results = unsafe { Vec::from_raw_parts(vec.as_ptr() as *mut T, values_len, values_len) };
    mem::forget(vec);
    results
}

/// convert bytes into <T> values, in a complete zero-copy way
///
/// # Safety
///
/// This is a shortcut of [`from_vec`] for [`Vec<u8>`], but it is more intuitive to use.
#[inline]
pub unsafe fn from_bytes<T: Sized>(bytes: Vec<u8>) -> Vec<T> {
    from_vec(bytes)
}

#[inline]
pub const fn to_nbytes<T: Sized>(values_len: usize) -> usize {
    values_len * mem::size_of::<T>()
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
