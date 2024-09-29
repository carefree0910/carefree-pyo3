use crate::{
    df::{ColumnsDtype, DataFrame, IndexDtype, COLUMNS_NBYTES, INDEX_NBYTES},
    toolkit::{
        array::AFloat,
        convert::{from_bytes, to_bytes, to_nbytes},
    },
};
use std::{
    fs::File,
    io::{self, Read, Write},
};

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path)?;
        self.save_to(&mut file)
    }
    pub fn load(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        DataFrame::load_from(&mut file)
    }
    pub fn save_to(&self, file: &mut impl Write) -> io::Result<()> {
        let index = &self.index;
        let columns = &self.columns;
        let index_nbytes = to_nbytes::<IndexDtype>(index.len()) as i64;
        let columns_nbytes = to_nbytes::<ColumnsDtype>(columns.len()) as i64;
        file.write_all(&index_nbytes.to_le_bytes())?;
        file.write_all(&columns_nbytes.to_le_bytes())?;
        unsafe {
            file.write_all(to_bytes(index.as_slice().unwrap()))?;
            file.write_all(to_bytes(columns.as_slice().unwrap()))?;
            file.write_all(to_bytes(self.values.as_slice().unwrap()))?;
        }
        Ok(())
    }
    pub fn load_from(file: &mut impl Read) -> io::Result<Self> {
        let mut nbytes_buffer = [0u8; 8];
        file.read_exact(&mut nbytes_buffer)?;
        let index_nbytes = i64::from_le_bytes(nbytes_buffer) as usize;
        file.read_exact(&mut nbytes_buffer)?;
        let columns_nbytes = i64::from_le_bytes(nbytes_buffer) as usize;
        let index_shape = index_nbytes / INDEX_NBYTES;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;
        let values_nbytes = to_nbytes::<T>(index_shape * columns_shape);
        let mut index_buffer = vec![0u8; index_nbytes];
        let mut columns_buffer = vec![0u8; columns_nbytes];
        let mut values_buffer = vec![0u8; values_nbytes];
        file.read_exact(&mut index_buffer)?;
        file.read_exact(&mut columns_buffer)?;
        file.read_exact(&mut values_buffer)?;

        let (index, columns, values) = unsafe {
            (
                from_bytes(index_buffer),
                from_bytes(columns_buffer),
                from_bytes(values_buffer),
            )
        };
        DataFrame::from_owned(index, columns, values)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::df::io::bytes::tests::get_test_df;
    use tempfile::tempdir;

    #[test]
    fn test_fs_io() {
        let df = get_test_df();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("t0.cfdf");
        let mut file = File::create(&file_path).unwrap();
        df.save_to(&mut file).unwrap();
        let mut file = File::open(&file_path).unwrap();
        let loaded = DataFrame::<f32>::load_from(&mut file).unwrap();
        assert_eq!(df.index, loaded.index);
        assert_eq!(df.columns, loaded.columns);
        assert_eq!(df.values, loaded.values);
        drop(file);
        let file_path = dir.path().join("t1.cfdf");
        let file_path = file_path.to_str().unwrap();
        df.save(file_path).unwrap();
        let loaded = DataFrame::<f32>::load(file_path).unwrap();
        assert_eq!(df.index, loaded.index);
        assert_eq!(df.columns, loaded.columns);
        assert_eq!(df.values, loaded.values);
        dir.close().unwrap();
    }
}
