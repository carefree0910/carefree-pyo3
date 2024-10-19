use super::{ColumnsDtype, DataFrame, IndexDtype, COLUMNS_NBYTES, INDEX_NBYTES};
use crate::{
    as_data_slice_or_err,
    toolkit::{
        array::AFloat,
        convert::{from_bytes, to_bytes, to_nbytes},
    },
};
use anyhow::Result;
use std::io::{Read, Write};

mod buffer;
mod bytes;
mod fs;

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn read(reader: &mut impl Read) -> Result<Self> {
        let mut nbytes_buffer = [0u8; 8];
        reader.read_exact(&mut nbytes_buffer)?;
        let index_nbytes = i64::from_le_bytes(nbytes_buffer) as usize;
        reader.read_exact(&mut nbytes_buffer)?;
        let columns_nbytes = i64::from_le_bytes(nbytes_buffer) as usize;
        let index_shape = index_nbytes / INDEX_NBYTES;
        let columns_shape = columns_nbytes / COLUMNS_NBYTES;
        let values_nbytes = to_nbytes::<T>(index_shape * columns_shape);
        let mut index_buffer = vec![0u8; index_nbytes];
        let mut columns_buffer = vec![0u8; columns_nbytes];
        let mut values_buffer = vec![0u8; values_nbytes];
        reader.read_exact(&mut index_buffer)?;
        reader.read_exact(&mut columns_buffer)?;
        reader.read_exact(&mut values_buffer)?;

        let (index, columns, values) = unsafe {
            (
                from_bytes(index_buffer),
                from_bytes(columns_buffer),
                from_bytes(values_buffer),
            )
        };
        DataFrame::from_vec(index, columns, values)
    }

    pub fn write(&self, writer: &mut impl Write) -> Result<()> {
        let index = self.index();
        let columns = self.columns();
        let index_nbytes = to_nbytes::<IndexDtype>(index.len()) as i64;
        let columns_nbytes = to_nbytes::<ColumnsDtype>(columns.len()) as i64;
        writer.write_all(&index_nbytes.to_le_bytes())?;
        writer.write_all(&columns_nbytes.to_le_bytes())?;
        unsafe {
            writer.write_all(to_bytes(as_data_slice_or_err!(index)))?;
            writer.write_all(to_bytes(as_data_slice_or_err!(columns)))?;
            writer.write_all(to_bytes(as_data_slice_or_err!(self.values())))?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::df::io::bytes::tests::get_test_df;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_io() {
        let df = get_test_df();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.cfdf");
        let mut file = File::create(&file_path).unwrap();
        df.write(&mut file).unwrap();
        let mut file = File::open(&file_path).unwrap();
        let loaded = DataFrame::<f32>::read(&mut file).unwrap();
        assert_eq!(df.index(), loaded.index());
        assert_eq!(df.columns(), loaded.columns());
        assert_eq!(df.values(), loaded.values());
        drop(file);
        dir.close().unwrap();
    }
}
