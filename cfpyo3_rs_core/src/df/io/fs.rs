use crate::{df::DataFrame, toolkit::array::AFloat};
use std::{
    fs::{self, File},
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
    pub fn save_to(&self, file: &mut fs::File) -> io::Result<()> {
        let bytes = self.to_bytes();
        file.write_all(&bytes)
    }
    pub fn load_from(file: &mut fs::File) -> io::Result<Self> {
        // `read_to_end` will try to reserve appropriate capacity for the buffer
        // internally, so we only need to new an empty Vec.
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(unsafe { DataFrame::from_buffer(bytes.as_slice()) })
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
