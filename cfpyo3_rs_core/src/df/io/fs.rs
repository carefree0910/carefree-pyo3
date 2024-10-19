use crate::{df::DataFrame, toolkit::array::AFloat};
use anyhow::Result;
use std::fs::File;

impl<'a, T: AFloat> DataFrame<'a, T> {
    pub fn save(&self, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        self.write(&mut file)
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        DataFrame::read(&mut file)
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
        let file_path = dir.path().join("test.cfdf");
        let file_path = file_path.to_str().unwrap();
        df.save(file_path).unwrap();
        let loaded = DataFrame::<f32>::load(file_path).unwrap();
        assert_eq!(df.index(), loaded.index());
        assert_eq!(df.columns(), loaded.columns());
        assert_eq!(df.values(), loaded.values());
        dir.close().unwrap();
    }
}
