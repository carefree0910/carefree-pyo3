use super::{ColumnsDtype, IndexDtype};
use crate::toolkit::array::AFloat;
use numpy::{ndarray::CowArray, Ix1, Ix2};

mod io;
mod meta;
mod ops;

#[derive(Debug)]
pub struct DataFrame<'a, T: AFloat> {
    pub index: CowArray<'a, IndexDtype, Ix1>,
    pub columns: CowArray<'a, ColumnsDtype, Ix1>,
    pub values: CowArray<'a, T, Ix2>,
}
