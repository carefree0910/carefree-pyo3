use crate::toolkit::array::AFloat;

use super::{ColumnsDtype, IndexDtype};
use numpy::{ndarray::CowArray, Ix1, Ix2};

mod meta;
mod ops;

pub struct DataFrame<'a, T: AFloat> {
    pub index: CowArray<'a, IndexDtype, Ix1>,
    pub columns: CowArray<'a, ColumnsDtype, Ix1>,
    pub data: CowArray<'a, T, Ix2>,
}
