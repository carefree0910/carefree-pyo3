use crate::toolkit::array::AFloat;

use super::{ColumnsDtype, IndexDtype};
use numpy::ndarray::{ArrayView1, ArrayView2};

mod ops;

pub struct DataFrame<'a, T: AFloat> {
    pub index: ArrayView1<'a, IndexDtype>,
    pub columns: ArrayView1<'a, ColumnsDtype>,
    pub data: ArrayView2<'a, T>,
}
