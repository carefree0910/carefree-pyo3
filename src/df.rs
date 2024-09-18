use numpy::{
    datetime::{units::Nanoseconds, Datetime},
    PyFixedString,
};

pub const INDEX_CHAR_LEN: usize = 256;
pub type IndexDtype = Datetime<Nanoseconds>;
pub type ColumnsDtype = PyFixedString<INDEX_CHAR_LEN>;

pub mod frame;
