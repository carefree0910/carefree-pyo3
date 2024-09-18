from typing import List
from typing import TYPE_CHECKING
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f32
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f64

if TYPE_CHECKING:
    import numpy as np


def fast_concat_2d_axis0(arrays: List["np.ndarray"]) -> "np.ndarray":
    import numpy as np

    pivot = arrays[0]
    if pivot.dtype == np.float32:
        out = fast_concat_2d_axis0_f32(arrays)
    elif pivot.dtype == np.float64:
        out = fast_concat_2d_axis0_f64(arrays)
    else:
        raise ValueError(
            "`fast_concat_2d_axis0` only supports `f32` & `f64`, "
            f"'{pivot.dtype}' found"
        )
    return out.reshape([-1, pivot.shape[1]])


__all__ = [
    "fast_concat_2d_axis0",
]
