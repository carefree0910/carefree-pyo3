from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f32
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f64

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


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


def fast_concat_dfs_axis0(
    dfs: List["pd.DataFrame"],
    *,
    columns: Optional["pd.Index"] = None,
    to_fp32: bool = False,
) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    if not to_fp32:
        values = [d.values for d in dfs]
    else:
        values = [d.values.astype(np.float32, copy=False) for d in dfs]
    values = fast_concat_2d_axis0(values)
    indexes = np.concatenate([d.index for d in dfs])
    if columns is None:
        columns = dfs[0].columns
    return pd.DataFrame(values, index=indexes, columns=columns, copy=False)


__all__ = [
    "fast_concat_2d_axis0",
    "fast_concat_dfs_axis0",
]
