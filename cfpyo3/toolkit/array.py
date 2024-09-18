from typing import Any
from typing import List
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
from cfpyo3._rs.toolkit.array import mean_axis1_f32
from cfpyo3._rs.toolkit.array import mean_axis1_f64
from cfpyo3._rs.toolkit.array import corr_axis1_f32
from cfpyo3._rs.toolkit.array import corr_axis1_f64
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f32
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f64

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def _dispatch(
    name: str,
    f32_fn: Callable,
    f64_fn: Callable,
    pivot: "np.ndarray",
    *args: Any,
    **kwargs: Any,
) -> "np.ndarray":
    import numpy as np

    if pivot.dtype == np.float32:
        return f32_fn(*args, **kwargs)
    if pivot.dtype == np.float64:
        return f64_fn(*args, **kwargs)
    raise ValueError(f"`{name}` only supports `f32` & `f64`, '{pivot.dtype}' found")


def mean_axis1(array: "np.ndarray") -> "np.ndarray":
    return _dispatch("mean_axis1", mean_axis1_f32, mean_axis1_f64, array, array)


def corr_axis1(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    return _dispatch("corr_axis1", corr_axis1_f32, corr_axis1_f64, a, a, b)


def fast_concat_2d_axis0(arrays: List["np.ndarray"]) -> "np.ndarray":
    pivot = arrays[0]
    out = _dispatch(
        "fast_concat_2d_axis0",
        fast_concat_2d_axis0_f32,
        fast_concat_2d_axis0_f64,
        pivot,
        arrays,
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
    values = fast_concat_2d_axis0(values)  # type: ignore
    indexes = np.concatenate([d.index for d in dfs])
    if columns is None:
        columns = dfs[0].columns
    return pd.DataFrame(values, index=indexes, columns=columns, copy=False)


__all__ = [
    "mean_axis1",
    "corr_axis1",
    "fast_concat_2d_axis0",
    "fast_concat_dfs_axis0",
]
