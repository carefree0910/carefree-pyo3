from typing import Any
from typing import List
from typing import Tuple
from typing import Generic
from typing import TypeVar
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING

from cfpyo3._rs.toolkit.array import sum_axis1_f32
from cfpyo3._rs.toolkit.array import sum_axis1_f64
from cfpyo3._rs.toolkit.array import mean_axis1_f32
from cfpyo3._rs.toolkit.array import mean_axis1_f64
from cfpyo3._rs.toolkit.array import nanmean_axis1_f32
from cfpyo3._rs.toolkit.array import nanmean_axis1_f64
from cfpyo3._rs.toolkit.array import masked_mean_axis1_f32
from cfpyo3._rs.toolkit.array import masked_mean_axis1_f64
from cfpyo3._rs.toolkit.array import corr_axis1_f32
from cfpyo3._rs.toolkit.array import corr_axis1_f64
from cfpyo3._rs.toolkit.array import nancorr_axis1_f32
from cfpyo3._rs.toolkit.array import nancorr_axis1_f64
from cfpyo3._rs.toolkit.array import masked_corr_axis1_f32
from cfpyo3._rs.toolkit.array import masked_corr_axis1_f64
from cfpyo3._rs.toolkit.array import coeff_axis1_f32
from cfpyo3._rs.toolkit.array import coeff_axis1_f64
from cfpyo3._rs.toolkit.array import masked_coeff_axis1_f32
from cfpyo3._rs.toolkit.array import masked_coeff_axis1_f64
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f32
from cfpyo3._rs.toolkit.array import fast_concat_2d_axis0_f64

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

TRes = TypeVar("TRes", covariant=True)


class Fn(Protocol, Generic[TRes]):
    def __call__(self, *args: Any, **kwargs: Any) -> TRes: ...


def _dispatch(
    name: str,
    f32_fn: Fn[TRes],
    f64_fn: Fn[TRes],
    pivot: "np.ndarray",
    *args: Any,
    **kwargs: Any,
) -> TRes:
    import numpy as np

    if pivot.dtype == np.float32:
        return f32_fn(*args, **kwargs)
    if pivot.dtype == np.float64:
        return f64_fn(*args, **kwargs)
    raise ValueError(f"`{name}` only supports `f32` & `f64`, '{pivot.dtype}' found")


def sum_axis1(array: "np.ndarray", num_threads: int = 0) -> "np.ndarray":
    return _dispatch(
        "sum_axis1",
        sum_axis1_f32,
        sum_axis1_f64,
        array,
        array,
        num_threads=num_threads,
    )


def mean_axis1(array: "np.ndarray", num_threads: int = 0) -> "np.ndarray":
    return _dispatch(
        "mean_axis1",
        mean_axis1_f32,
        mean_axis1_f64,
        array,
        array,
        num_threads=num_threads,
    )


def nanmean_axis1(array: "np.ndarray", num_threads: int = 4) -> "np.ndarray":
    return _dispatch(
        "nanmean_axis1",
        nanmean_axis1_f32,
        nanmean_axis1_f64,
        array,
        array,
        num_threads=num_threads,
    )


def masked_mean_axis1(
    array: "np.ndarray",
    mask: "np.ndarray",
    num_threads: int = 4,
) -> "np.ndarray":
    return _dispatch(
        "masked_mean_axis1",
        masked_mean_axis1_f32,
        masked_mean_axis1_f64,
        array,
        array,
        mask,
        num_threads=num_threads,
    )


def corr_axis1(a: "np.ndarray", b: "np.ndarray", num_threads: int = 0) -> "np.ndarray":
    return _dispatch(
        "corr_axis1",
        corr_axis1_f32,
        corr_axis1_f64,
        a,
        a,
        b,
        num_threads=num_threads,
    )


def nancorr_axis1(
    a: "np.ndarray", b: "np.ndarray", num_threads: int = 4
) -> "np.ndarray":
    return _dispatch(
        "nancorr_axis1",
        nancorr_axis1_f32,
        nancorr_axis1_f64,
        a,
        a,
        b,
        num_threads=num_threads,
    )


def masked_corr_axis1(
    a: "np.ndarray",
    b: "np.ndarray",
    mask: "np.ndarray",
    num_threads: int = 4,
) -> "np.ndarray":
    return _dispatch(
        "masked_corr_axis1",
        masked_corr_axis1_f32,
        masked_corr_axis1_f64,
        a,
        a,
        b,
        mask,
        num_threads=num_threads,
    )


def coeff_axis1(
    x: "np.ndarray",
    y: "np.ndarray",
    q: Optional[float] = None,
    num_threads: int = 8,
) -> Tuple["np.ndarray", "np.ndarray"]:
    return _dispatch(
        "coeff_axis1",
        coeff_axis1_f32,
        coeff_axis1_f64,
        x,
        x,
        y,
        q,
        num_threads=num_threads,
    )


def masked_coeff_axis1(
    x: "np.ndarray",
    y: "np.ndarray",
    mask: "np.ndarray",
    q: Optional[float] = None,
    num_threads: int = 8,
) -> Tuple["np.ndarray", "np.ndarray"]:
    return _dispatch(
        "masked_coeff_axis1",
        masked_coeff_axis1_f32,
        masked_coeff_axis1_f64,
        x,
        x,
        y,
        mask,
        q,
        num_threads=num_threads,
    )


def fast_concat_2d_axis0(arrays: List["np.ndarray"]) -> "np.ndarray":
    import numpy as np

    arrays = [np.ascontiguousarray(a) for a in arrays]
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
    "sum_axis1",
    "mean_axis1",
    "nanmean_axis1",
    "masked_mean_axis1",
    "corr_axis1",
    "nancorr_axis1",
    "masked_corr_axis1",
    "coeff_axis1",
    "masked_coeff_axis1",
    "fast_concat_2d_axis0",
    "fast_concat_dfs_axis0",
]
