import random

import numpy as np
import pandas as pd

from typing import List
from cfpyo3.toolkit.array import corr_axis1
from cfpyo3.toolkit.array import mean_axis1
from cfpyo3.toolkit.array import fast_concat_2d_axis0
from cfpyo3.toolkit.array import fast_concat_dfs_axis0


def generate_array(dtype: np.dtype) -> np.ndarray:
    x = np.random.random(239 * 5000).astype(dtype)
    mask = x <= 0.25
    x[mask] = np.nan
    return x.reshape([239, 5000])


def generate_arrays(dtype: np.dtype) -> List[np.ndarray]:
    return [
        np.random.random([random.randint(10, 20), 100]).astype(dtype)
        for _ in range(100)
    ]


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid_mask = np.isfinite(a) & np.isfinite(b)
    a = a[valid_mask]
    b = b[valid_mask]
    return np.corrcoef(a, b)[0, 1]


def batch_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([corr(a[i], b[i]) for i in range(a.shape[0])])


def test_fast_concat_2d_axis0():
    for dtype in [np.float32, np.float64]:
        for _ in range(10):
            arrays = generate_arrays(dtype)
            np.testing.assert_allclose(
                np.concatenate(arrays, axis=0),
                fast_concat_2d_axis0(arrays),
            )
            dfs = [pd.DataFrame(a) for a in arrays]
            assert pd.concat(dfs).equals(fast_concat_dfs_axis0(dfs))


def test_mean_axis1():
    for dtype in [np.float32, np.float64]:
        for _ in range(3):
            a = generate_array(dtype)
            assert np.allclose(np.nanmean(a, axis=1), mean_axis1(a))


def test_corr_axis1():
    for dtype in [np.float32, np.float64]:
        for _ in range(3):
            a = generate_array(dtype)
            b = generate_array(dtype)
            assert np.allclose(
                batch_corr(a, b), corr_axis1(a, b), rtol=1.0e-3, atol=1.0e-3
            )
