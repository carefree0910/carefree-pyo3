import random

import numpy as np
import pandas as pd

from typing import List
from cfpyo3.toolkit.array import sum_axis1
from cfpyo3.toolkit.array import mean_axis1
from cfpyo3.toolkit.array import nanmean_axis1
from cfpyo3.toolkit.array import nancorr_axis1
from cfpyo3.toolkit.array import coeff_axis1
from cfpyo3.toolkit.array import masked_mean_axis1
from cfpyo3.toolkit.array import masked_corr_axis1
from cfpyo3.toolkit.array import masked_coeff_axis1
from cfpyo3.toolkit.array import fast_concat_2d_axis0
from cfpyo3.toolkit.array import fast_concat_dfs_axis0


def generate_array(dtype: np.dtype, *, no_nan: bool = False) -> np.ndarray:
    x = np.random.random(239 * 5000).astype(dtype)
    if not no_nan:
        mask = x <= 0.25
        x[mask] = np.nan
    return x.reshape([239, 5000])


def generate_arrays(dtype: np.dtype) -> List[np.ndarray]:
    return [
        np.random.random([random.randint(10, 20), 100]).astype(dtype)
        for _ in range(100)
    ]


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


def assert_allclose(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-3)


def test_mean_axis1():
    for dtype in [np.float32, np.float64]:
        a = generate_array(dtype)
        valid_mask = np.isfinite(a)
        assert_allclose(np.nanmean(a, axis=1), nanmean_axis1(a))
        assert_allclose(np.nanmean(a, axis=1), masked_mean_axis1(a, valid_mask))
        a = generate_array(dtype, no_nan=True)
        assert_allclose(np.sum(a, axis=1), sum_axis1(a))
        assert_allclose(np.mean(a, axis=1), mean_axis1(a))


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid_mask = np.isfinite(a) & np.isfinite(b)
    a = a[valid_mask]
    b = b[valid_mask]
    return np.corrcoef(a, b)[0, 1]


def batch_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([corr(a[i], b[i]) for i in range(a.shape[0])])


def test_corr_axis1():
    for dtype in [np.float32, np.float64]:
        for _ in range(3):
            a = generate_array(dtype)
            b = generate_array(dtype)
            valid_mask = np.isfinite(a) & np.isfinite(b)
            assert_allclose(batch_corr(a, b), nancorr_axis1(a, b))
            assert_allclose(batch_corr(a, b), masked_corr_axis1(a, b, valid_mask))


def lstsq(q: float, valid: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    coefficients = np.zeros([a0.shape[0], 2], a0.dtype)
    for t, t_labels in enumerate(a1):
        t_valid = valid[t]
        t_labels = t_labels[t_valid]
        t_signal = a0[t][t_valid]
        t_signal_med = np.median(t_signal)
        t_signal_mad = np.median(np.abs(t_signal - t_signal_med))
        t_signal = np.clip(
            t_signal,
            t_signal_med - 100.0 * t_signal_mad,
            t_signal_med + 100.0 * t_signal_mad,
        )
        t_signal = (t_signal - np.mean(t_signal)) / np.std(t_signal)
        if q > 0.0:
            floor_mask = t_signal <= np.quantile(t_signal, q)
            ceil_mask = t_signal >= np.quantile(t_signal, 1.0 - q)
            picked_mask = floor_mask | ceil_mask
            t_labels = t_labels[picked_mask]
            t_signal = t_signal[picked_mask]
        t_signal = np.column_stack((t_signal, np.ones_like(t_signal)))
        w, b = np.linalg.lstsq(t_signal, t_labels, rcond=None)[0]
        coefficients[t] = [w, b]
    return coefficients


def test_coeff_axis1():
    for dtype in [np.float32, np.float64]:
        for q in [0.0, 0.1]:
            a0 = generate_array(dtype)
            a1 = generate_array(dtype)
            valid_mask = np.isfinite(a0) & np.isfinite(a1)
            gt_w, gt_b = lstsq(q, valid_mask, a0, a1).T
            w, b = coeff_axis1(a0, a1, q)
            assert_allclose(gt_w, w)
            assert_allclose(gt_b, b)
            w, b = masked_coeff_axis1(a0, a1, valid_mask, q)
            assert_allclose(gt_w, w)
            assert_allclose(gt_b, b)
