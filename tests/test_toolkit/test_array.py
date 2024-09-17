import random

import numpy as np

from typing import List
from cfpyo3.toolkit.array import fast_concat_2d_axis0


def generate_arrays(dtype: np.dtype) -> List[np.ndarray]:
    return [
        np.random.random([random.randint(10, 20), 100]).astype(dtype)
        for _ in range(100)
    ]


def test_fast_concat_2d_axis0_f32():
    for _ in range(10):
        arrays = generate_arrays(np.float32)
        np.testing.assert_allclose(
            np.concatenate(arrays, axis=0),
            fast_concat_2d_axis0(arrays),
        )


def test_fast_concat_2d_axis0_f64():
    for _ in range(10):
        arrays = generate_arrays(np.float64)
        np.testing.assert_allclose(
            np.concatenate(arrays, axis=0),
            fast_concat_2d_axis0(arrays),
        )
