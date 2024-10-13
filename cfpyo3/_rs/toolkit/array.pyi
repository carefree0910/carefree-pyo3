import numpy as np

from typing import List
from typing import Tuple
from typing import Optional

def sum_axis1_f32(a: np.ndarray, num_threads: int = 8) -> np.ndarray: ...
def sum_axis1_f64(a: np.ndarray, num_threads: int = 8) -> np.ndarray: ...
def mean_axis1_f32(a: np.ndarray, num_threads: int = 8) -> np.ndarray: ...
def mean_axis1_f64(a: np.ndarray, num_threads: int = 8) -> np.ndarray: ...
def nanmean_axis1_f32(a: np.ndarray, num_threads: int = 8) -> np.ndarray: ...
def nanmean_axis1_f64(a: np.ndarray, num_threads: int = 8) -> np.ndarray: ...
def masked_mean_axis1_f32(
    a: np.ndarray,
    valid_mask: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def masked_mean_axis1_f64(
    a: np.ndarray,
    valid_mask: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def corr_axis1_f32(
    a: np.ndarray,
    b: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def corr_axis1_f64(
    a: np.ndarray,
    b: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def nancorr_axis1_f32(
    a: np.ndarray,
    b: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def nancorr_axis1_f64(
    a: np.ndarray,
    b: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def masked_corr_axis1_f32(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def masked_corr_axis1_f64(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: np.ndarray,
    num_threads: int = 8,
) -> np.ndarray: ...
def coeff_axis1_f32(
    x: np.ndarray,
    y: np.ndarray,
    q: Optional[float] = None,
    num_threads: int = 8,
) -> Tuple[np.ndarray, np.ndarray]: ...
def coeff_axis1_f64(
    x: np.ndarray,
    y: np.ndarray,
    q: Optional[float] = None,
    num_threads: int = 8,
) -> Tuple[np.ndarray, np.ndarray]: ...
def masked_coeff_axis1_f32(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    q: Optional[float] = None,
    num_threads: int = 8,
) -> Tuple[np.ndarray, np.ndarray]: ...
def masked_coeff_axis1_f64(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    q: Optional[float] = None,
    num_threads: int = 8,
) -> Tuple[np.ndarray, np.ndarray]: ...
def fast_concat_2d_axis0_f32(arrays: List[np.ndarray]) -> np.ndarray: ...
def fast_concat_2d_axis0_f64(arrays: List[np.ndarray]) -> np.ndarray: ...
