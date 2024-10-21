import numpy as np

from typing import Tuple
from typing import Optional

COLUMNS_NBYTES: int

class DataFrameF64:
    # meta
    @staticmethod
    def new(
        index: np.ndarray,
        columns: np.ndarray,
        values: np.ndarray,
    ) -> DataFrameF64: ...
    @property
    def index(self) -> np.ndarray: ...
    @property
    def columns(self) -> np.ndarray: ...
    @property
    def values(self) -> np.ndarray: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    def with_data(self, values: np.ndarray) -> DataFrameF64: ...
    # io
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> DataFrameF64: ...
    # ops
    def nanmean_axis1(self, num_threads: Optional[int]) -> np.ndarray: ...
    def nancorr_with_axis1(
        self,
        other: np.ndarray,
        num_threads: Optional[int],
    ) -> np.ndarray: ...
