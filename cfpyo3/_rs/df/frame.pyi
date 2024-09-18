import numpy as np

from typing import Tuple

class DataFrameF64:
    @staticmethod
    def new(
        index: np.ndarray,
        columns: np.ndarray,
        data: np.ndarray,
    ) -> DataFrameF64: ...
    @property
    def index(self) -> np.ndarray: ...
    @property
    def columns(self) -> np.ndarray: ...
    @property
    def values(self) -> np.ndarray: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    def rows(self, indices: np.ndarray) -> DataFrameF64: ...
