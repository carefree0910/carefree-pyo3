import numpy as np

from typing import Tuple

class DataFrameF64:
    index: np.ndarray
    columns: np.ndarray
    data: np.ndarray

    @staticmethod
    def new(
        index: np.ndarray,
        columns: np.ndarray,
        data: np.ndarray,
    ) -> DataFrameF64: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    def rows(self, indices: np.ndarray) -> DataFrameF64: ...

def index(df: DataFrameF64) -> np.ndarray: ...
def columns(df: DataFrameF64) -> np.ndarray: ...
def values(df: DataFrameF64) -> np.ndarray: ...
