import numpy as np

from typing import Tuple
from typing import Generic
from typing import TypeVar

COLUMNS_NBYTES: int

TDF = TypeVar("TDF")

class IOs(Generic[TDF]):
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> TDF: ...

class Ops:
    def mean_axis1(self) -> np.ndarray: ...
    def corr_with_axis1(self, other: np.ndarray) -> np.ndarray: ...

class DataFrameF64(IOs[DataFrameF64], Ops):
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

class ArcDataFrameF64(IOs[ArcDataFrameF64], Ops):
    def to_py(self) -> DataFrameF64: ...
