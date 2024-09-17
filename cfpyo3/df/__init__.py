import numpy as np
import pandas as pd

from typing import Tuple
from cfpyo3._rs.df import new
from cfpyo3._rs.df import rows
from cfpyo3._rs.df import shape
from cfpyo3._rs.df import index
from cfpyo3._rs.df import columns
from cfpyo3._rs.df import values
from cfpyo3._rs.df import DataFrameF64
from cfpyo3._rs.df import INDEX_CHAR_LEN


class DataFrame:
    """
    A DataFrame which aims to efficiently process a specific type of data:
    - index: datetime64[ns]
    - columns: S{INDEX_CHAR_LEN}
    - values: f64
    """

    def __init__(self, _df: DataFrameF64) -> None:
        self._df = _df

    @property
    def shape(self) -> Tuple[int, int]:
        return shape(self._df)

    def rows(self, indices: np.ndarray) -> "DataFrame":
        if indices.dtype != np.int64:
            indices = indices.astype(np.int64)
        return DataFrame(rows(self._df, indices))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            values(self._df),
            index=index(self._df),
            columns=columns(self._df),
            copy=False,
        )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "DataFrame":
        index = df.index.values
        columns = df.columns.values.astype(f"S{INDEX_CHAR_LEN}")
        values = df.values
        if index.dtype != "datetime64[ns]":
            index = index.astype("datetime64[ns]")
        if values.dtype != np.float64:
            values = values.astype(np.float64)
        return DataFrame(new(index, columns, values))


__all__ = [
    "DataFrame",
]
