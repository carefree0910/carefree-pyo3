from os import PathLike
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING

from cfpyo3._rs.df import COLUMNS_NBYTES
from cfpyo3._rs.df import DataFrameF64
from cfpyo3._rs.df import ArcDataFrameF64

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

TDF = Union[DataFrameF64, ArcDataFrameF64]
RHS = Union["np.ndarray", "pd.DataFrame", "DataFrame"]


def rhs_to_np(rhs: RHS) -> "np.ndarray":
    import pandas as pd

    if isinstance(rhs, pd.DataFrame):
        return rhs.values
    if isinstance(rhs, DataFrame):
        return rhs.py_df.values
    return rhs


class DataFrame:
    """
    A DataFrame which aims to efficiently process a specific type of data:
    - index: datetime64[ns]
    - columns: S{COLUMNS_NBYTES}
    - values: f64
    """

    def __init__(self, _df: TDF) -> None:
        self._df = _df

    def __sub__(self, other: RHS) -> "DataFrame":
        return DataFrame(self.py_df.with_data(self.py_df.values - rhs_to_np(other)))

    @property
    def py_df(self) -> "DataFrameF64":
        if isinstance(self._df, ArcDataFrameF64):
            self._df = self._df.to_py()
        return self._df

    @property
    def shape(self) -> Tuple[int, int]:
        return self.py_df.shape

    def rows(self, indices: "np.ndarray") -> "DataFrame":
        import numpy as np

        df = self.py_df
        index = np.ascontiguousarray(df.index[indices])
        values = np.ascontiguousarray(df.values[indices])
        return DataFrame(DataFrameF64.new(index, df.columns, values))

    def pow(self, exponent: float) -> "DataFrame":
        df = self.py_df
        return DataFrame(df.with_data(df.values**exponent))

    def mean_axis1(self) -> "np.ndarray":
        return self._df.mean_axis1()

    def corr_with_axis1(self, other: RHS) -> "np.ndarray":
        return self._df.corr_with_axis1(rhs_to_np(other))

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        df = self.py_df
        return pd.DataFrame(df.values, index=df.index, columns=df.columns, copy=False)

    @classmethod
    def from_pandas(cls, df: "pd.DataFrame") -> "DataFrame":
        import numpy as np

        index = np.require(df.index.values, "datetime64[ns]", "C")
        columns = np.require(df.columns.values, f"S{COLUMNS_NBYTES}", "C")
        values = np.require(df.values, np.float64, "C")
        return DataFrame(DataFrameF64.new(index, columns, values))

    def save(self, path: PathLike) -> None:
        self._df.save(str(path))

    @staticmethod
    def load(path: PathLike) -> "DataFrame":
        return DataFrame(ArcDataFrameF64.load(str(path)))


__all__ = [
    "DataFrame",
]
