from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING
from cfpyo3._rs.df import INDEX_CHAR_LEN
from cfpyo3._rs.df.frame import DataFrameF64

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

RHS = Union["np.ndarray", "pd.DataFrame", "DataFrame"]


def rhs_to_np(rhs: RHS) -> "np.ndarray":
    import pandas as pd

    if isinstance(rhs, pd.DataFrame):
        return rhs.values
    if isinstance(rhs, DataFrame):
        return rhs._df.values
    return rhs


class DataFrame:
    """
    A DataFrame which aims to efficiently process a specific type of data:
    - index: datetime64[ns]
    - columns: S{INDEX_CHAR_LEN}
    - values: f64
    """

    def __init__(self, _df: DataFrameF64) -> None:
        self._df = _df

    def __sub__(self, other: RHS) -> "DataFrame":
        return DataFrame(self._df.with_data(self._df.values - rhs_to_np(other)))

    @property
    def shape(self) -> Tuple[int, int]:
        return self._df.shape

    def rows(self, indices: "np.ndarray") -> "DataFrame":
        import numpy as np

        data = np.ascontiguousarray(self._df.values[indices])
        index = np.ascontiguousarray(self._df.index[indices])
        return DataFrame(DataFrameF64.new(index, self._df.columns, data))

    def pow(self, exponent: float) -> "DataFrame":
        return DataFrame(self._df.with_data(self._df.values**exponent))

    def mean_axis1(self) -> "np.ndarray":
        return self._df.mean_axis1()

    def corr_with_axis1(self, other: RHS) -> "np.ndarray":
        return self._df.corr_with_axis1(rhs_to_np(other))

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(
            self._df.values,
            index=self._df.index,
            columns=self._df.columns,
            copy=False,
        )

    @classmethod
    def from_pandas(cls, df: "pd.DataFrame") -> "DataFrame":
        import numpy as np

        index = np.require(df.index.values, "datetime64[ns]", "C")
        columns = np.require(df.columns.values, f"S{INDEX_CHAR_LEN}", "C")
        values = np.require(df.values, np.float64, "C")
        return DataFrame(DataFrameF64.new(index, columns, values))


__all__ = [
    "DataFrame",
]
