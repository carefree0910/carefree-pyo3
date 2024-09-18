from typing import Tuple
from typing import TYPE_CHECKING
from cfpyo3._rs.df import INDEX_CHAR_LEN
from cfpyo3._rs.df.frame import DataFrameF64

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


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
        return self._df.shape

    def rows(self, indices: "np.ndarray") -> "DataFrame":
        import numpy as np

        if indices.dtype != np.int64:
            indices = indices.astype(np.int64)
        return DataFrame(self._df.rows(indices))

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

        index = df.index.values
        columns = df.columns.values.astype(f"S{INDEX_CHAR_LEN}")
        values = df.values
        if index.dtype != "datetime64[ns]":
            index = index.astype("datetime64[ns]")
        if values.dtype != np.float64:
            values = values.astype(np.float64)
        return DataFrame(DataFrameF64.new(index, columns, values))


__all__ = [
    "DataFrame",
]
