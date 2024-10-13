from os import PathLike
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from typing import TYPE_CHECKING

from cfpyo3._rs.df import COLUMNS_NBYTES
from cfpyo3._rs.df import DataFrameF64
from cfpyo3._rs.df import OwnedDataFrameF64

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

TDF = Union[DataFrameF64, OwnedDataFrameF64]
RHS = Union["np.ndarray", "pd.DataFrame", "DataFrame"]


def rhs_to_np(rhs: RHS) -> "np.ndarray":
    import pandas as pd

    if isinstance(rhs, pd.DataFrame):
        return rhs.values
    if isinstance(rhs, DataFrame):
        return rhs.py_df.values
    return rhs


class DataFrameBindings(NamedTuple):
    """
    The `bindings` is useful iff you are trying to extend `cfpyo3_rs_core` /
    `cfpyo3_rs_bindings` with your own rust codes. In this case, you should
    re-export the `DataFrameF64` / `OwnedDataFrameF64` classes and set the `bindings`
    class attribute of `DataFrame` to your re-exported ones.

    The reason behind is that pyo3 will treat the same `[pyclass]` differently when
    it is added to different rust pymodules.

    For example, in this package, the `DataFrameF64` is binded to `cfpyo3` pymodule.
    If you reused the rust `DataFrameF64` in your own rust codes and binded them to
    your own pymodule, pyo3 will think your `DataFrameF64` is DIFFERENT from the one
    in `cfpyo3`, and therefore will reject your `DataFrameF64` to be passed to methods
    defined in `cfpyo3`, so `DataFrame` will not be usable.

    In this case, you can simply re-export the `DataFrameF64` in your own pymodule and
    set the `DataFrame.bindings` to your re-exported `DataFrameF64` class:

    ```python
    from cfpyo3.df import DataFrame
    from cfpyo3.df import DataFrameBindings
    from your_pymodule import DataFrameF64
    from your_pymodule import OwnedDataFrameF64

    DataFrame.bindings = DataFrameBindings(DataFrameF64, OwnedDataFrameF64)
    ```
    """

    py_cls: Type[DataFrameF64]
    rs_cls: Type[OwnedDataFrameF64]


class DataFrame:
    """
    A DataFrame which aims to efficiently process a specific type of data:
    - index: datetime64[ns]
    - columns: S{COLUMNS_NBYTES}
    - values: f64
    """

    bindings: DataFrameBindings = DataFrameBindings(DataFrameF64, OwnedDataFrameF64)

    def __init__(self, _df: TDF) -> None:
        self._df = _df

    def __sub__(self, other: RHS) -> "DataFrame":
        return DataFrame(self.py_df.with_data(self.py_df.values - rhs_to_np(other)))

    @property
    def py_df(self) -> "DataFrameF64":
        if self.is_owned:
            self._df = self._df.to_py()  # type: ignore
        return self._df  # type: ignore

    @property
    def rs_df(self) -> "OwnedDataFrameF64":
        if not self.is_owned:
            self._df = self._df.to_owned()  # type: ignore
        return self._df  # type: ignore

    @property
    def is_owned(self) -> bool:
        return isinstance(self._df, self.bindings.rs_cls)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.py_df.shape

    def rows(self, indices: "np.ndarray") -> "DataFrame":
        import numpy as np

        df = self.py_df
        index = np.ascontiguousarray(df.index[indices])
        values = np.ascontiguousarray(df.values[indices])
        return DataFrame(self.bindings.py_cls.new(index, df.columns, values))

    def pow(self, exponent: float) -> "DataFrame":
        df = self.py_df
        return DataFrame(df.with_data(df.values**exponent))

    def nanmean_axis1(self, num_threads: Optional[int] = None) -> "np.ndarray":
        return self._df.nanmean_axis1(num_threads)

    def nancorr_with_axis1(
        self,
        other: RHS,
        num_threads: Optional[int] = None,
    ) -> "np.ndarray":
        return self._df.nancorr_with_axis1(rhs_to_np(other), num_threads)

    def to_pandas(self) -> "pd.DataFrame":
        import pandas as pd

        df = self.py_df
        return pd.DataFrame(df.values, index=df.index, columns=df.columns, copy=False)

    @classmethod
    def to_py_df(cls, df: "pd.DataFrame") -> "DataFrameF64":
        import numpy as np

        index = np.require(df.index.values, "datetime64[ns]", "C")
        columns = np.require(df.columns.values, f"S{COLUMNS_NBYTES}", "C")
        values = np.require(df.values, np.float64, "C")
        return cls.bindings.py_cls.new(index, columns, values)

    @classmethod
    def from_pandas(cls, df: "pd.DataFrame") -> "DataFrame":
        return DataFrame(cls.to_py_df(df))

    def save(self, path: PathLike) -> None:
        self._df.save(str(path))

    @classmethod
    def load(cls, path: PathLike) -> "DataFrame":
        return DataFrame(cls.bindings.rs_cls.load(str(path)))

    def to_py(self) -> None:
        self.py_df

    def to_owned(self) -> None:
        self.rs_df


__all__ = [
    "DataFrame",
    "DataFrameBindings",
]
