from typing import TYPE_CHECKING

from cfpyo3._rs.df import COLUMNS_NBYTES

if TYPE_CHECKING:
    import numpy as np


def to_index(array: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return np.require(array, "datetime64[ns]", "C")


def to_columns(array: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return np.require(array, f"S{COLUMNS_NBYTES}", "C")


def to_values(array: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return np.require(array, np.float64, "C")
