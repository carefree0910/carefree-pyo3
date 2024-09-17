import numpy as np
import pandas as pd

from cfpyo3.df import DataFrame
from functools import lru_cache
from cfpyo3._rs.df import INDEX_CHAR_LEN


NUM_ROWS = 239
NUM_COLUMNS = 5000


@lru_cache
def get_pandas_df() -> pd.DataFrame:
    return pd.DataFrame(
        np.random.random([NUM_ROWS, NUM_COLUMNS]),
        index=np.arange(0, NUM_ROWS, dtype="datetime64[ns]"),
        columns=np.arange(NUM_COLUMNS).astype(f"S{INDEX_CHAR_LEN}"),
    )


def test_shape():
    df = DataFrame.from_pandas(get_pandas_df())
    assert df.shape == (NUM_ROWS, NUM_COLUMNS)


def test_rows():
    pandas_df = get_pandas_df()
    df = DataFrame.from_pandas(pandas_df)
    for _ in range(10):
        indices = np.random.choice(NUM_ROWS, 100, replace=False)
        df_rows = df.rows(indices).to_pandas()
        pandas_df_rows = pandas_df.iloc[indices]
        assert df_rows.equals(pandas_df_rows)
