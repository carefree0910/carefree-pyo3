import numpy as np
import pandas as pd

from functools import lru_cache
from cfpyo3.df import DataFrame
from cfpyo3._rs.df import INDEX_CHAR_LEN


NUM_ROWS = 239
NUM_COLUMNS = 5000


def np_to_df(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=np.arange(0, NUM_ROWS, dtype="datetime64[ns]"),
        columns=np.arange(NUM_COLUMNS).astype(f"S{INDEX_CHAR_LEN}"),
    )


@lru_cache
def get_pandas_df() -> pd.DataFrame:
    return np_to_df(np.random.random([NUM_ROWS, NUM_COLUMNS]))


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


def get_random_pandas_df() -> pd.DataFrame:
    x = np.random.random(NUM_ROWS * NUM_COLUMNS)
    mask = x <= 0.25
    x[mask] = np.nan
    x = x.reshape([NUM_ROWS, NUM_COLUMNS])
    return np_to_df(x)


def test_mse():
    for _ in range(3):
        df0 = get_random_pandas_df()
        df1 = get_random_pandas_df()
        df_rs0 = DataFrame.from_pandas(df0)
        mse0 = (df0 - df1).pow(2).mean(axis=1).values
        mse1 = (df_rs0 - df1).pow(2).mean_axis1()
        mse2 = (df_rs0 - df1.values).pow(2).mean_axis1()
        mse3 = (df_rs0 - DataFrame.from_pandas(df1)).pow(2).mean_axis1()
        np.testing.assert_allclose(mse0, mse1)
        np.testing.assert_allclose(mse0, mse2)
        np.testing.assert_allclose(mse0, mse3)


def test_corr():
    for _ in range(3):
        df0 = get_random_pandas_df()
        df1 = get_random_pandas_df()
        df_rs0 = DataFrame.from_pandas(df0)
        c0 = df0.corrwith(df1, axis=1).values
        c1 = df_rs0.corr_with_axis1(df1)
        c2 = df_rs0.corr_with_axis1(df1.values)
        c3 = df_rs0.corr_with_axis1(DataFrame.from_pandas(df1))
        np.testing.assert_allclose(c0, c1)
        np.testing.assert_allclose(c0, c2)
        np.testing.assert_allclose(c0, c3)
