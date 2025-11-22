import polars as pl
from core.preprocessing.base_preprocessor import BasePreprocessor

def test_all_steps_return_df():
    df = pl.DataFrame({"a": [1]})
    bp = BasePreprocessor()
    out = bp.preprocess(df)
    assert isinstance(out, pl.DataFrame)
