import polars as pl
from core.preprocessing.lubw_minute_preprocessor import LUBWMinutePreprocessor

def test_lubw_negative_values_filtered():
    df = pl.DataFrame({
        "timestamp": ["2024-01-01 12:00:00"],
        "NO2": [-1],
        "PM10": [10],
    })

    pre = LUBWMinutePreprocessor()
    out = pre.preprocess(df)

    assert out.height == 0
