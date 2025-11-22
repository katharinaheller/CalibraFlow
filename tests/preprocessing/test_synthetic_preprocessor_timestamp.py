import polars as pl
from core.preprocessing.synthetic_air_quality_preprocessor import SyntheticAirQualityPreprocessor

def test_synthetic_timestamp_cast():
    df = pl.DataFrame({
        "timestamp": ["2021-01-01 00:00:00"],
        "station_id": ["X1"],
    })

    pre = SyntheticAirQualityPreprocessor()
    out = pre.preprocess(df)

    assert out["timestamp"].dtype == pl.Datetime
    assert out.height == 1
