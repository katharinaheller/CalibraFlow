import polars as pl
from core.preprocessing.airup_sensor_preprocessor import AirUpSensorPreprocessor

def test_airup_timestamp_handling():
    df = pl.DataFrame({
        "timestamp_hr": ["2024-01-01 12:00:00"],
        "NO": [5]
    })

    pre = AirUpSensorPreprocessor()
    out = pre.preprocess(df)

    assert "timestamp" in out.columns
    assert out["timestamp"].dtype == pl.Datetime
