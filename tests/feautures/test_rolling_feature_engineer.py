import polars as pl
from core.features.RollingFeatureEngineer import RollingFeatureEngineer

def test_rolling_mean_and_std_basic():
    # Test input: simple trend for deterministic windows
    df = pl.DataFrame({
        "timestamp": [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-01 02:00:00",
            "2023-01-01 03:00:00",
        ],
        "value": [10.0, 20.0, 30.0, 40.0],
    })

    fe = RollingFeatureEngineer()

    out = fe.add_rolling_features(
        df,
        feature_columns=["value"],
        windows=["2h"],  # rolling window of 2 hours
    )

    # ensure output columns exist
    assert "value_roll_mean_2h" in out.columns
    assert "value_roll_std_2h" in out.columns

    # check deterministic mean values
    # window includes timestamps >= current_time - 2h
    expected_mean = [
        10.0,      # only first value
        15.0,      # (10 + 20) / 2
        25.0,      # (20 + 30) / 2
        35.0,      # (30 + 40) / 2
    ]

    assert out["value_roll_mean_2h"].round(3).to_list() == expected_mean

def test_empty_dataframe_handling():
    fe = RollingFeatureEngineer()
    df = pl.DataFrame({"timestamp": [], "value": []})

    out = fe.add_rolling_features(df, ["value"], ["1h"])
    assert out.is_empty()

def test_multiple_feature_columns():
    df = pl.DataFrame({
        "timestamp": [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-01 02:00:00",
        ],
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })

    fe = RollingFeatureEngineer()
    out = fe.add_rolling_features(df, ["a", "b"], ["1h"])

    assert "a_roll_mean_1h" in out.columns
    assert "b_roll_mean_1h" in out.columns
    assert "a_roll_std_1h" in out.columns
    assert "b_roll_std_1h" in out.columns
