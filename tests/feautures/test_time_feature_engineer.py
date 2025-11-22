import polars as pl
from core.features.TimeFeatureEngineer import TimeFeatureEngineer

def test_basic_time_features():
    df = pl.DataFrame({
        "timestamp": [
            "2023-01-01 00:30:00",
            "2023-07-10 15:45:00",
        ]
    })

    fe = TimeFeatureEngineer()
    out = fe.add_time_features(df)

    # basic assertions
    assert "hour" in out.columns
    assert "day_of_week" in out.columns
    assert "month" in out.columns
    assert "year" in out.columns
    assert "is_weekend" in out.columns
    assert "season" in out.columns

    # check correctness
    assert out["hour"].to_list() == [0, 15]
    assert out["day_of_week"].to_list() == [6, 0]  # Jan 1 2023 = Sunday(6), Jul 10 2023 = Monday(0)
    assert out["month"].to_list() == [1, 7]
    assert out["year"].to_list() == [2023, 2023]

    # weekend flag
    assert out["is_weekend"].to_list() == [True, False]

    # seasons: winter=0, spring=1, summer=2, fall=3
    assert out["season"].to_list() == [0, 2]  # January, July
