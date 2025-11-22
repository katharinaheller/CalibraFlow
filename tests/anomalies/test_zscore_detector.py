import pytest
import polars as pl
from core.anomalies.ZScoreDetector import ZScoreDetector

def test_fit_fails_on_empty_df():
    det = ZScoreDetector()
    with pytest.raises(ValueError):
        det.fit(pl.DataFrame({}), ["a"])

def test_fit_stores_stats():
    df = pl.DataFrame({"a":[1.0,3.0,5.0]})
    det = ZScoreDetector()
    det.fit(df, ["a"])
    assert "a" in det._stats
    mean, std = det._stats["a"]
    assert pytest.approx(mean, rel=1e-6) == 3.0
    assert std > 0

def test_score_requires_fit():
    df = pl.DataFrame({"a":[1.0]})
    det = ZScoreDetector()
    with pytest.raises(RuntimeError):
        det.score(df)

def test_score_produces_z_and_anomaly_score():
    df = pl.DataFrame({"a":[1.0,2.0,3.0]})
    det = ZScoreDetector()
    det.fit(df, ["a"])
    out = det.score(df)
    assert "z_a" in out.columns
    assert "anomaly_score" in out.columns

def test_detect_flags_anomalies():
    df = pl.DataFrame({"a":[1.0,100.0]})
    det = ZScoreDetector()
    det.fit(df, ["a"])
    out = det.detect(df, threshold=2.0)
    assert "is_anomaly" in out.columns
    assert out["is_anomaly"][1] is True
