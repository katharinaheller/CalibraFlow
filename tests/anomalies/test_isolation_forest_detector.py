import pytest
import polars as pl
from core.anomalies.IsolationForestDetector import IsolationForestDetector

def test_fit_on_numeric_data():
    df = pl.DataFrame({"a":[1.0,2.0,3.0]})
    det = IsolationForestDetector()
    det.fit(df, ["a"])
    assert det._feature_columns == ["a"]

def test_fit_rejects_empty():
    det = IsolationForestDetector()
    with pytest.raises(ValueError):
        det.fit(pl.DataFrame({}), ["a"])

def test_score_requires_fit():
    df = pl.DataFrame({"a":[1.0]})
    det = IsolationForestDetector()
    with pytest.raises(RuntimeError):
        det.score(df)

def test_score_produces_anomaly_score():
    df = pl.DataFrame({"a":[1.0,2.0,3.0]})
    det = IsolationForestDetector()
    det.fit(df, ["a"])
    out = det.score(df)
    assert "anomaly_score" in out.columns

def test_detect_adds_is_anomaly():
    df = pl.DataFrame({"a":[1.0,50.0]})
    det = IsolationForestDetector(contamination=0.1)
    det.fit(df, ["a"])
    out = det.detect(df, threshold=0.5)
    assert "is_anomaly" in out.columns
