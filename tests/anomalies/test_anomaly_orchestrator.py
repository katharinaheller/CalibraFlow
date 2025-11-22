import polars as pl
from core.anomalies.AnomalyOrchestrator import AnomalyOrchestrator
from core.interfaces.IAnomalyDetector import IAnomalyDetector

class DummyDetector(IAnomalyDetector):
    def __init__(self):
        self.fitted = False

    def fit(self, df, feature_columns):
        self.fitted = True

    def score(self, df):
        return df.with_columns(pl.lit(0.5).alias("anomaly_score"))

    def detect(self, df, threshold):
        return df.with_columns(pl.lit(True).alias("is_anomaly"))

def test_fit_on_reference_calls_detector():
    dummy = DummyDetector()
    orch = AnomalyOrchestrator(dummy)
    df = pl.DataFrame({"x":[1]})
    orch.fit_on_reference(df, ["x"])
    assert dummy.fitted is True

def test_run_detection_returns_df_with_flag():
    dummy = DummyDetector()
    orch = AnomalyOrchestrator(dummy)
    df = pl.DataFrame({"x":[1]})
    out = orch.run_detection(df, threshold=0.1)
    assert "is_anomaly" in out.columns
