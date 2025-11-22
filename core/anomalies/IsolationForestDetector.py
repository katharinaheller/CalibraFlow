import logging
from typing import Sequence
import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from core.interfaces.IAnomalyDetector import IAnomalyDetector

logger = logging.getLogger(__name__)

# numeric dtype set for polars
_NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

def _is_numeric_dtype(dtype) -> bool:
    return dtype in _NUMERIC_DTYPES


class IsolationForestDetector(IAnomalyDetector):
    # wrapper around sklearn IsolationForest for multivariate anomaly detection

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | None = 0.01,
        random_state: int | None = 42,
    ) -> None:
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._feature_columns: Sequence[str] = []

    def fit(self, df: pl.DataFrame, feature_columns: Sequence[str]) -> None:
        if df.is_empty():
            raise ValueError("Cannot fit IsolationForest on empty dataframe")

        self._feature_columns = list(feature_columns)

        for col in self._feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not in dataframe")

            if not _is_numeric_dtype(df[col].dtype):
                raise TypeError(f"Feature column '{col}' must be numeric")

        X = df.select(self._feature_columns).to_numpy()
        self._model.fit(X)

        logger.info(
            "Fitted IsolationForest on columns: %s (n_samples=%s)",
            self._feature_columns,
            X.shape[0],
        )

    def score(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._feature_columns:
            raise RuntimeError("IsolationForestDetector must be fitted before 'score'")

        for col in self._feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not present at scoring time")

        X = df.select(self._feature_columns).to_numpy()

        # sklearn: more negative = more anomalous
        raw_scores = self._model.score_samples(X)
        anomaly_score = -raw_scores

        anomaly_score = anomaly_score - float(np.min(anomaly_score))
        max_val = float(np.max(anomaly_score))
        if max_val > 0:
            anomaly_score = anomaly_score / max_val

        scored = df.with_columns(
            pl.Series("anomaly_score", anomaly_score.tolist())
        )

        return scored

    def detect(self, df: pl.DataFrame, threshold: float) -> pl.DataFrame:
        scored = self.score(df)

        scored = scored.with_columns(
            (pl.col("anomaly_score") >= threshold).alias("is_anomaly")
        )

        return scored
