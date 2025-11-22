import logging
from typing import Dict, Sequence, Tuple
import polars as pl
from core.interfaces.IAnomalyDetector import IAnomalyDetector

logger = logging.getLogger(__name__)

_NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

def _is_numeric_dtype(dtype) -> bool:
    return dtype in _NUMERIC_DTYPES


class ZScoreDetector(IAnomalyDetector):

    def __init__(self) -> None:
        self._stats: Dict[str, Tuple[float, float]] = {}
        self._feature_columns: Sequence[str] = []

    def fit(self, df: pl.DataFrame, feature_columns: Sequence[str]) -> None:
        if df.is_empty():
            raise ValueError("Cannot fit ZScoreDetector on empty dataframe")

        self._feature_columns = list(feature_columns)
        n = df.height

        for col in self._feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not in dataframe")

            series = df[col]

            if not _is_numeric_dtype(series.dtype):
                raise TypeError(f"Feature column '{col}' must be numeric, got {series.dtype}")

            mean_val = float(series.mean())

            # stabilization for small sample sizes
            if n < 3:
                # minimal variance assumption → treat extreme values as anomalous
                std_val = 1.0
            else:
                std_val = float(series.std())
                if std_val == 0.0 or std_val != std_val:
                    std_val = 1.0

            self._stats[col] = (mean_val, std_val)

        logger.info("Fitted ZScoreDetector on columns: %s", self._feature_columns)

    def score(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._stats:
            raise RuntimeError("ZScoreDetector must be fitted before calling 'score'")

        working = df.clone()
        z_expressions = []

        for col in self._feature_columns:
            mean_val, std_val = self._stats[col]

            z_expr = (
                (pl.col(col) - mean_val)
                .cast(pl.Float64)
                .truediv(std_val)
                .alias(f"z_{col}")
            )
            z_expressions.append(z_expr)

        working = working.with_columns(z_expressions)

        abs_cols = [f"z_{c}" for c in self._feature_columns]

        working = working.with_columns(
            pl.concat_list([pl.col(c).abs() for c in abs_cols])
            .list.mean()
            .alias("anomaly_score")
        )

        return working

    def detect(self, df: pl.DataFrame, threshold: float) -> pl.DataFrame:
        scored = self.score(df)
        scored = scored.with_columns(
            (pl.col("anomaly_score") >= threshold).alias("is_anomaly")
        )
        return scored
