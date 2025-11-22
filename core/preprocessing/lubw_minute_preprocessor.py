import polars as pl
from .base_preprocessor import BasePreprocessor
from core.preprocessing.utils.time_utils import parse_timestamp

class LUBWMinutePreprocessor(BasePreprocessor):

    def _resolve_timestamps(self, df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp" in df.columns:
            return df.with_columns(
                parse_timestamp(pl.col("timestamp")).alias("timestamp")
            )

        if "Hour" in df.columns:
            return df.with_columns(
                parse_timestamp(pl.col("Hour")).alias("timestamp")
            )

        return df

    def _validate_ranges(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_columns = [c for c in df.columns if c not in ("timestamp", "flag")]
        for col in numeric_columns:
            df = df.filter(pl.col(col).is_not_null())
            df = df.filter(pl.col(col) >= 0)
        return df
