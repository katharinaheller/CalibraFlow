import polars as pl
from .base_preprocessor import BasePreprocessor
from core.preprocessing.utils.time_utils import parse_timestamp

class SyntheticAirQualityPreprocessor(BasePreprocessor):

    def _select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _resolve_timestamps(self, df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp" in df.columns:
            return df.with_columns(
                parse_timestamp(pl.col("timestamp")).alias("timestamp")
            )
        return df
