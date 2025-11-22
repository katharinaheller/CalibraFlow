import polars as pl
import logging
from .base_preprocessor import BasePreprocessor
from core.preprocessing.utils.time_utils import parse_timestamp

logger = logging.getLogger(__name__)


class AirUpSensorPreprocessor(BasePreprocessor):

    def _select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        keep_exact = {
            "pm1", "pm25", "pm10",
            "sht_humid", "sht_temp",
            "CO", "NO", "NO2", "O3",
            "timestamp_hr", "timestamp_gps", "timestamp",
            "lat", "lon", "alt",
        }
        selected = [c for c in df.columns if c in keep_exact]
        return df.select(selected)

    def _resolve_timestamps(self, df: pl.DataFrame) -> pl.DataFrame:
        # correct handling: epoch seconds -> local naive datetime
        if "timestamp_gps" in df.columns:
            return df.with_columns(
                (
                    pl.col("timestamp_gps")
                    .cast(pl.Float64)
                    .map_elements(lambda s: __import__("datetime").datetime.fromtimestamp(s))
                ).alias("timestamp")
            )

        # HR timestamps -> parse and keep naive
        if "timestamp_hr" in df.columns:
            return df.with_columns(
                parse_timestamp(pl.col("timestamp_hr").cast(pl.Utf8))
                .alias("timestamp")
            )

        # raw timestamp -> parse
        if "timestamp" in df.columns:
            return df.with_columns(
                parse_timestamp(pl.col("timestamp").cast(pl.Utf8))
                .alias("timestamp")
            )

        return df

    def _normalize_units(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {
            "sht_humid": "humidity",
            "sht_temp": "temperature",
        }
        return df.rename({k: v for k, v in rename_map.items() if k in df.columns})

    def _validate_ranges(self, df: pl.DataFrame) -> pl.DataFrame:
        for col in ["NO", "NO2", "O3", "CO"]:
            if col in df.columns:
                logger.warning(
                    "Column '%s' in AirUp dataset is marked as DO NOT USE per hackathon specification. "
                    "Setting values to null.",
                    col,
                )
                df = df.with_columns(pl.lit(None).alias(col))

        if "humidity" in df.columns:
            df = df.filter(
                (pl.col("humidity") >= 0) & (pl.col("humidity") <= 100)
            )

        if "temperature" in df.columns:
            df = df.filter(
                (pl.col("temperature") > -50) & (pl.col("temperature") < 80)
            )

        return df
