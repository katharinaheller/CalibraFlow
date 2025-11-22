import polars as pl
from .base_preprocessor import BasePreprocessor

class AirUpSensorPreprocessor(BasePreprocessor):
    def _select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # keep only meaningful columns for analytics
        keep = [
            c for c in df.columns 
            if not c.startswith(("RAW_OPC", "RAW_ADC", "Laser", "Heater", "Fan"))
        ]
        return df.select(keep)

    def _resolve_timestamps(self, df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp_gps" in df.columns and df["timestamp_gps"].dtype != pl.Utf8:
            return df.with_columns(
                pl.col("timestamp_gps").cast(pl.Datetime).alias("timestamp")
            )
        if "timestamp_hr" in df.columns:
            return df.with_columns(
                pl.col("timestamp_hr").str.strptime(pl.Datetime).alias("timestamp")
            )
        if "timestamp" in df.columns:
            return df.with_columns(
                pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime, "ms")
            )
        return df

    def _validate_ranges(self, df: pl.DataFrame) -> pl.DataFrame:
        rules = {
            "NO": (0, None),
            "NO2": (0, None),
            "O3": (0, None),
            "temperature": (-40, 80),
            "humidity": (0, 100),
        }
        for col, (min_v, max_v) in rules.items():
            if col in df.columns:
                if min_v is not None:
                    df = df.filter(pl.col(col) >= min_v)
                if max_v is not None:
                    df = df.filter(pl.col(col) <= max_v)
        return df
