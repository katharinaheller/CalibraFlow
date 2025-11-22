import polars as pl
from .base_preprocessor import BasePreprocessor
from core.preprocessing.utils.time_utils import parse_timestamp


class LUBWMinutePreprocessor(BasePreprocessor):

    def _resolve_timestamps(self, df: pl.DataFrame) -> pl.DataFrame:
        # prefer standard timestamp column
        if "timestamp" in df.columns:
            return df.with_columns(
                parse_timestamp(pl.col("timestamp")).alias("timestamp")
            )

        # LUBW minute dataset uses 'datetime'
        if "datetime" in df.columns:
            return df.with_columns(
                pl.col("datetime")
                .str.strptime(pl.Datetime, strict=False)
                .alias("timestamp")
            )

        # fallback: synthetic hour column
        if "Hour" in df.columns:
            return df.with_columns(
                parse_timestamp(pl.col("Hour")).alias("timestamp")
            )

        return df

    def _normalize_units(self, df: pl.DataFrame) -> pl.DataFrame:
        # ensure all numeric env columns are numeric
        numeric_cols = [
            c for c in df.columns
            if c not in ("timestamp", "datetime", "Hour", "flag")
        ]

        for col in numeric_cols:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        return df

    def _validate_ranges(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [
            c for c in df.columns
            if c not in ("timestamp", "datetime", "flag")
        ]

        return (
            df
            .with_columns([
                pl.col(numeric_cols).cast(pl.Float64, strict=False)
            ])
            .filter(
                pl.all_horizontal(
                    [pl.col(c).is_not_null() & (pl.col(c) >= 0) for c in numeric_cols]
                )
            )
        )
