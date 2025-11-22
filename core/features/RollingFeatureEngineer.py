import polars as pl
from typing import Sequence, List


class RollingFeatureEngineer:
    # Robust time-based rolling windows (Polars >= 1.20)

    def add_rolling_features(
        self,
        df: pl.DataFrame,
        feature_columns: Sequence[str],
        windows: Sequence[str],
    ) -> pl.DataFrame:

        if df.is_empty():
            return df

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column")

        # ensure timestamp is datetime
        working = df.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, strict=False)
        )

        exprs: List[pl.Expr] = []

        for col in feature_columns:
            if col not in working.columns:
                raise ValueError(f"Feature column '{col}' not in DataFrame")

            # require numeric dtype
            if working[col].dtype not in pl.NUMERIC_DTYPES:
                raise TypeError(f"Feature '{col}' must be numeric")

            for win in windows:
                # rolling returns list -> use .list.mean() / .list.std()
                roll = (
                    pl.col(col)
                    .rolling(index_column="timestamp", period=win, closed="right")
                )

                exprs.append(
                    roll.list.mean().alias(f"{col}_roll_mean_{win}")
                )
                exprs.append(
                    roll.list.std().alias(f"{col}_roll_std_{win}")
                )

        return working.with_columns(exprs)
