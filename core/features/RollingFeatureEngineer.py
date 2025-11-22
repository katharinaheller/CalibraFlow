import polars as pl
from typing import List, Sequence

class RollingFeatureEngineer:
    # adds rolling mean and std features based on time windows

    def add_rolling_features(
        self,
        df: pl.DataFrame,
        feature_columns: Sequence[str],
        windows: Sequence[str],
    ) -> pl.DataFrame:
        # return early if empty
        if df.is_empty():
            return df

        # ensure timestamp exists
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column")

        # safely parse timestamp column
        working = df.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, strict=False)
        )

        # list of expressions for rolling statistics
        exprs: List[pl.Expr] = []

        for col in feature_columns:
            if col not in working.columns:
                raise ValueError(f"Feature column '{col}' not in DataFrame")

            # robust numeric check, compatible with all polars versions
            if working[col].dtype not in pl.NUMERIC_DTYPES:
                raise TypeError(f"Feature column '{col}' must be numeric")

            for win in windows:
                # rolling mean
                exprs.append(
                    pl.col(col)
                    .rolling_mean(
                        window=win,
                        min_periods=1,
                        by="timestamp",
                    )
                    .alias(f"{col}_roll_mean_{win}")
                )

                # rolling std
                exprs.append(
                    pl.col(col)
                    .rolling_std(
                        window=win,
                        min_periods=1,
                        by="timestamp",
                    )
                    .alias(f"{col}_roll_std_{win}")
                )

        # attach generated rolling features
        working = working.with_columns(exprs)

        return working
