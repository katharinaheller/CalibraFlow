import polars as pl

class TimeFeatureEngineer:
    # add derived time-based features to a dataframe containing a datetime column 'timestamp'

    def add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp" not in df.columns:
            raise ValueError("Expected column 'timestamp' in dataframe")

        # robust timestamp parsing
        working = df.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, strict=False)
            .alias("timestamp")
        )

        # season mapping helper
        def season_expr(col):
            return (
                pl.when(col.is_in([12, 1, 2])).then(0)  # winter
                .when(col.is_in([3, 4, 5])).then(1)     # spring
                .when(col.is_in([6, 7, 8])).then(2)     # summer
                .otherwise(3)                           # fall
            )

        # compute weekday in Python convention:
        # Monday=0, Sunday=6
        weekday_expr = (pl.col("timestamp").dt.weekday() - 1)

        working = working.with_columns([
            pl.col("timestamp").dt.hour().alias("hour"),
            weekday_expr.alias("day_of_week"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.year().alias("year"),
            (weekday_expr >= 5).alias("is_weekend"),  # Saturday(5), Sunday(6)
            season_expr(pl.col("timestamp").dt.month()).alias("season"),
        ])

        return working
