import polars as pl
import logging

logger = logging.getLogger(__name__)

def parse_timestamp(expr: pl.Expr) -> pl.Expr:
    # attempt string → datetime parsing
    parsed_str = (
        expr.str.strptime(pl.Datetime, strict=False)
        .alias("ts_from_str")
    )

    # numeric epoch detection (seconds or ms)
    numeric = expr.cast(pl.Float64, strict=False)

    # safely convert numeric epoch → datetime(ms)
    # epoch_seconds < 10^11 ; epoch_milliseconds ≥ 10^11
    epoch_ms = (numeric.round() * 1_000).cast(pl.Datetime("ms"), strict=False)
    epoch_s = (numeric.round()).cast(pl.Datetime("ms"), strict=False)

    parsed_num = (
        pl.when(numeric.is_not_null() & (numeric > 10_000_000_000))
        .then(epoch_ms)
        .otherwise(epoch_s)
    )

    # priority: string → epoch
    return parsed_str.fill_null(parsed_num)
