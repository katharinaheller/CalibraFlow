import polars as pl
import logging

logger = logging.getLogger(__name__)

def parse_timestamp(expr: pl.Expr) -> pl.Expr:
    # best effort parsing for timestamps represented as strings or numeric epochs

    # step 1: try to parse ISO-like string timestamps (safe, strict=False)
    parsed_str = expr.str.strptime(pl.Datetime, strict=False)

    # step 2: try to interpret values as integer epochs
    # numeric_int ist entweder eine Ganzzahl oder null, aber wirft keine Fehler
    numeric_int = expr.cast(pl.Int64, strict=False)

    # epoch in millisekunden direkt zu datetime[ms]
    parsed_epoch_ms = numeric_int.cast(pl.Datetime("ms"), strict=False)

    # epoch in sekunden: erst in millisekunden hochskalieren, dann nach datetime[ms]
    parsed_epoch_s = numeric_int.mul(1000).cast(pl.Datetime("ms"), strict=False)

    # wenn der numerische Wert groß genug ist, behandeln wir ihn als ms, sonst als s
    numeric_parsed = (
        pl.when(numeric_int >= 1_000_000_000_000)
        .then(parsed_epoch_ms)
        .otherwise(parsed_epoch_s)
    )

    # finale Strategie:
    # 1) wenn String-Parsing geklappt hat, diesen Wert verwenden
    # 2) sonst numerische Epoch-Interpretation
    # fehlgeschlagene Werte bleiben null, um den Typ Datetime konsistent zu halten
    result = parsed_str.fill_null(numeric_parsed)

    return result
