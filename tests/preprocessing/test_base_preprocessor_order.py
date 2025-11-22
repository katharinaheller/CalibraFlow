import polars as pl
from core.preprocessing.base_preprocessor import BasePreprocessor

def test_base_preprocessor_step_order():
    df = pl.DataFrame({"a": [1]})
    bp = BasePreprocessor()

    steps = [step.__name__ for step in bp.get_steps()]

    assert steps == [
        "_select_columns",
        "_resolve_timestamps",
        "_normalize_units",
        "_validate_ranges",
        "_handle_missing",
        "_finalize",
    ]
