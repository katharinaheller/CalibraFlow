import logging
from typing import Callable, List
import polars as pl
from core.interfaces.IDataPreprocessor import IDataPreprocessor

logger = logging.getLogger(__name__)

class BasePreprocessor(IDataPreprocessor):
    def __init__(self) -> None:
        self._steps: List[Callable[[pl.DataFrame], pl.DataFrame]] = [
            self._select_columns,
            self._resolve_timestamps,
            self._normalize_units,
            self._validate_ranges,
            self._handle_missing,
            self._finalize,
        ]

    def get_steps(self) -> List[Callable[[pl.DataFrame], pl.DataFrame]]:
        return list(self._steps)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        if df is None:
            raise ValueError("Input dataframe cannot be None")

        current = df

        for step in self._steps:
            step_name = step.__name__
            logger.debug("Entering step '%s'", step_name)

            next_df = step(current)

            if next_df is None:
                raise RuntimeError(
                    f"Preprocessing step '{step_name}' returned None. "
                    "Each step must return a polars DataFrame."
                )

            if not isinstance(next_df, pl.DataFrame):
                raise TypeError(
                    f"Step '{step_name}' must return a pl.DataFrame, "
                    f"but returned {type(next_df)}"
                )

            logger.debug(
                "Exiting step '%s' (rows=%s, cols=%s)",
                step_name,
                next_df.height,
                next_df.width,
            )

            current = next_df

        return current

    def _select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _resolve_timestamps(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _normalize_units(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _validate_ranges(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _handle_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _finalize(self, df: pl.DataFrame) -> pl.DataFrame:
        return df
