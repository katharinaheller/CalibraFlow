from pathlib import Path
import logging
from typing import Set
import polars as pl

from core.interfaces.IDataLoader import IDataLoader
from .dataset_ids import DatasetId
from .dataset_config import DATASET_REGISTRY, DatasetConfig

logger = logging.getLogger(__name__)

class CsvDatasetLoader(IDataLoader):
    # single responsibility: reading datasets using dataset-specific configs
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def load_dataset(self, dataset_id: DatasetId) -> pl.DataFrame:
        config: DatasetConfig = self._get_config(dataset_id)
        file_path = self._base_path / config.relative_path

        logger.debug("Loading dataset '%s' from '%s'", dataset_id.value, file_path)

        if not file_path.exists():
            logger.error("Dataset file not found: %s", file_path)
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        scan = pl.scan_csv(
            file_path,
            has_header=config.has_header,
            separator=config.delimiter,
            encoding=config.encoding,
            null_values=config.null_values,
        )

        schema = scan.collect_schema()
        available_columns: Set[str] = set(schema.names())

        for col in config.parse_dates:
            if col in available_columns:
                scan = scan.with_columns(
                    pl.col(col).str.strptime(pl.Datetime, strict=False)
                )
                logger.debug("Parsed datetime column '%s' for dataset '%s'",
                             col, dataset_id.value)
            else:
                logger.warning("Configured datetime column '%s' not found in dataset '%s'",
                               col, dataset_id.value)

        df = scan.collect()

        if config.rename_columns:
            df = df.rename(config.rename_columns)
            logger.debug("Applied column renames for dataset '%s': %s",
                         dataset_id.value, config.rename_columns)

        missing = set(config.required_columns) - set(df.columns)
        if missing:
            logger.error("Missing required columns for dataset '%s': %s",
                         dataset_id.value, sorted(missing))
            raise ValueError(f"Missing required columns for {dataset_id.value}: {sorted(missing)}")

        if config.dtypes:
            for col_name, dtype in config.dtypes.items():
                if col_name in df.columns:
                    df = df.with_columns(pl.col(col_name).cast(dtype))
                    logger.debug("Cast column '%s' to dtype '%s' for dataset '%s'",
                                 col_name, dtype, dataset_id.value)
                else:
                    logger.warning(
                        "Configured dtype for column '%s', but column not present in dataset '%s'",
                        col_name, dataset_id.value,
                    )

        logger.info(
            "Loaded dataset '%s' (rows=%s, cols=%s)",
            dataset_id.value, df.height, df.width,
        )

        return df

    def _get_config(self, dataset_id: DatasetId) -> DatasetConfig:
        try:
            return DATASET_REGISTRY[dataset_id]
        except KeyError as exc:
            logger.error("No DatasetConfig registered for id=%s", dataset_id.value)
            raise KeyError(f"No DatasetConfig registered for id={dataset_id.value}") from exc
