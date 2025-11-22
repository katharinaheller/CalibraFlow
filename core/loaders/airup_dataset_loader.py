from pathlib import Path
import logging
from typing import List, Dict
import polars as pl

from core.interfaces.IDataLoader import IDataLoader
from .dataset_ids import DatasetId
from .dataset_config import DatasetConfig, DATASET_REGISTRY

logger = logging.getLogger(__name__)


class AirUpDatasetLoader(IDataLoader):
    # loader for AirUp sensor directories containing multiple log files

    def __init__(
        self,
        base_path: Path,
        registry: Dict[DatasetId, DatasetConfig] = DATASET_REGISTRY
    ) -> None:
        # store base path and injected registry
        self._base_path = base_path
        self._registry = registry

    def _get_config(self, dataset_id: DatasetId) -> DatasetConfig:
        # retrieve dataset config from injected registry
        try:
            return self._registry[dataset_id]
        except KeyError as exc:
            logger.error("No DatasetConfig registered for id=%s", dataset_id.value)
            raise KeyError(f"No DatasetConfig registered for id={dataset_id.value}") from exc

    def _resolve_pattern(self, dataset_id: DatasetId) -> str:
        # choose filename pattern based on dataset id
        if dataset_id == DatasetId.AIRUP_SONT_A:
            return "airup_sont_a_avg_every_minute_data.log.*"
        if dataset_id == DatasetId.AIRUP_SONT_C:
            return "airup_sont_c_avg_every_minute_data.log.*"
        raise ValueError(f"AirUpDatasetLoader does not support dataset_id={dataset_id.value}")

    def load_dataset(self, dataset_id: DatasetId) -> pl.DataFrame:
        # load and concatenate all AirUp log files for dataset id
        config = self._get_config(dataset_id)
        directory = self._base_path / config.relative_path

        logger.debug(
            "Loading AirUp dataset '%s' from directory '%s'",
            dataset_id.value,
            directory,
        )

        if not directory.exists():
            logger.error("AirUp dataset directory not found: %s", directory)
            raise FileNotFoundError(f"AirUp dataset directory not found: {directory}")

        if not directory.is_dir():
            logger.error("AirUp dataset path is not a directory: %s", directory)
            raise NotADirectoryError(f"AirUp dataset path is not a directory: {directory}")

        pattern = self._resolve_pattern(dataset_id)
        files: List[Path] = sorted(directory.glob(pattern))

        if not files:
            logger.error(
                "No AirUp log files found for dataset '%s' using pattern '%s' in '%s'",
                dataset_id.value,
                pattern,
                directory,
            )
            raise FileNotFoundError(
                f"No AirUp log files found for {dataset_id.value} in {directory} "
                f"with pattern '{pattern}'"
            )

        lazy_frames: List[pl.LazyFrame] = []

        for file_path in files:
            logger.debug(
                "Scanning AirUp log file '%s' for dataset '%s'",
                file_path,
                dataset_id.value
            )
            lf = pl.scan_csv(
                file_path,
                has_header=config.has_header,
                separator=config.delimiter,
                encoding=config.encoding,
                null_values=config.null_values,
            )
            lazy_frames.append(lf)

        scan = lazy_frames[0] if len(lazy_frames) == 1 else pl.concat(lazy_frames)
        df = scan.collect()

        if config.rename_columns:
            df = df.rename(config.rename_columns)
            logger.debug(
                "Applied column renames for AirUp dataset '%s': %s",
                dataset_id.value,
                config.rename_columns,
            )

        missing = set(config.required_columns) - set(df.columns)
        if missing:
            logger.error(
                "Missing required columns for AirUp dataset '%s': %s",
                dataset_id.value,
                sorted(missing),
            )
            raise ValueError(
                f"Missing required columns for AirUp dataset {dataset_id.value}: "
                f"{sorted(missing)}"
            )

        if config.dtypes:
            for col_name, dtype in config.dtypes.items():
                if col_name in df.columns:
                    df = df.with_columns(pl.col(col_name).cast(dtype))
                    logger.debug(
                        "Cast column '%s' to dtype '%s' for AirUp dataset '%s'",
                        col_name,
                        dtype,
                        dataset_id.value,
                    )
                else:
                    logger.warning(
                        "Configured dtype for column '%s' but column not present in AirUp dataset '%s'",
                        col_name,
                        dataset_id.value,
                    )

        logger.info(
            "Loaded AirUp dataset '%s' (files=%s, rows=%s, cols=%s)",
            dataset_id.value,
            len(files),
            df.height,
            df.width,
        )

        return df
