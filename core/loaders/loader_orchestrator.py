from typing import Dict
import logging

import polars as pl

from .csv_dataset_loader import CsvDatasetLoader
from .dataset_ids import DatasetId

logger = logging.getLogger(__name__)  # module level logger


class LoaderOrchestrator:
    # single responsibility: orchestrating which datasets to load together
    def __init__(self, dataset_loader: CsvDatasetLoader) -> None:
        # injects dataset loader dependency for better testability
        self._dataset_loader = dataset_loader

    def load_all_synthetic(self) -> Dict[DatasetId, pl.DataFrame]:
        # loads all synthetic datasets defined in the DatasetId enum
        result: Dict[DatasetId, pl.DataFrame] = {}

        for ds_id in DatasetId:
            logger.info("Loading synthetic dataset '%s'", ds_id.value)
            df = self._dataset_loader.load_dataset(ds_id)
            result[ds_id] = df
            logger.debug(
                "Dataset '%s' loaded into orchestrator with shape rows=%s, cols=%s",
                ds_id.value,
                df.height,
                df.width,
            )

        logger.info("Loaded %s synthetic datasets via orchestrator", len(result))
        return result
