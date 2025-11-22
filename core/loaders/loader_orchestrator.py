from typing import Dict
import logging
import polars as pl

from core.interfaces.IDataLoader import IDataLoader
from .dataset_ids import DatasetId

logger = logging.getLogger(__name__)

class LoaderOrchestrator:
    # orchestrates loading of logical datasets via injected loader
    def __init__(self, dataset_loader: IDataLoader) -> None:
        self._dataset_loader = dataset_loader

    def load_all(self, dataset_ids: list[DatasetId]) -> Dict[DatasetId, pl.DataFrame]:
        result: Dict[DatasetId, pl.DataFrame] = {}
        for ds_id in dataset_ids:
            logger.info("Loading dataset '%s'", ds_id.value)
            df = self._dataset_loader.load_dataset(ds_id)
            result[ds_id] = df
            logger.debug(
                "Dataset '%s' loaded into orchestrator with rows=%s, cols=%s",
                ds_id.value, df.height, df.width,
            )
        logger.info("Loaded %s datasets via orchestrator", len(result))
        return result
