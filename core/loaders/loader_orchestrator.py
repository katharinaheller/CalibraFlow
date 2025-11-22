from typing import Dict, Optional
import logging
import polars as pl

from core.interfaces.IDataLoader import IDataLoader
from .dataset_ids import DatasetId

logger = logging.getLogger(__name__)


class LoaderOrchestrator:
    # coordinates loading of all datasets with override capability

    def __init__(
        self,
        default_loader: Optional[IDataLoader] = None,
        loader_overrides: Optional[Dict[DatasetId, IDataLoader]] = None,
        dataset_loader: Optional[IDataLoader] = None
    ) -> None:
        resolved_default = default_loader or dataset_loader
        if resolved_default is None:
            raise ValueError("LoaderOrchestrator requires either default_loader or dataset_loader")

        if not isinstance(resolved_default, IDataLoader):
            raise TypeError("default_loader must implement IDataLoader")

        self._default_loader = resolved_default
        self._loader_overrides: Dict[DatasetId, IDataLoader] = {}

        if loader_overrides:
            for ds_id, loader in loader_overrides.items():
                if not isinstance(loader, IDataLoader):
                    raise TypeError(
                        f"Override loader for dataset '{ds_id.value}' must implement IDataLoader"
                    )
                self._loader_overrides[ds_id] = loader

    def register_loader(self, dataset_id: DatasetId, loader: IDataLoader) -> None:
        if not isinstance(loader, IDataLoader):
            raise TypeError(
                f"Loader for dataset '{dataset_id.value}' must implement IDataLoader"
            )
        self._loader_overrides[dataset_id] = loader
        logger.info(
            "Registered dedicated loader for dataset '%s': %s",
            dataset_id.value,
            type(loader).__name__,
        )

    def get_loader(self, dataset_id: DatasetId) -> IDataLoader:
        return self._loader_overrides.get(dataset_id, self._default_loader)

    def load(self, dataset_id: DatasetId) -> pl.DataFrame:
        loader = self.get_loader(dataset_id)

        logger.info(
            "Loading dataset '%s' via loader '%s'",
            dataset_id.value,
            type(loader).__name__,
        )

        df = loader.load_dataset(dataset_id)

        logger.debug(
            "Dataset '%s' loaded (rows=%s, cols=%s)",
            dataset_id.value,
            df.height,
            df.width,
        )

        return df

    def load_all(self, dataset_ids: list[DatasetId]) -> Dict[DatasetId, pl.DataFrame]:
        result: Dict[DatasetId, pl.DataFrame] = {}
        for ds_id in dataset_ids:
            result[ds_id] = self.load(ds_id)

        logger.info("Loaded %s datasets via LoaderOrchestrator", len(result))
        return result
