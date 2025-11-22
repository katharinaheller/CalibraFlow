from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl
from core.loaders.dataset_ids import DatasetId

class IDataLoader(ABC):
    # contract for all dataset loaders
    @abstractmethod
    def load_dataset(self, dataset_id: DatasetId) -> pl.DataFrame:
        pass
