import polars as pl
from core.loaders.dataset_ids import DatasetId
from core.interfaces.IDataPreprocessor import IDataPreprocessor

class PreprocessingOrchestrator:
    def __init__(self, registry: dict[DatasetId, IDataPreprocessor]) -> None:
        self._registry = registry

    def preprocess(self, dataset_id: DatasetId, df: pl.DataFrame) -> pl.DataFrame:
        if dataset_id not in self._registry:
            raise ValueError(f"No preprocessor registered for dataset {dataset_id}")

        processor = self._registry[dataset_id]

        if not isinstance(processor, IDataPreprocessor):
            raise TypeError(
                f"Registered processor for {dataset_id} must implement IDataPreprocessor"
            )

        return processor.preprocess(df)
