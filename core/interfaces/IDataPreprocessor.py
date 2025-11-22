from abc import ABC, abstractmethod
import polars as pl

class IDataPreprocessor(ABC):
    # contract for all preprocessing components
    @abstractmethod
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        pass
