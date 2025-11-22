from abc import ABC, abstractmethod
from typing import Sequence
import polars as pl

class IAnomalyDetector(ABC):
    # contract for all anomaly detectors in the system

    @abstractmethod
    def fit(self, df: pl.DataFrame, feature_columns: Sequence[str]) -> None:
        # train the detector on (assumed) mostly normal data
        pass

    @abstractmethod
    def score(self, df: pl.DataFrame) -> pl.DataFrame:
        # compute anomaly scores for each row and return df with extra column 'anomaly_score'
        pass

    @abstractmethod
    def detect(self, df: pl.DataFrame, threshold: float) -> pl.DataFrame:
        # apply a threshold to 'anomaly_score' and return df with extra boolean column 'is_anomaly'
        pass
