import logging
from typing import Sequence
import polars as pl
from core.interfaces.IAnomalyDetector import IAnomalyDetector

logger = logging.getLogger(__name__)

class AnomalyOrchestrator:
    # orchestrates anomaly detection on preprocessed datasets

    def __init__(self, detector: IAnomalyDetector) -> None:
        # dependency injected detector implementation
        self._detector = detector

    def fit_on_reference(
        self,
        df_reference: pl.DataFrame,
        feature_columns: Sequence[str],
    ) -> None:
        # train anomaly detector on (assumed) mostly normal reference data
        logger.info("Fitting anomaly detector on reference dataset")
        self._detector.fit(df_reference, feature_columns)

    def run_detection(
        self,
        df_target: pl.DataFrame,
        threshold: float,
    ) -> pl.DataFrame:
        # run anomaly detection on target dataset and return annotated dataframe
        logger.info(
            "Running anomaly detection (threshold=%s) on target dataset",
            threshold,
        )
        result = self._detector.detect(df_target, threshold)
        return result
