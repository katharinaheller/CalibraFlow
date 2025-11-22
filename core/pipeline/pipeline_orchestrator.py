from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import logging
import polars as pl

from core.loaders.loader_orchestrator import LoaderOrchestrator
from core.loaders.dataset_ids import DatasetId
from core.preprocessing.preprocessing_orchestrator import PreprocessingOrchestrator
from core.preprocessing.preprocessing_config import PREPROCESSOR_REGISTRY

logger = logging.getLogger(__name__)


# pipeline execution phases
class PipelinePhase(str):
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    CALIBRATION = "calibration"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class PipelineResult:
    # raw loaded dataframe
    raw_loaded: Optional[pl.DataFrame] = None

    # preprocessed dataframe
    preprocessed: Optional[pl.DataFrame] = None

    # later phases
    features: Optional[pl.DataFrame] = None
    calibrated: Optional[pl.DataFrame] = None
    anomalies: Optional[pl.DataFrame] = None


class PipelineOrchestrator:
    """
    High-level orchestrator coordinating loading, preprocessing and later phases.
    Fully TDD aligned and modular.
    """

    def __init__(
        self,
        loader_orchestrator: LoaderOrchestrator,
        preprocessor_registry: Dict[DatasetId, object] = PREPROCESSOR_REGISTRY
    ) -> None:
        # dependency injection
        self._loader = loader_orchestrator
        self._preprocessors = PreprocessingOrchestrator(preprocessor_registry)

    # ---------------------------------------------------------
    #                     LOADING PHASE
    # ---------------------------------------------------------
    def _execute_loading(self, dataset_id: DatasetId) -> pl.DataFrame:
        # loading via LoaderOrchestrator
        logger.info("Pipeline phase: LOADING dataset '%s'", dataset_id.value)
        df = self._loader.load(dataset_id)
        logger.debug(
            "Loading phase complete for dataset '%s' (rows=%s, cols=%s)",
            dataset_id.value,
            df.height,
            df.width,
        )
        return df

    # ---------------------------------------------------------
    #                  PREPROCESSING PHASE
    # ---------------------------------------------------------
    def _execute_preprocessing(
        self,
        dataset_id: DatasetId,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        # preprocessing via PreprocessingOrchestrator
        logger.info("Pipeline phase: PREPROCESSING dataset '%s'", dataset_id.value)
        pre = self._preprocessors.preprocess(dataset_id, df)
        logger.debug(
            "Preprocessing phase complete for dataset '%s' (rows=%s, cols=%s)",
            dataset_id.value,
            pre.height,
            pre.width,
        )
        return pre

    # ---------------------------------------------------------
    #                 MAIN PIPELINE ENTRYPOINT
    # ---------------------------------------------------------
    def run(
        self,
        dataset_id: DatasetId,
        phases: Optional[list[str]] = None
    ) -> PipelineResult:
        # default to only loading
        if phases is None:
            phases = [PipelinePhase.LOADING]

        result = PipelineResult()

        # LOADING
        if PipelinePhase.LOADING in phases:
            result.raw_loaded = self._execute_loading(dataset_id)

        # PREPROCESSING
        if PipelinePhase.PREPROCESSING in phases:
            if result.raw_loaded is None:
                raise RuntimeError("PREPROCESSING requires LOADING to run first")
            result.preprocessed = self._execute_preprocessing(
                dataset_id,
                result.raw_loaded
            )

        # FEATURE ENGINEERING (placeholder)
        if PipelinePhase.FEATURE_ENGINEERING in phases:
            logger.warning("FEATURE_ENGINEERING phase selected but not implemented yet.")

        # CALIBRATION (placeholder)
        if PipelinePhase.CALIBRATION in phases:
            logger.warning("CALIBRATION phase selected but not implemented yet.")

        # ANOMALY DETECTION (placeholder)
        if PipelinePhase.ANOMALY_DETECTION in phases:
            logger.warning("ANOMALY_DETECTION phase selected but not implemented yet.")

        logger.info(
            "Pipeline run complete for dataset '%s' (executed phases: %s)",
            dataset_id.value,
            phases,
        )

        return result
