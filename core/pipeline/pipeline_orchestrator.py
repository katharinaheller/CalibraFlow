from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import logging
import polars as pl

from core.loaders.loader_orchestrator import LoaderOrchestrator
from core.loaders.dataset_ids import DatasetId

logger = logging.getLogger(__name__)


# pipeline execution phases
class PipelinePhase(str):
    # only loading for now
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    CALIBRATION = "calibration"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class PipelineResult:
    # represents intermediate and final artifacts of the pipeline
    raw_loaded: Optional[pl.DataFrame] = None

    # placeholders for later phases
    preprocessed: Optional[pl.DataFrame] = None
    features: Optional[pl.DataFrame] = None
    calibrated: Optional[pl.DataFrame] = None
    anomalies: Optional[pl.DataFrame] = None


class PipelineOrchestrator:
    """
    High-level pipeline orchestrator coordinating all phases of the sensor data workflow.

    Currently only LOADING phase is fully implemented.
    Other phases are prepared but inactive.
    """

    def __init__(
        self,
        loader_orchestrator: LoaderOrchestrator,
    ) -> None:
        # dependency injection: loader orchestrator of all datasets
        self._loader = loader_orchestrator

    # ---------------------------------------------------------
    #                     LOADING PHASE
    # ---------------------------------------------------------
    def _execute_loading(self, dataset_id: DatasetId) -> pl.DataFrame:
        """
        Executes the loading phase using the LoaderOrchestrator.
        """
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
    #                 MAIN PIPELINE ENTRYPOINT
    # ---------------------------------------------------------
    def run(
        self,
        dataset_id: DatasetId,
        phases: Optional[list[str]] = None
    ) -> PipelineResult:
        """
        Execute selected pipeline phases.
        Default: only loading.
        """
        if phases is None:
            phases = [PipelinePhase.LOADING]

        result = PipelineResult()

        # LOADING
        if PipelinePhase.LOADING in phases:
            result.raw_loaded = self._execute_loading(dataset_id)

        # PREPROCESSING (not yet implemented)
        if PipelinePhase.PREPROCESSING in phases:
            logger.warning("PREPROCESSING phase selected but not implemented yet.")

        # FEATURE ENGINEERING (not yet implemented)
        if PipelinePhase.FEATURE_ENGINEERING in phases:
            logger.warning("FEATURE_ENGINEERING phase selected but not implemented yet.")

        # CALIBRATION (not yet implemented)
        if PipelinePhase.CALIBRATION in phases:
            logger.warning("CALIBRATION phase selected but not implemented yet.")

        # ANOMALY DETECTION (not yet implemented)
        if PipelinePhase.ANOMALY_DETECTION in phases:
            logger.warning("ANOMALY_DETECTION phase selected but not implemented yet.")

        logger.info(
            "Pipeline run complete for dataset '%s' (executed phases: %s)",
            dataset_id.value,
            phases,
        )

        return result
