import polars as pl
from core.preprocessing.preprocessing_orchestrator import PreprocessingOrchestrator
from core.preprocessing.synthetic_air_quality_preprocessor import SyntheticAirQualityPreprocessor
from core.loaders.dataset_ids import DatasetId

def test_orchestrator_routing():
    registry = {
        DatasetId.AIR_QUALITY_RAW: SyntheticAirQualityPreprocessor()
    }
    orch = PreprocessingOrchestrator(registry)

    df = pl.DataFrame({"timestamp": ["2021-01-01"], "station_id": ["A"]})
    out = orch.preprocess(DatasetId.AIR_QUALITY_RAW, df)

    assert isinstance(out, pl.DataFrame)
