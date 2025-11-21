from pathlib import Path
import polars as pl
from core.loaders.loader_orchestrator import LoaderOrchestrator
from core.loaders.csv_dataset_loader import CsvDatasetLoader
from core.loaders.dataset_ids import DatasetId


def test_load_all_synthetic_returns_expected_keys(project_root: Path):
    # ensures that all synthetic datasets defined in the registry are returned
    orchestrator = LoaderOrchestrator(CsvDatasetLoader(base_path=project_root))
    result = orchestrator.load_all_synthetic()

    expected_keys = {
        DatasetId.AIR_QUALITY_REFERENCE,
        DatasetId.AIR_QUALITY_RAW,
        DatasetId.AIR_QUALITY_CALIBRATED,
        DatasetId.NOISE_RAW,
        DatasetId.NOISE_CALIBRATED,
        DatasetId.WEATHER_RAW,
        DatasetId.WEATHER_CALIBRATED,
    }

    assert set(result.keys()) == expected_keys


def test_each_synthetic_dataset_is_non_empty_dataframe(project_root: Path):
    # ensures each dataset is a non empty polars DataFrame
    orchestrator = LoaderOrchestrator(CsvDatasetLoader(base_path=project_root))
    result = orchestrator.load_all_synthetic()

    for ds_id, df in result.items():
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

        # only time series datasets must have a timestamp column
        if ds_id != DatasetId.AIR_QUALITY_REFERENCE:
            assert "timestamp" in df.columns


def test_timestamp_column_is_datetime(project_root: Path):
    # ensures timestamp columns are parsed correctly for time series datasets
    orchestrator = LoaderOrchestrator(CsvDatasetLoader(base_path=project_root))
    result = orchestrator.load_all_synthetic()

    for ds_id, df in result.items():
        if "timestamp" in df.columns:
            assert str(df.schema["timestamp"]).startswith("Datetime")
