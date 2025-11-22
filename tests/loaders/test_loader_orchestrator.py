import polars as pl
import pytest
from pathlib import Path

from core.loaders.csv_dataset_loader import CsvDatasetLoader
from core.loaders.loader_orchestrator import LoaderOrchestrator
from core.loaders.dataset_ids import DatasetId
from core.loaders.dataset_config import DATASET_REGISTRY


@pytest.fixture(scope="session")
def project_root() -> Path:
    # point to repository root (CalibraFlow/)
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def synthetic_dataset_ids() -> list[DatasetId]:
    return [
        DatasetId.AIR_QUALITY_REFERENCE,
        DatasetId.AIR_QUALITY_RAW,
        DatasetId.AIR_QUALITY_CALIBRATED,
        DatasetId.NOISE_RAW,
        DatasetId.NOISE_CALIBRATED,
        DatasetId.WEATHER_RAW,
        DatasetId.WEATHER_CALIBRATED,
    ]


@pytest.fixture(scope="session")
def orchestrator(project_root: Path) -> LoaderOrchestrator:
    loader = CsvDatasetLoader(project_root, registry=DATASET_REGISTRY)
    return LoaderOrchestrator(default_loader=loader)


def test_load_all_returns_expected_keys(orchestrator, synthetic_dataset_ids):
    result = orchestrator.load_all(synthetic_dataset_ids)
    assert set(result.keys()) == set(synthetic_dataset_ids)


def test_each_synthetic_dataset_is_non_empty_dataframe(orchestrator, synthetic_dataset_ids):
    result = orchestrator.load_all(synthetic_dataset_ids)
    for ds_id, df in result.items():
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert df.width > 0


def test_timestamp_column_is_datetime_where_configured(orchestrator, synthetic_dataset_ids):
    result = orchestrator.load_all(synthetic_dataset_ids)

    datasets_with_timestamp = {
        DatasetId.AIR_QUALITY_RAW,
        DatasetId.AIR_QUALITY_CALIBRATED,
        DatasetId.NOISE_RAW,
        DatasetId.NOISE_CALIBRATED,
        DatasetId.WEATHER_RAW,
        DatasetId.WEATHER_CALIBRATED,
    }

    for ds_id in synthetic_dataset_ids:
        if ds_id not in datasets_with_timestamp:
            continue

        df = result[ds_id]
        assert "timestamp" in df.columns
        assert df["timestamp"].dtype == pl.Datetime
