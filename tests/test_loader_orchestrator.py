from pathlib import Path

import polars as pl
import pytest

from core.loaders.csv_dataset_loader import CsvDatasetLoader
from core.loaders.loader_orchestrator import LoaderOrchestrator
from core.loaders.dataset_ids import DatasetId


@pytest.fixture(scope="session")
def project_root() -> Path:
    # resolve project root assuming tests/ is directly under root
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_dataset_ids() -> list[DatasetId]:
    # list of all synthetic datasets that are backed by CSVs in DATASET_REGISTRY
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
    # construct orchestrator with real CsvDatasetLoader pointing at project root
    loader = CsvDatasetLoader(base_path=project_root)
    return LoaderOrchestrator(dataset_loader=loader)


def test_load_all_returns_expected_keys(
    orchestrator: LoaderOrchestrator,
    synthetic_dataset_ids: list[DatasetId],
) -> None:
    # ensures that all configured synthetic datasets are loaded and keyed correctly
    result = orchestrator.load_all(synthetic_dataset_ids)

    assert set(result.keys()) == set(synthetic_dataset_ids), (
        "Loaded dataset keys do not match the expected synthetic DatasetIds"
    )


def test_each_synthetic_dataset_is_non_empty_dataframe(
    orchestrator: LoaderOrchestrator,
    synthetic_dataset_ids: list[DatasetId],
) -> None:
    # ensures each synthetic dataset is a non empty polars DataFrame
    result = orchestrator.load_all(synthetic_dataset_ids)

    for ds_id, df in result.items():
        assert isinstance(df, pl.DataFrame), (
            f"Dataset '{ds_id.value}' is not a polars DataFrame"
        )
        assert df.height > 0, (
            f"Dataset '{ds_id.value}' is unexpectedly empty"
        )
        assert df.width > 0, (
            f"Dataset '{ds_id.value}' has no columns"
        )


def test_timestamp_column_is_datetime_where_configured(
    orchestrator: LoaderOrchestrator,
    synthetic_dataset_ids: list[DatasetId],
) -> None:
    # ensures timestamp columns are parsed to Datetime for time series datasets
    result = orchestrator.load_all(synthetic_dataset_ids)

    # datasets that should have a timestamp parsed via parse_dates
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
            # reference table has no timestamp column by design
            continue

        df = result[ds_id]
        assert "timestamp" in df.columns, (
            f"Dataset '{ds_id.value}' is expected to have a 'timestamp' column"
        )
        assert df["timestamp"].dtype == pl.Datetime, (
            f"'timestamp' column in dataset '{ds_id.value}' is not of type Datetime "
            f"(actual dtype: {df['timestamp'].dtype})"
        )
