import polars as pl
from pathlib import Path
from core.loaders.csv_dataset_loader import CsvDatasetLoader
from core.loaders.dataset_ids import DatasetId


def test_csv_loader_reads_single_file(tmp_path: Path, patched_registry):
    test_file = tmp_path / "file.csv"
    test_file.write_text(
        "timestamp,station_id,value\n"
        "2020-01-01 00:00:00,1,10\n"
    )

    patched_registry[DatasetId.AIR_QUALITY_RAW] = patched_registry[DatasetId.AIR_QUALITY_RAW].__class__(
        dataset_id=DatasetId.AIR_QUALITY_RAW,
        relative_path=test_file.relative_to(tmp_path),
        parse_dates=["timestamp"],
        rename_columns={"timestamp": "timestamp", "station_id": "station_id"},
        required_columns=["timestamp", "station_id"],
        dtypes=None,
    )

    # --- DI FIX ---
    loader = CsvDatasetLoader(tmp_path, registry=patched_registry)

    df = loader.load_dataset(DatasetId.AIR_QUALITY_RAW)

    assert df.height == 1
    assert df["value"][0] == 10
