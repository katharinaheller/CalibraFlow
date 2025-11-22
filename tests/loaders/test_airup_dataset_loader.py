import polars as pl
from pathlib import Path
from core.loaders.airup_dataset_loader import AirUpDatasetLoader
from core.loaders.dataset_ids import DatasetId


def test_airup_loader_loads_and_concatenates_multiple_files(tmp_path: Path, patched_registry):
    data_dir = tmp_path / "sont_a"
    data_dir.mkdir()

    header = "pm1,pm25,pm10,sht_humid,sht_temp,CO,NO,NO2,O3,timestamp_hr\n"
    row1 = "1.0,2.0,3.0,40,20,1,2,3,4,2024-11-13 08:00:00\n"
    row2 = "1.1,2.1,3.1,41,21,1,2,3,4,2024-11-13 08:01:00\n"

    (data_dir / "airup_sont_a_avg_every_minute_data.log.2024-11-13").write_text(header + row1)
    (data_dir / "airup_sont_a_avg_every_minute_data.log.2024-11-14").write_text(header + row2)

    patched_registry[DatasetId.AIRUP_SONT_A] = patched_registry[DatasetId.AIRUP_SONT_A].__class__(
        dataset_id=DatasetId.AIRUP_SONT_A,
        relative_path=data_dir.relative_to(tmp_path),
        parse_dates=[],
        rename_columns={"sht_humid": "humidity", "sht_temp": "temperature"},
        required_columns=[
            "timestamp_hr", "pm1", "pm25", "pm10",
            "CO", "NO", "NO2", "O3", "humidity", "temperature"
        ],
        dtypes=None,
    )

    # --- DI FIX ---
    loader = AirUpDatasetLoader(tmp_path, registry=patched_registry)

    df = loader.load_dataset(DatasetId.AIRUP_SONT_A)

    assert df.height == 2
    assert set(df.columns) >= {"pm1", "pm25", "pm10", "humidity", "temperature", "timestamp_hr"}
    assert df["pm1"].to_list() == [1.0, 1.1]
    assert df["humidity"].to_list() == [40, 41]
