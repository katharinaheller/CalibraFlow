import polars as pl
from pathlib import Path
from core.loaders.csv_dataset_loader import CsvDatasetLoader
from core.loaders.airup_dataset_loader import AirUpDatasetLoader
from core.loaders.loader_orchestrator import LoaderOrchestrator
from core.loaders.dataset_ids import DatasetId


def test_full_loader_integration(tmp_path: Path, patched_registry):
    lubw_dir = tmp_path / "lubw"
    lubw_dir.mkdir()
    lubw_file = lubw_dir / "minute_data_lubw_full.csv"
    lubw_file.write_text(
        "datetime,NO2,O3,PM10,PM2p5,TEMP,RLF,p-Luft,NSCH,WIR,WIV\n"
        "2024-11-14 00:01:00,24,3,23,22,5,90,1012,0,180,1"
    )

    patched_registry[DatasetId.LUBW_MINUTE] = patched_registry[DatasetId.LUBW_MINUTE].__class__(
        dataset_id=DatasetId.LUBW_MINUTE,
        relative_path=lubw_file.relative_to(tmp_path),
        parse_dates=[],
        rename_columns={"datetime": "timestamp"},
        required_columns=["timestamp", "NO2", "O3", "PM10", "PM2p5", "TEMP", "RLF", "p-Luft", "NSCH", "WIR", "WIV"],
        dtypes=None,
    )

    a_dir = tmp_path / "sont_a"
    a_dir.mkdir()
    a_file = a_dir / "airup_sont_a_avg_every_minute_data.log.2024-11-13"
    a_file.write_text(
        "pm1,pm25,pm10,sht_humid,sht_temp,CO,NO,NO2,O3,timestamp_hr\n"
        "1,2,3,40,20,1,2,3,4,2024-11-13 10:00:00"
    )

    patched_registry[DatasetId.AIRUP_SONT_A] = patched_registry[DatasetId.AIRUP_SONT_A].__class__(
        dataset_id=DatasetId.AIRUP_SONT_A,
        relative_path=a_dir.relative_to(tmp_path),
        parse_dates=[],
        rename_columns={"sht_humid": "humidity", "sht_temp": "temperature"},
        required_columns=[
            "timestamp_hr", "pm1", "pm25", "pm10",
            "CO", "NO", "NO2", "O3", "humidity", "temperature"
        ],
        dtypes=None,
    )

    # --- DI FIX ---
    csv_loader = CsvDatasetLoader(tmp_path, registry=patched_registry)
    air_loader = AirUpDatasetLoader(tmp_path, registry=patched_registry)

    orch = LoaderOrchestrator(default_loader=csv_loader, loader_overrides={
        DatasetId.AIRUP_SONT_A: air_loader
    })

    df_lubw = orch.load(DatasetId.LUBW_MINUTE)
    df_air = orch.load(DatasetId.AIRUP_SONT_A)

    assert df_lubw.height == 1
    assert df_air.height == 1
    assert "timestamp" in df_lubw.columns
    assert df_air["pm1"][0] == 1
    assert df_air["humidity"][0] == 40
