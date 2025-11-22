# core/loaders/dataset_config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional
import polars as pl
from .dataset_ids import DatasetId

@dataclass(frozen=True)
class DatasetConfig:
    # describes how to read and normalize a dataset
    dataset_id: DatasetId
    relative_path: Path
    parse_dates: List[str]
    rename_columns: Mapping[str, str]
    required_columns: List[str]
    dtypes: Optional[Mapping[str, pl.DataType]] = None

    # extended loader settings
    delimiter: str = ","  # default CSV separator
    has_header: bool = True  # default header row
    encoding: str = "utf8"  # default encoding
    null_values: Optional[list[str]] = None  # optional null markers

# base folder for synthetic datasets
BASE_SYN = Path("data/syntetische_daten_heilbronn_2021_2023")

DATASET_REGISTRY: Dict[DatasetId, DatasetConfig] = {
    DatasetId.AIR_QUALITY_REFERENCE: DatasetConfig(
        dataset_id=DatasetId.AIR_QUALITY_REFERENCE,
        relative_path=BASE_SYN / "air_quality_reference_values_germany.csv",
        parse_dates=[],
        rename_columns={"pollutant": "pollutant"},
        required_columns=["pollutant"],
        dtypes=None,
    ),
    DatasetId.AIR_QUALITY_RAW: DatasetConfig(
        dataset_id=DatasetId.AIR_QUALITY_RAW,
        relative_path=BASE_SYN / "heilbronn_air_quality_2021_2023.csv",
        parse_dates=["timestamp"],
        rename_columns={"timestamp": "timestamp", "station_id": "station_id"},
        required_columns=["timestamp", "station_id"],
        dtypes=None,
    ),
    DatasetId.AIR_QUALITY_CALIBRATED: DatasetConfig(
        dataset_id=DatasetId.AIR_QUALITY_CALIBRATED,
        relative_path=BASE_SYN / "heilbronn_air_quality_2021_2023_calibrated.csv",
        parse_dates=["timestamp"],
        rename_columns={"timestamp": "timestamp", "station_id": "station_id"},
        required_columns=["timestamp", "station_id"],
        dtypes=None,
    ),
    DatasetId.NOISE_RAW: DatasetConfig(
        dataset_id=DatasetId.NOISE_RAW,
        relative_path=BASE_SYN / "heilbronn_noise_2021_2023.csv",
        parse_dates=["timestamp"],
        rename_columns={"timestamp": "timestamp", "sensor_id": "sensor_id"},
        required_columns=["timestamp", "sensor_id"],
        dtypes=None,
    ),
    DatasetId.NOISE_CALIBRATED: DatasetConfig(
        dataset_id=DatasetId.NOISE_CALIBRATED,
        relative_path=BASE_SYN / "heilbronn_noise_2021_2023_calibrated.csv",
        parse_dates=["timestamp"],
        rename_columns={"timestamp": "timestamp", "sensor_id": "sensor_id"},
        required_columns=["timestamp", "sensor_id"],
        dtypes=None,
    ),
    DatasetId.WEATHER_RAW: DatasetConfig(
        dataset_id=DatasetId.WEATHER_RAW,
        relative_path=BASE_SYN / "heilbronn_weather_2021_2023.csv",
        parse_dates=["timestamp"],
        rename_columns={"timestamp": "timestamp", "station_id": "station_id"},
        required_columns=["timestamp", "station_id"],
        dtypes=None,
    ),
    DatasetId.WEATHER_CALIBRATED: DatasetConfig(
        dataset_id=DatasetId.WEATHER_CALIBRATED,
        relative_path=BASE_SYN / "heilbronn_weather_2021_2023_calibrated.csv",
        parse_dates=["timestamp"],
        rename_columns={"timestamp": "timestamp", "station_id": "station_id"},
        required_columns=["timestamp", "station_id"],
        dtypes=None,
    ),
}
