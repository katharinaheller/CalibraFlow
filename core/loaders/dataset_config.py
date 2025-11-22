
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional
import polars as pl
from .dataset_ids import DatasetId


@dataclass(frozen=True)
class DatasetConfig:
    # describes how to read and normalize a dataset
    dataset_id: DatasetId
    relative_path: Path  # path relative to project base_path
    parse_dates: List[str]  # raw columns that should be parsed as datetime
    rename_columns: Mapping[str, str]  # mapping raw -> normalized column names
    required_columns: List[str]  # columns that must be present after rename
    dtypes: Optional[Mapping[str, pl.DataType]] = None  # optional dtype casting

    # extended loader settings
    delimiter: str = ","  # default CSV separator
    has_header: bool = True  # default header row
    encoding: str = "utf8"  # default encoding
    null_values: Optional[list[str]] = None  # optional null markers


# base folder for synthetic datasets
BASE_SYN = Path("data/syntetische_daten_heilbronn_2021_2023")

# base folder for real comparison campaign datasets (HHN / LUBW / AirUp)
BASE_HHN = Path("data/hhn_daten_vergleichskampagne_20241115-20250205")


DATASET_REGISTRY: Dict[DatasetId, DatasetConfig] = {
    DatasetId.AIR_QUALITY_REFERENCE: DatasetConfig(
        dataset_id=DatasetId.AIR_QUALITY_REFERENCE,
        relative_path=BASE_SYN / "air_quality_reference_values_germany.csv",
        parse_dates=[],  # timestamps are not relevant here
        rename_columns={
            "pollutant": "pollutant",
        },
        required_columns=[
            "pollutant",
        ],
        dtypes=None,
    ),
    DatasetId.AIR_QUALITY_RAW: DatasetConfig(
        dataset_id=DatasetId.AIR_QUALITY_RAW,
        relative_path=BASE_SYN / "heilbronn_air_quality_2021_2023.csv",
        parse_dates=[
            "timestamp",
        ],
        rename_columns={
            "timestamp": "timestamp",
            "station_id": "station_id",
        },
        required_columns=[
            "timestamp",
            "station_id",
        ],
        dtypes=None,
    ),
    DatasetId.AIR_QUALITY_CALIBRATED: DatasetConfig(
        dataset_id=DatasetId.AIR_QUALITY_CALIBRATED,
        relative_path=BASE_SYN / "heilbronn_air_quality_2021_2023_calibrated.csv",
        parse_dates=[
            "timestamp",
        ],
        rename_columns={
            "timestamp": "timestamp",
            "station_id": "station_id",
        },
        required_columns=[
            "timestamp",
            "station_id",
        ],
        dtypes=None,
    ),
    DatasetId.NOISE_RAW: DatasetConfig(
        dataset_id=DatasetId.NOISE_RAW,
        relative_path=BASE_SYN / "heilbronn_noise_2021_2023.csv",
        parse_dates=[
            "timestamp",
        ],
        rename_columns={
            "timestamp": "timestamp",
            "sensor_id": "sensor_id",
        },
        required_columns=[
            "timestamp",
            "sensor_id",
        ],
        dtypes=None,
    ),
    DatasetId.NOISE_CALIBRATED: DatasetConfig(
        dataset_id=DatasetId.NOISE_CALIBRATED,
        relative_path=BASE_SYN / "heilbronn_noise_2021_2023_calibrated.csv",
        parse_dates=[
            "timestamp",
        ],
        rename_columns={
            "timestamp": "timestamp",
            "sensor_id": "sensor_id",
        },
        required_columns=[
            "timestamp",
            "sensor_id",
        ],
        dtypes=None,
    ),
    DatasetId.WEATHER_RAW: DatasetConfig(
        dataset_id=DatasetId.WEATHER_RAW,
        relative_path=BASE_SYN / "heilbronn_weather_2021_2023.csv",
        parse_dates=[
            "timestamp",
        ],
        rename_columns={
            "timestamp": "timestamp",
            "station_id": "station_id",
        },
        required_columns=[
            "timestamp",
            "station_id",
        ],
        dtypes=None,
    ),
    DatasetId.WEATHER_CALIBRATED: DatasetConfig(
        dataset_id=DatasetId.WEATHER_CALIBRATED,
        relative_path=BASE_SYN / "heilbronn_weather_2021_2023_calibrated.csv",
        parse_dates=[
            "timestamp",
        ],
        rename_columns={
            "timestamp": "timestamp",
            "station_id": "station_id",
        },
        required_columns=[
            "timestamp",
            "station_id",
        ],
        dtypes=None,
    ),

    # real data: LUBW minute data (single CSV file)
    DatasetId.LUBW_MINUTE: DatasetConfig(
        dataset_id=DatasetId.LUBW_MINUTE,
        relative_path=BASE_HHN / "lubw" / "minute_data_lubw_full.csv",
        parse_dates=[],  # timestamp parsing is handled by the LUBW preprocessor
        rename_columns={
            # normalize to the standard timestamp column name
            "datetime": "timestamp",
        },
        # required columns after rename
        required_columns=[
            "timestamp",
            "NO2",
            "O3",
            "PM10",
            "PM2p5",
            "TEMP",
            "RLF",
            "p-Luft",
            "NSCH",
            "WIR",
            "WIV",
        ],
        dtypes=None,
    ),

    # real data: AirUp sensor SONT A (directory with multiple daily log files)
    DatasetId.AIRUP_SONT_A: DatasetConfig(
        dataset_id=DatasetId.AIRUP_SONT_A,
        # points to the directory containing the daily log files
        relative_path=BASE_HHN / "sont_a",
        parse_dates=[],  # timestamp parsing is handled in the AirUp preprocessor
        rename_columns={
            # unify environmental measurements with synthetic datasets
            "sht_humid": "humidity",
            "sht_temp": "temperature",
        },
        # required columns after rename
        required_columns=[
            "timestamp_hr",  # human readable timestamp
            "pm1",
            "pm25",
            "pm10",
            "CO",
            "NO",
            "NO2",
            "O3",
            "humidity",
            "temperature",
        ],
        dtypes=None,
    ),

    # real data: AirUp sensor SONT C (directory with multiple daily log files)
    DatasetId.AIRUP_SONT_C: DatasetConfig(
        dataset_id=DatasetId.AIRUP_SONT_C,
        # points to the directory containing the daily log files
        relative_path=BASE_HHN / "sont_c",
        parse_dates=[],  # timestamp parsing is handled in the AirUp preprocessor
        rename_columns={
            "sht_humid": "humidity",
            "sht_temp": "temperature",
        },
        required_columns=[
            "timestamp_hr",
            "pm1",
            "pm25",
            "pm10",
            "CO",
            "NO",
            "NO2",
            "O3",
            "humidity",
            "temperature",
        ],
        dtypes=None,
    ),
}
