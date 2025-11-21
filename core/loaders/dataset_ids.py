from enum import Enum

class DatasetId(str, Enum):
    # synthetic datasets only
    AIR_QUALITY_REFERENCE = "air_quality_reference"
    AIR_QUALITY_RAW = "air_quality_raw"
    AIR_QUALITY_CALIBRATED = "air_quality_calibrated"
    NOISE_RAW = "noise_raw"
    NOISE_CALIBRATED = "noise_calibrated"
    WEATHER_RAW = "weather_raw"
    WEATHER_CALIBRATED = "weather_calibrated"
