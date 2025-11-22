from enum import Enum

class DatasetId(str, Enum):
    # logical dataset identifiers
    AIR_QUALITY_REFERENCE = "air_quality_reference"
    AIR_QUALITY_RAW = "air_quality_raw"
    AIR_QUALITY_CALIBRATED = "air_quality_calibrated"
    NOISE_RAW = "noise_raw"
    NOISE_CALIBRATED = "noise_calibrated"
    WEATHER_RAW = "weather_raw"
    WEATHER_CALIBRATED = "weather_calibrated"

    LUBW_MINUTE = "lubw_minute_data"
    AIRUP_SONT_A = "airup_sont_a_minute"
    AIRUP_SONT_C = "airup_sont_c_minute"
