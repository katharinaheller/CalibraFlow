from core.loaders.dataset_ids import DatasetId
from .synthetic_air_quality_preprocessor import SyntheticAirQualityPreprocessor
from .lubw_minute_preprocessor import LUBWMinutePreprocessor
from .airup_sensor_preprocessor import AirUpSensorPreprocessor

PREPROCESSOR_REGISTRY = {
    DatasetId.AIR_QUALITY_REFERENCE: SyntheticAirQualityPreprocessor(),
    DatasetId.AIR_QUALITY_RAW: SyntheticAirQualityPreprocessor(),
    DatasetId.AIR_QUALITY_CALIBRATED: SyntheticAirQualityPreprocessor(),
    DatasetId.NOISE_RAW: SyntheticAirQualityPreprocessor(),
    DatasetId.NOISE_CALIBRATED: SyntheticAirQualityPreprocessor(),
    DatasetId.WEATHER_RAW: SyntheticAirQualityPreprocessor(),
    DatasetId.WEATHER_CALIBRATED: SyntheticAirQualityPreprocessor(),

    DatasetId.LUBW_MINUTE: LUBWMinutePreprocessor(),

    DatasetId.AIRUP_SONT_A: AirUpSensorPreprocessor(),
    DatasetId.AIRUP_SONT_C: AirUpSensorPreprocessor(),
}
