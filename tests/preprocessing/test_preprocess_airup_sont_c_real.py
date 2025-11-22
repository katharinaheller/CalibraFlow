import polars as pl
from core.preprocessing.preprocessing_orchestrator import PreprocessingOrchestrator
from core.preprocessing.preprocessing_config import PREPROCESSOR_REGISTRY
from core.loaders.dataset_ids import DatasetId

def test_airup_sont_c_timestamp_gps_and_column_filtering():
    df = pl.DataFrame({
        "pm1": [1.07],
        "pm25": [1.72],
        "pm10": [6.46],
        "sht_humid": [39.19],
        "sht_temp": [22.46],
        "CO": [339.03],
        "NO": [-66.9],
        "NO2": [-2.1],
        "O3": [14.72],
        "timestamp_gps": [1731487080.3068864],  # epoch seconds
        "RAW_OPC_Bin 3": [0.05],
        "lat": [23.1],
        "lon": [0.0],
    })

    orch = PreprocessingOrchestrator(PREPROCESSOR_REGISTRY)
    out = orch.preprocess(DatasetId.AIRUP_SONT_C, df)

    # Timestamp muss korrekt aus epoch gelesen werden
    assert out["timestamp"].dtype == pl.Datetime
    assert abs(out["timestamp"][0].timestamp() - 1731487080.306) < 5

    # RAW_OPC Spalten müssen entfernt sein
    assert "RAW_OPC_Bin 3" not in out.columns
