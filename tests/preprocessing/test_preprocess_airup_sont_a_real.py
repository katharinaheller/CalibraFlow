import polars as pl
from core.preprocessing.preprocessing_orchestrator import PreprocessingOrchestrator
from core.preprocessing.preprocessing_config import PREPROCESSOR_REGISTRY
from core.loaders.dataset_ids import DatasetId

def test_airup_sont_a_selects_and_renames_columns():
    df = pl.DataFrame({
        "pm1": [0.57],
        "pm25": [0.60],
        "pm10": [0.60],
        "sht_humid": [31.54],
        "sht_temp": [29.4],
        "CO": [-250.96],
        "NO": [-18.45],
        "NO2": [-5.44],
        "O3": [-9.56],
        "timestamp_hr": ["2024-11-13 00:00:00"],
        "RAW_OPC_Bin 0": [17.93],
        "RAW_ADC_CO_A": [334.31],
    })

    orch = PreprocessingOrchestrator(PREPROCESSOR_REGISTRY)
    out = orch.preprocess(DatasetId.AIRUP_SONT_A, df)

    # nur die "echten" columns bleiben
    assert set(out.columns) >= {
        "pm1", "pm25", "pm10",
        "humidity", "temperature",
        "CO", "NO", "NO2", "O3",
        "timestamp"
    }

    # renaming korrekt
    assert "humidity" in out.columns
    assert "temperature" in out.columns

    # timestamp korrekt geparsed
    assert out["timestamp"].dtype == pl.Datetime
